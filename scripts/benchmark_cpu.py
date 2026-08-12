"""Phase 1: measure real CPU training throughput on a GitHub Actions runner.

Every performance figure in ``docs/BASE_MODEL_PLAN.md`` is an estimate until this runs.
It sweeps model size x dtype x batch size, timing forward + backward + optimizer step on
synthetic tokens, and reports tokens/sec and effective GFLOP/s.

Synthetic data is deliberate: this isolates *compute* from the data pipeline, so the number
is an upper bound on what the trainer could reach. The gap between this and the trainer's
real throughput is the pipeline's overhead, which is itself worth knowing.

Primary hypothesis under test: DTYPE=bfloat16 is why production sits at 1 tok/s. Without
AVX512-BF16 or AMX, PyTorch emulates bf16 on CPU and it runs *slower* than fp32.

Usage:
    python scripts/benchmark_cpu.py                     # full sweep
    python scripts/benchmark_cpu.py --quick             # 25M only, short budget
    python scripts/benchmark_cpu.py --include-baseline  # add the 494M control (slow)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch

# How a child process hands its measurement back to the parent.
RESULT_PREFIX = "RESULT_JSON:"
CONTROL_NAME = "494M-control"

# Saved tensors per layer kept alive for the backward pass, as a multiple of one
# (batch x block x hidden) activation. Calibrated against observed RSS on the 4-core runner;
# deliberately on the high side, since overestimating costs a skipped data point while
# underestimating costs the entire job.
ACT_TENSORS_PER_LAYER = 12

# Configs predicted above this are skipped. The runner has 15 GB, and systemd-oomd starts
# killing well before that, so this leaves real headroom for the OS and the runner agent.
DEFAULT_MEM_BUDGET_GB = 10.5

# FLOPs per token for forward + backward is ~6N (2N forward, 4N backward). Attention adds a
# term that is small at these depths and context lengths; we note it rather than model it.
FLOPS_PER_PARAM_PER_TOKEN = 6

# Candidate shapes. Names are targets; actual parameter counts are measured and reported,
# since the config -> param-count mapping is what we are trying to pin down.
SHAPES: dict[str, dict[str, int]] = {
    "25M": dict(hidden=384, layers=12, heads=6, kv_heads=2, ffn=1024),
    "57M": dict(hidden=512, layers=16, heads=8, kv_heads=2, ffn=1536),
    "126M": dict(hidden=768, layers=18, heads=12, kv_heads=4, ffn=2048),
}

# The production model today, as a control: reproducing ~1 tok/s here confirms the diagnosis.
BASELINE_SHAPE = dict(hidden=896, layers=24, heads=14, kv_heads=2, ffn=4864)

VOCAB_SIZE = 16384
BASELINE_VOCAB = 151936


@dataclass
class Result:
    shape: str
    params: int
    dtype: str
    batch_size: int
    block_size: int
    steps_timed: int
    tokens: int
    seconds: float
    tokens_per_sec: float
    gflops: float
    peak_rss_gb: float
    status: str = "ok"
    note: str = ""


def cpu_info() -> dict:
    """Capture what the runner actually is, since the bf16 question turns on CPU flags."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith("model name"):
                info["model_name"] = line.split(":", 1)[1].strip()
                break
        flags = ""
        for line in text.splitlines():
            if line.startswith("flags"):
                flags = line.split(":", 1)[1]
                break
        # These decide whether bf16 is native or emulated.
        for flag in ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_tile"):
            info[f"has_{flag}"] = f" {flag} " in f" {flags} "
    except OSError:
        pass
    try:
        info["mkldnn_available"] = torch.backends.mkldnn.is_available()
    except AttributeError:
        pass
    return info


def rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**3)
    except Exception:  # noqa: BLE001 — diagnostics must never break the benchmark
        return 0.0


def build_model(shape: dict[str, int], vocab: int, block: int, dtype: torch.dtype):
    """A Llama-shaped decoder: RMSNorm, SwiGLU, RoPE, GQA, tied embeddings."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=shape["hidden"],
        num_hidden_layers=shape["layers"],
        num_attention_heads=shape["heads"],
        num_key_value_heads=shape["kv_heads"],
        intermediate_size=shape["ffn"],
        max_position_embeddings=block,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg)
    return model.to(dtype=dtype).train()


def run_config(
    name: str,
    shape: dict[str, int],
    vocab: int,
    dtype_name: str,
    batch_size: int,
    block_size: int,
    budget_s: float,
    max_steps: int,
) -> Result:
    """Time one (shape, dtype, batch) point. Steps run until the budget elapses."""
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    empty = Result(
        shape=name,
        params=0,
        dtype=dtype_name,
        batch_size=batch_size,
        block_size=block_size,
        steps_timed=0,
        tokens=0,
        seconds=0.0,
        tokens_per_sec=0.0,
        gflops=0.0,
        peak_rss_gb=0.0,
    )

    try:
        model = build_model(shape, vocab, block_size, dtype)
        params = sum(p.numel() for p in model.parameters())
        empty.params = params
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = torch.randint(0, vocab, (batch_size, block_size), dtype=torch.long)

        def one_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            optimizer.step()

        tokens_per_step = batch_size * block_size

        # Warmup: the first step pays lazy allocation and oneDNN primitive selection.
        # Time it, because it also tells us whether a second step is affordable — a step
        # cannot be interrupted once started, so a config slower than the budget would
        # otherwise overrun it without bound. (bf16 at batch 32 takes ~20 min/step on a
        # runner without AVX512-BF16, which is what timed out the first sweep.)
        warm_start = time.perf_counter()
        one_step()
        warm_elapsed = time.perf_counter() - warm_start

        if warm_elapsed >= budget_s:
            steps, elapsed = 1, warm_elapsed
            empty.note = "warmup-only; includes one-time init, so a slight underestimate"
        else:
            steps = 0
            start = time.perf_counter()
            while steps < max_steps:
                one_step()
                steps += 1
                if time.perf_counter() - start >= budget_s:
                    break
            elapsed = time.perf_counter() - start

        tokens = steps * tokens_per_step
        tps = tokens / elapsed if elapsed > 0 else 0.0
        empty.steps_timed = steps
        empty.tokens = tokens
        empty.seconds = round(elapsed, 2)
        empty.tokens_per_sec = round(tps, 2)
        empty.gflops = round(FLOPS_PER_PARAM_PER_TOKEN * params * tps / 1e9, 2)
        empty.peak_rss_gb = round(rss_gb(), 2)
        return empty

    except (RuntimeError, MemoryError) as exc:
        msg = str(exc)
        oom = "out of memory" in msg.lower() or isinstance(exc, MemoryError)
        empty.status = "oom" if oom else "error"
        empty.note = msg.split("\n")[0][:200]
        empty.peak_rss_gb = round(rss_gb(), 2)
        return empty


def estimate_peak_gb(
    shape: dict[str, int], vocab: int, batch: int, block: int, dtype_name: str
) -> float:
    """Predict peak memory for a config, so we can decline to run the ones that die.

    This has to be predictive rather than defensive. When a config exhausts the runner,
    systemd-oomd SIGTERMs the whole cgroup — parent and child together — so neither an
    ``except`` nor subprocess isolation survives it. Sweeps 2 and 3 both died at
    126M/batch32 for exactly this reason (exit 143).

    Three terms: optimizer state (weights + grads + AdamW's two fp32 moments), saved
    activations for the backward pass, and the vocab-wide logits, which are materialized in
    fp32 for the loss and are far from negligible — at batch 32 they alone are ~2 GB.
    """
    bytes_per = 2 if dtype_name == "bf16" else 4
    hidden, layers = shape["hidden"], shape["layers"]

    params = vocab * hidden  # tied embeddings, counted once
    params += layers * (
        2 * hidden * hidden  # q_proj, o_proj
        + 2 * hidden * (shape["kv_heads"] * (hidden // shape["heads"]))  # k_proj, v_proj
        + 3 * hidden * shape["ffn"]  # SwiGLU gate/up/down
    )

    state = params * (2 * bytes_per + 8)
    activations = batch * block * hidden * layers * bytes_per * ACT_TENSORS_PER_LAYER
    logits = batch * block * vocab * 4 * 2  # fp32 logits plus their gradient
    return (state + activations + logits) / 1024**3


def parse_spec(spec: str) -> tuple[str, dict[str, int], int, str, int, int]:
    """Decode ``name:dtype:batch:block`` into the arguments ``run_config`` needs."""
    name, dtype_name, batch_raw, block_raw = spec.split(":")
    if name == CONTROL_NAME:
        shape, vocab = BASELINE_SHAPE, BASELINE_VOCAB
    else:
        shape, vocab = SHAPES[name], VOCAB_SIZE
    return name, shape, vocab, dtype_name, int(batch_raw), int(block_raw)


def run_spec_isolated(
    spec: str, budget_s: float, max_steps: int, mem_budget_gb: float = DEFAULT_MEM_BUDGET_GB
) -> Result:
    """Run one config in a child process and return its Result.

    Isolation is not optional here. A config that exhausts the runner's 15 GB is
    SIGKILLed by the kernel OOM killer, which no ``except`` can intercept — in sweep 2
    that killed the whole job at 126M/batch32 and cost every result after it. In a child,
    the same kill costs one data point and the sweep continues.
    """
    name, shape, vocab, dtype_name, batch_size, block_size = parse_spec(spec)
    label = f"{name}/{dtype_name}/batch{batch_size}"

    def blank(status: str, note: str) -> Result:
        return Result(
            shape=name,
            params=0,
            dtype=dtype_name,
            batch_size=batch_size,
            block_size=block_size,
            steps_timed=0,
            tokens=0,
            seconds=0.0,
            tokens_per_sec=0.0,
            gflops=0.0,
            peak_rss_gb=0.0,
            status=status,
            note=note,
        )

    predicted = estimate_peak_gb(shape, vocab, batch_size, block_size, dtype_name)
    if predicted > mem_budget_gb:
        note = f"predicted {predicted:.1f} GB peak > {mem_budget_gb:.1f} GB budget"
        print(f"  [SKIP] {label}: {note}", flush=True)
        return blank("skipped-memory", note)

    print(f"  [RUN ] {label} (est. {predicted:.1f} GB) ...", flush=True)

    # The warmup guard in run_config bounds a config at roughly two steps, but a pathological
    # shape could still stall; cap it so one config cannot consume the whole job.
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--single",
                spec,
                "--budget",
                str(budget_s),
                "--max-steps",
                str(max_steps),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(900.0, budget_s * 20),
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIME] {label}: exceeded per-config wall clock", flush=True)
        return blank("timeout", "exceeded per-config wall clock")

    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result = Result(**json.loads(line[len(RESULT_PREFIX) :]))
            if result.status == "ok":
                print(
                    f"  [DONE] {label}: {result.tokens_per_sec:,.1f} tok/s, "
                    f"{result.gflops:,.1f} GFLOP/s, {result.steps_timed} steps, "
                    f"{result.peak_rss_gb:.1f} GB RSS",
                    flush=True,
                )
            else:
                print(f"  [{result.status.upper():4}] {label}: {result.note}", flush=True)
            return result

    # No result line: the child died before it could report. Negative return codes are
    # signals, and -9 (SIGKILL) from the kernel means it ran the runner out of memory.
    killed = proc.returncode == -9 or proc.returncode == 137
    status = "oom-killed" if killed else "error"
    note = (
        "SIGKILL — exceeded runner memory"
        if killed
        else (proc.stderr.strip().splitlines() or ["no output"])[-1][:200]
    )
    print(f"  [{status.upper():4}] {label}: {note}", flush=True)
    return blank(status, note)


def markdown_table(results: list[Result]) -> str:
    header = (
        "| Shape | Params | dtype | Batch | Steps | tok/s | GFLOP/s | Peak RSS | Status |\n"
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    rows = []
    for r in results:
        tps = f"{r.tokens_per_sec:,.1f}" if r.status == "ok" else "—"
        gf = f"{r.gflops:,.1f}" if r.status == "ok" else "—"
        rows.append(
            f"| {r.shape} | {r.params / 1e6:,.1f}M | {r.dtype} | {r.batch_size} | "
            f"{r.steps_timed} | {tps} | {gf} | {r.peak_rss_gb:.1f} GB | {r.status} |"
        )
    return header + "\n".join(rows)


def summarize(results: list[Result]) -> list[str]:
    """Answer the two questions the sweep exists to settle."""
    lines: list[str] = []
    ok = [r for r in results if r.status == "ok" and r.tokens_per_sec > 0]
    if not ok:
        return ["No configuration completed successfully."]

    # Q1: is bf16 slower than fp32 on this hardware?
    pairs = []
    for r in ok:
        if r.dtype != "fp32":
            continue
        match = next(
            (
                o
                for o in ok
                if o.dtype == "bf16" and o.shape == r.shape and o.batch_size == r.batch_size
            ),
            None,
        )
        if match:
            pairs.append((r, match))
    if pairs:
        ratios = [f32.tokens_per_sec / bf.tokens_per_sec for f32, bf in pairs if bf.tokens_per_sec]
        if ratios:
            mean = sum(ratios) / len(ratios)
            verdict = "SLOWER" if mean > 1.05 else ("faster" if mean < 0.95 else "comparable")
            lines.append(
                f"bf16 is {verdict} than fp32 on this runner "
                f"(fp32/bf16 = {mean:.2f}x across {len(ratios)} matched pairs)."
            )

    # Q2: how much does batching buy?
    for shape in SHAPES:
        pts = sorted(
            (r for r in ok if r.shape == shape and r.dtype == "fp32"),
            key=lambda r: r.batch_size,
        )
        if len(pts) >= 2 and pts[0].tokens_per_sec:
            gain = pts[-1].tokens_per_sec / pts[0].tokens_per_sec
            lines.append(
                f"{shape} fp32: batch {pts[0].batch_size} -> {pts[-1].batch_size} "
                f"gives {gain:.1f}x ({pts[0].tokens_per_sec:,.0f} -> "
                f"{pts[-1].tokens_per_sec:,.0f} tok/s)."
            )

    # Q3: what does the best 25M config imply for the schedule?
    best = max((r for r in ok if r.shape == "25M"), key=lambda r: r.tokens_per_sec, default=None)
    if best:
        monthly = best.tokens_per_sec * 1.17e6
        months = 500e6 / monthly if monthly else float("inf")
        lines.append(
            f"Best 25M config: {best.dtype}/batch {best.batch_size} at "
            f"{best.tokens_per_sec:,.0f} tok/s -> ~{monthly / 1e6:,.0f}M tokens/month -> "
            f"Chinchilla (500M tok) in ~{months:.1f} months."
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=40.0, help="Seconds per configuration")
    parser.add_argument("--max-steps", type=int, default=200, help="Step cap per configuration")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--quick", action="store_true", help="25M / batch 1,8 only")
    # bf16 defaults off: the 2026-08-11 sweep measured it at 20x slower than fp32 on the
    # runner's EPYC 7763 (no avx512_bf16, no amx_bf16), so it is a settled question and
    # sweeping it just burns the time budget. Pass --dtypes fp32 bf16 to re-check on new hardware.
    parser.add_argument("--dtypes", nargs="+", default=["fp32"], choices=["fp32", "bf16"])
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Add the 494M bf16 batch-1 control (slow: several minutes)",
    )
    parser.add_argument("--out", default="docs/benchmarks/latest.json")
    parser.add_argument(
        "--mem-budget-gb",
        type=float,
        default=DEFAULT_MEM_BUDGET_GB,
        help="Skip configs predicted to exceed this peak memory",
    )
    parser.add_argument(
        "--single",
        metavar="NAME:DTYPE:BATCH:BLOCK",
        help="Internal: measure one config and emit its Result as JSON. Used by the parent "
        "sweep so an OOM kill costs one data point instead of the whole job.",
    )
    args = parser.parse_args()

    threads = os.cpu_count() or 1
    torch.set_num_threads(threads)

    if args.single:
        name, shape, vocab, dtype_name, batch_size, block_size = parse_spec(args.single)
        result = run_config(
            name, shape, vocab, dtype_name, batch_size, block_size, args.budget, args.max_steps
        )
        print(RESULT_PREFIX + json.dumps(asdict(result)), flush=True)
        return 0

    info = cpu_info()
    print("=" * 78)
    print("  Meridian Phase 1 — CPU training throughput")
    print("=" * 78)
    for key, value in info.items():
        print(f"  {key:>18}: {value}")
    print()

    shapes = ["25M"] if args.quick else list(SHAPES)
    batches = [1, 8] if args.quick else args.batch_sizes

    specs = [
        f"{name}:{dtype_name}:{batch_size}:{args.block_size}"
        for name in shapes
        for dtype_name in args.dtypes
        for batch_size in batches
    ]
    if args.include_baseline:
        specs.append(f"{CONTROL_NAME}:bf16:1:256")

    results = [
        run_spec_isolated(spec, args.budget, args.max_steps, args.mem_budget_gb) for spec in specs
    ]

    table = markdown_table(results)
    findings = summarize(results)

    print()
    print(table)
    print()
    print("  Findings")
    print("  " + "-" * 40)
    for line in findings:
        print(f"  - {line}")
    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": info,
        "settings": {
            "block_size": args.block_size,
            "budget_s": args.budget,
            "max_steps": args.max_steps,
            "vocab_size": VOCAB_SIZE,
        },
        "results": [asdict(r) for r in results],
        "findings": findings,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Wrote {args.out}")

    # Surface the table in the Actions run summary rather than burying it in logs.
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write("## Phase 1 — CPU training throughput\n\n")
            fh.write(f"`{info.get('model_name', 'unknown CPU')}` · ")
            fh.write(f"{info.get('cpu_count')} vCPU · torch {info.get('torch')}\n\n")
            fh.write(table + "\n\n")
            fh.write("### Findings\n\n")
            for line in findings:
                fh.write(f"- {line}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
