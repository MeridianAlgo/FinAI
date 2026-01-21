"""Simplified trainer for Fin.AI"""

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass

import torch
import yaml
from torch.cuda.amp import GradScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    eval_steps: int = 500
    eval_batches: int = 25
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: bool = True
    log_steps: int = 10
    use_comet: bool = True
    comet_project: str = "fin-ai"
    comet_workspace: str = "meridianalgo"
    fp16: bool = True
    plot_steps: int = 500
    gradient_checkpointing: bool = False

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        training_config = config.get("training", {})
        checkpointing = config.get("checkpointing", {})
        logging_config = config.get("logging", {})
        return cls(
            **{k: v for k, v in training_config.items() if hasattr(cls, k)},
            **{k: v for k, v in checkpointing.items() if hasattr(cls, k)},
            **{k: v for k, v in logging_config.items() if hasattr(cls, k)},
        )


class DatasetCycler:
    def __init__(
        self,
        datasets_config_path: str,
        state_file: str = "checkpoints/dataset_state.json",
    ):
        self.config_path = datasets_config_path
        self.state_file = state_file
        with open(datasets_config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.datasets = self.config.get("datasets", [])
        self.current_dataset_idx = 0
        self.dataset_offsets = {}  # Track offsets for each dataset
        self._load_state()  # Load existing state
        self._update_current_dataset()
        self._save_state()

    def _update_current_dataset(self):
        """Update dataset index based on current hour."""
        from datetime import datetime

        if self.datasets:
            current_hour = datetime.now().hour
            self.current_dataset_idx = current_hour % len(self.datasets)

    def _load_state(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.current_dataset_idx = state.get("current_dataset_idx", 0)
                    self.dataset_offsets = state.get("dataset_offsets", {})
                    # Ensure offsets are integers/valid
                    for k, v in self.dataset_offsets.items():
                        try:
                            self.dataset_offsets[k] = int(v)
                        except:
                            self.dataset_offsets[k] = 0
            except Exception as e:
                logger.warning(f"Failed to load dataset state: {e}")
                self.dataset_offsets = {}

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "current_dataset_idx": self.current_dataset_idx,
                    "dataset_offsets": self.dataset_offsets,
                },
                f,
                indent=2,
            )

    def get_current_dataset(self):
        return self.datasets[self.current_dataset_idx] if self.datasets else {}

    @property
    def current_dataset_name(self):
        ds = self.get_current_dataset()
        name = ds.get("name", "unknown")
        # Extract short name (e.g., "roneneldan/TinyStories" -> "TinyStories")
        if "/" in name:
            name = name.split("/")[-1]
        return name

    def get_current_offset(self) -> int:
        """Get the offset for the current dataset."""
        name = self.get_current_dataset().get("name")
        if not name:
            return 0
        return self.dataset_offsets.get(name, 0)

    def increment_offset(self, amount: int):
        """Increment the offset for the current dataset."""
        name = self.get_current_dataset().get("name")
        if name:
            current = self.dataset_offsets.get(name, 0)
            self.dataset_offsets[name] = current + amount
            self._save_state()
            print(
                f"Updated offset for {name}: {current} -> {self.dataset_offsets[name]}"
            )


class FinAITrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader=None,
        config=None,
        dataset_cycler=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.dataset_cycler = dataset_cycler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable gradient checkpointing on CPU by default for better memory efficiency
        if self.device.type != "cuda":
            self.config.gradient_checkpointing = True
        if getattr(self.config, "gradient_checkpointing", False) and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = (
            GradScaler() if self.config.fp16 and self.device.type == "cuda" else None
        )

        self.global_step = 0
        self.epoch = 0

        # Comet ML
        comet_disabled = os.environ.get("COMET_DISABLE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        if self.config.use_comet and not comet_disabled:
            try:
                from comet_ml import Experiment

                # Enhanced Comet ML config
                comet_config = {
                    **self.config.__dict__,
                    "model_parameters": self.model.config.num_parameters,
                    "model_layers": self.model.config.n_layers,
                    "model_heads": self.model.config.n_heads,
                    "model_embed_dim": self.model.config.embed_dim,
                    "device": str(self.device),
                    "dataset": (
                        dataset_cycler.current_dataset_name
                        if dataset_cycler
                        else "unknown"
                    ),
                    "dataset_offset": (
                        dataset_cycler.get_current_offset() if dataset_cycler else 0
                    ),
                }

                run_name = f"train-{comet_config['dataset']}-run{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"

                self.experiment = Experiment(
                    api_key=os.environ.get("COMET_API_KEY"),
                    project_name=self.config.comet_project,
                    workspace=self.config.comet_workspace,
                )

                self.experiment.set_name(run_name)
                self.experiment.log_parameters(comet_config)
                self.experiment.add_tags(
                    [
                        "continuous-training",
                        comet_config["dataset"],
                        f"v{os.environ.get('GITHUB_RUN_NUMBER', '0')}",
                    ]
                )

                print("Comet ML initialized with enhanced logging")
            except Exception as e:
                logger.warning(f"Comet ML not available: {e}")
                self.experiment = None
        else:
            self.experiment = None

    def _create_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "ln" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )

    def _create_scheduler(self):
        total_optimizer_steps = max(
            1,
            math.ceil(
                self.config.max_steps / max(1, self.config.gradient_accumulation_steps)
            ),
        )
        warmup_optimizer_steps = max(
            0,
            math.ceil(
                self.config.warmup_steps
                / max(1, self.config.gradient_accumulation_steps)
            ),
        )

        def lr_lambda(step):
            if step < warmup_optimizer_steps:
                return float(step) / float(max(1, warmup_optimizer_steps))
            progress = float(step - warmup_optimizer_steps) / float(
                max(1, total_optimizer_steps - warmup_optimizer_steps)
            )
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @torch.no_grad()
    def _evaluate(self):
        dataloader = self.eval_dataloader or self.train_dataloader
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += float(outputs["loss"].detach().cpu().item())
            total_batches += 1
            if total_batches >= self.config.eval_batches:
                break

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))

        if self.experiment:
            try:
                self.experiment.log_metrics(
                    {
                        "eval_loss": avg_loss,
                        "eval_perplexity": perplexity,
                    },
                    step=self.global_step,
                )
            except Exception:
                pass

        self.model.train()

    def train(self):
        # Clean memory before training
        print("🧹 Pre-training memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nStarting Fin.AI training on {self.device}")
        print(f"Model: {self.model.config.num_parameters:,} parameters")
        print(f"Target steps: {self.config.max_steps:,}")
        print(f"Checkpoints: {self.config.output_dir}\n")

        if self.config.resume_from_checkpoint:
            self._load_checkpoint()

        self.model.train()
        train_iter = iter(self.train_dataloader)
        log_loss_sum = 0.0
        log_loss_count = 0
        last_logged_loss = None
        start_time = time.time()

        # Detect CI environment
        is_ci = (
            os.environ.get("CI", "false").lower() == "true"
            or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
        )

        pbar = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
            disable=is_ci,
        )

        # Keep track of local progress to update dataset offset later
        steps_processed = 0

        while self.global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                "cuda", enabled=self.config.fp16 and self.device.type == "cuda"
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                raw_loss = outputs["loss"]
                loss = raw_loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Free memory
            del outputs

            # Periodic garbage collection to prevent memory buildup
            if self.global_step % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            log_loss_sum += float(raw_loss.detach().cpu().item())
            log_loss_count += 1

            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            steps_processed += 1
            if not is_ci:
                pbar.update(1)

            if self.global_step % self.config.log_steps == 0:
                elapsed = time.time() - start_time
                current_lr = self.scheduler.get_last_lr()[0]
                tokens_per_sec = (
                    (
                        self.config.batch_size
                        * self.config.log_steps
                        * self.model.config.max_seq_len
                    )
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                if not is_ci:
                    pbar.set_postfix(
                        {
                            "loss": f"{(log_loss_sum / max(1, log_loss_count)):.4f}",
                            "lr": f"{current_lr:.2e}",
                            "tok/s": f"{tokens_per_sec:.0f}",
                        }
                    )
                else:
                    print(
                        f"Items processed: {self.global_step * self.config.batch_size} | Step {self.global_step}/{self.config.max_steps} | Loss: {(log_loss_sum / max(1, log_loss_count)):.4f} | LR: {current_lr:.2e} | Tokens/s: {tokens_per_sec:.0f}"
                    )

                if self.experiment:
                    try:
                        # Calculate additional metrics
                        mean_loss = log_loss_sum / max(1, log_loss_count)
                        last_logged_loss = mean_loss
                        perplexity = math.exp(min(mean_loss, 20))
                        progress = (self.global_step / self.config.max_steps) * 100

                        # Enhanced logging with descriptive names
                        self.experiment.log_metrics(
                            {
                                # Core metrics
                                "train_loss": mean_loss,
                                "train_perplexity": perplexity,
                                "learning_rate": current_lr,
                                # Performance metrics
                                "tokens_per_second": tokens_per_sec,
                                "steps_per_second": (
                                    self.config.log_steps / elapsed
                                    if elapsed > 0
                                    else 0
                                ),
                                "time_per_step": (
                                    elapsed / self.config.log_steps
                                    if elapsed > 0
                                    else 0
                                ),
                                # Progress metrics
                                "percent_complete": progress,
                                "epoch": self.epoch,
                                "steps_remaining": self.config.max_steps
                                - self.global_step,
                                # Gradient metrics
                                "gradient_global_norm": torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), float("inf")
                                ),
                                # Dataset info
                                "dataset_offset": (
                                    self.dataset_cycler.get_current_offset()
                                    if self.dataset_cycler
                                    else 0
                                ),
                            },
                            step=self.global_step,
                        )
                    except Exception:
                        pass

                log_loss_sum = 0.0
                log_loss_count = 0
                start_time = time.time()

            if self.global_step % self.config.eval_steps == 0:
                self._evaluate()

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()

        # Log final summary to Comet ML
        if self.experiment:
            try:
                self.experiment.log_metrics(
                    {
                        "final_step": self.global_step,
                        "total_epochs": self.epoch,
                        "final_loss": last_logged_loss,
                    }
                )
                self.experiment.end()
            except:
                pass

        print(f"\nTraining complete! Final step: {self.global_step}")

    def _save_checkpoint(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = os.path.join(
            self.config.output_dir, f"checkpoint-{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save model in Hugging Face format to 'model' subdir
        model_save_path = os.path.join(self.config.output_dir, "model")
        self.model.save_pretrained(model_save_path, safe_serialization=True)

        # Clean up old checkpoints based on save_total_limit
        if self.config.save_total_limit > 0:
            checkpoints = [
                f
                for f in os.listdir(self.config.output_dir)
                if f.startswith("checkpoint-") and f.endswith(".pt")
            ]
            if len(checkpoints) > self.config.save_total_limit:
                checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
                for old_checkpoint in checkpoints[: -self.config.save_total_limit]:
                    old_path = os.path.join(self.config.output_dir, old_checkpoint)
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass

        if (
            self.global_step % (self.config.save_steps * 5) == 0
        ):  # Only log every 5th save
            print(f"Checkpoint saved at step {self.global_step}")

    def _load_checkpoint(self):
        if not os.path.exists(self.config.output_dir):
            return

        # 1. Try to find the latest checkpoint-*.pt (FULL TRAINING STATE)
        checkpoints = [
            f
            for f in os.listdir(self.config.output_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]

        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            latest = checkpoints[-1]
            checkpoint_path = os.path.join(self.config.output_dir, latest)
            print(f"📂 Resuming from full checkpoint: {latest}")

            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )
                state_dict = checkpoint["model_state_dict"]
                # Filter out causal_mask from state_dict if present (legacy checkpoints)
                state_dict = {
                    k: v for k, v in state_dict.items() if "causal_mask" not in k
                }
                self.model.load_state_dict(state_dict)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.global_step = checkpoint["global_step"]
                self.epoch = checkpoint["epoch"]
                if self.scaler and "scaler_state_dict" in checkpoint:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print(f"✅ Resumed from step {self.global_step} (continuous training)")
                return
            except Exception as e:
                print(f"Failed to load full checkpoint: {e}")

        # 2. Fallback: Check for safetensors or classic format (WEIGHTS ONLY)
        # This is used when full checkpoint is not available (e.g., first run after HF download)
        model_paths = [
            os.path.join(self.config.output_dir, "model", "model.safetensors"),
            os.path.join(self.config.output_dir, "model", "model.pt"),
            os.path.join(self.config.output_dir, "model.pt"),
            os.path.join(self.config.output_dir, "model", "pytorch_model.bin"),
        ]

        for path in model_paths:
            if os.path.exists(path):
                try:
                    if path.endswith(".safetensors"):
                        from safetensors.torch import load_file

                        state_dict = load_file(path, device=str(self.device))
                    else:
                        # Load state dict
                        state_dict = torch.load(
                            path, map_location=self.device, weights_only=False
                        )

                    # Handle if it's a full checkpoint or just state dict
                    if "model_state_dict" in state_dict:
                        sd = state_dict["model_state_dict"]
                    else:
                        sd = state_dict

                    # Filter out causal_mask
                    sd = {k: v for k, v in sd.items() if "causal_mask" not in k}

                    # Check if architecture matches by looking at key patterns
                    model_keys = set(self.model.state_dict().keys())
                    checkpoint_keys = set(sd.keys())

                    # Check for architecture mismatch (e.g., GPT-2 style vs FinAI style)
                    if any(k.startswith("transformer.") for k in checkpoint_keys):
                        print(
                            f"WARNING: Skipping incompatible checkpoint at {path} (GPT-2 architecture)"
                        )
                        print(
                            "   Current model uses FinAI architecture. Starting fresh..."
                        )
                        continue

                    # Try to load the state dict
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        sd, strict=False
                    )

                    if len(missing_keys) > 10 or len(unexpected_keys) > 10:
                        print(
                            f"WARNING: Too many mismatched keys in {path}, skipping..."
                        )
                        print(
                            f"   Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
                        )
                        continue

                    print(f"Loaded pretrained weights from {path}")
                    if missing_keys:
                        print(
                            f"   Note: {len(missing_keys)} keys not found in checkpoint (will be randomly initialized)"
                        )

                    # IMPORTANT: When loading weights-only (no full checkpoint), we start fresh training
                    # This happens on first run after downloading from HF
                    print("   ⚠️  Weights-only load detected (no full checkpoint)")
                    print("   Starting fresh training with these weights (step 0)")
                    self.global_step = 0
                    self.epoch = 0
                    return
                except Exception as e:
                    print(f"Failed to load from {path}: {e}")

        print("Starting fresh training (random initialization)...")
        self.global_step = 0
        self.epoch = 0
