"""Simplified trainer for Fin.AI"""

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
    output_dir: str = "./checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: bool = True
    log_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "fin-ai"
    fp16: bool = True
    plot_steps: int = 500

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

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = (
            GradScaler() if self.config.fp16 and self.device.type == "cuda" else None
        )

        self.global_step = 0
        self.epoch = 0

        # Wandb
        if self.config.use_wandb:
            try:
                import wandb

                # Enhanced Wandb config
                wandb_config = {
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

                run_name = f"train-{wandb_config['dataset']}-run{os.environ.get('GITHUB_RUN_NUMBER', 'local')}"
                wandb.init(
                    project=self.config.wandb_project,
                    config=wandb_config,
                    name=run_name,
                    tags=[
                        "continuous-training",
                        wandb_config["dataset"],
                        f"v{os.environ.get('GITHUB_RUN_NUMBER', '0')}",
                    ],
                )

                # Define custom charts
                wandb.define_metric("train/step")
                wandb.define_metric("train/*", step_metric="train/step")

                print("Wandb initialized with enhanced logging")
            except Exception as e:
                logger.warning(f"Wandb not available: {e}")

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
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, self.config.warmup_steps))
            # Cosine decay after warmup
            progress = float(step - self.config.warmup_steps) / float(
                max(1, self.config.max_steps - self.config.warmup_steps)
            )
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        print(f"\nStarting Fin.AI training on {self.device}")
        print(f"Model: {self.model.config.num_parameters:,} parameters")
        print(f"Target steps: {self.config.max_steps:,}")
        print(f"Checkpoints: {self.config.output_dir}\n")

        if self.config.resume_from_checkpoint:
            self._load_checkpoint()

        self.model.train()
        train_iter = iter(self.train_dataloader)
        accumulation_loss = 0.0
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
                loss = outputs["loss"] / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Free memory
            del outputs

            accumulation_loss += loss.item()

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
                            "loss": f"{accumulation_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                            "tok/s": f"{tokens_per_sec:.0f}",
                        }
                    )
                else:
                    print(
                        f"Items processed: {self.global_step * self.config.batch_size} | Step {self.global_step}/{self.config.max_steps} | Loss: {accumulation_loss:.4f} | LR: {current_lr:.2e} | Tokens/s: {tokens_per_sec:.0f}"
                    )

                try:
                    import wandb

                    # Calculate additional metrics
                    perplexity = math.exp(
                        min(accumulation_loss, 20)
                    )  # Cap to avoid overflow
                    progress = (self.global_step / self.config.max_steps) * 100

                    # Enhanced logging with descriptive names
                    wandb.log(
                        {
                            # Core metrics
                            "train/step": self.global_step,
                            "train/loss": accumulation_loss,
                            "train/perplexity": perplexity,
                            "train/learning_rate": current_lr,
                            # Performance metrics
                            "performance/tokens_per_second": tokens_per_sec,
                            "performance/steps_per_second": (
                                self.config.log_steps / elapsed if elapsed > 0 else 0
                            ),
                            "performance/time_per_step": (
                                elapsed / self.config.log_steps if elapsed > 0 else 0
                            ),
                            # Progress metrics
                            "progress/percent_complete": progress,
                            "progress/epoch": self.epoch,
                            "progress/steps_remaining": self.config.max_steps
                            - self.global_step,
                            # Gradient metrics
                            "gradients/global_norm": torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), float("inf")
                            ),
                            # Dataset info
                            "dataset/name": (
                                self.dataset_cycler.current_dataset_name
                                if self.dataset_cycler
                                else "unknown"
                            ),
                            "dataset/offset": (
                                self.dataset_cycler.get_current_offset()
                                if self.dataset_cycler
                                else 0
                            ),
                        },
                        step=self.global_step,
                    )
                except Exception:
                    pass

                accumulation_loss = 0.0
                start_time = time.time()

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()

        # Log final summary to Wandb
        try:
            import wandb

            wandb.log(
                {
                    "summary/final_step": self.global_step,
                    "summary/total_epochs": self.epoch,
                    "summary/final_loss": accumulation_loss,
                }
            )
            wandb.finish()
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
        self.model.save_pretrained(model_save_path)
        
        if (
            self.global_step % (self.config.save_steps * 5) == 0
        ):  # Only log every 5th save
            print(f"Checkpoint saved at step {self.global_step}")

    def _load_checkpoint(self):
        if not os.path.exists(self.config.output_dir):
            return

        # 1. Try to find the latest checkpoint-*.pt
        checkpoints = [
            f
            for f in os.listdir(self.config.output_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]

        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            latest = checkpoints[-1]
            checkpoint_path = os.path.join(self.config.output_dir, latest)
            print(f"📂 Resuming from checkpoint: {latest}")

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
                print(f"Resumed from step {self.global_step}")
                return
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

        # 2. Fallback: Check for safetensors or classic format
        model_paths = [
            os.path.join(self.config.output_dir, "model", "model.safetensors"),
            os.path.join(self.config.output_dir, "model", "model.pt"),
            os.path.join(self.config.output_dir, "model.pt"),
            os.path.join(self.config.output_dir, "model", "pytorch_model.bin"),
        ]

        for path in model_paths:
            if os.path.exists(path):
                print(
                    f"Found pretrained model weights at {path}, starting fine-tuning..."
                )
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
                    self.model.load_state_dict(sd)
                    print(f"Loaded weights from {path}")
                    # Reset optimizer/scheduler since we are starting fresh/fine-tuning
                    self.global_step = 0
                    self.epoch = 0
                    return
                except Exception as e:
                    print(f"Failed to load pretrained weights: {e} - Starting fresh")

        print("Starting fresh training (random initialization)...")
        self.global_step = 0
        self.epoch = 0
