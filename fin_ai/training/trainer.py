"""FinAI-Core v2.2 Ultra-Lite Trainer"""

import json
import logging
import os
from dataclasses import dataclass

import torch
import yaml

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
    optimizer: str = "adamw_8bit"
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
    fp16: bool = False
    plot_steps: int = 500
    gradient_checkpointing: bool = False
    hf_repo_id: str = None
    push_to_hub: bool = True

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        training_config = config.get("training", {})
        checkpointing = config.get("checkpointing", {})
        logging_config = config.get("logging", {})
        merged_config = {**training_config, **checkpointing, **logging_config}
        return cls(**{k: v for k, v in merged_config.items() if hasattr(cls, k)})


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
        self.dataset_offsets = {}
        self._load_state()
        self._update_current_dataset()
        self._save_state()

    def _update_current_dataset(self):
        from datetime import datetime

        if self.datasets:
            current_hour = datetime.now().hour
            self.current_dataset_idx = current_hour % len(self.datasets)

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.current_dataset_idx = state.get("current_dataset_idx", 0)
                    self.dataset_offsets = state.get("dataset_offsets", {})
            except Exception as e:
                logger.warning(f"Failed to load dataset state: {e}")

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
        return name.split("/")[-1] if "/" in name else name

    def get_current_offset(self) -> int:
        name = self.get_current_dataset().get("name")
        if not name:
            return 0
        return self.dataset_offsets.get(name, 0)

    def increment_offset(self, amount: int):
        name = self.get_current_dataset().get("name")
        if name:
            current = self.dataset_offsets.get(name, 0)
            self.dataset_offsets[name] = current + amount
            self._save_state()


class FinAITrainer:
    def __init__(self, model, train_dataloader, config=None, dataset_cycler=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config or TrainingConfig()
        self.dataset_cycler = dataset_cycler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.global_step = 0

    def _create_optimizer(self):
        try:
            import bitsandbytes as bnb

            # bitsandbytes 8-bit optimizers require CUDA
            use_8bit = "8bit" in self.config.optimizer and torch.cuda.is_available()
            optim_cls = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW
            if "8bit" in self.config.optimizer and not torch.cuda.is_available():
                logger.warning(
                    "bitsandbytes 8-bit optimizer requested but CUDA not available. Falling back to torch.optim.AdamW."
                )
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to standard AdamW")
            optim_cls = torch.optim.AdamW

        return optim_cls(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
        )

    def _create_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

    def _get_hf_token(self):
        token = os.environ.get("HF_TOKEN")
        if not token and os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
        return token

    def train(self):
        self.model.train()
        train_iter = iter(self.train_dataloader)

        # max_steps refers to optimizer steps (after gradient accumulation)
        total_forward_steps = (
            self.config.max_steps * self.config.gradient_accumulation_steps
        )

        for step in range(total_forward_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.log_steps == 0:
                    print(
                        f"Step {self.global_step} | Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # Stop if we've reached max_steps optimizer updates
                if self.global_step >= self.config.max_steps:
                    break

    def save_checkpoint(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        dataset_name = (
            self.dataset_cycler.current_dataset_name
            if self.dataset_cycler
            else "unknown"
        )
        save_path = os.path.join(
            self.config.output_dir, f"checkpoint-{dataset_name}-step-{self.global_step}"
        )
        self.model.save_pretrained(
            os.path.join(save_path, "model"), safe_serialization=False
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            os.path.join(save_path, "training_state.pt"),
        )
        print(f"Saved checkpoint to {save_path}")
