"""Specialized Ternary Trainer with Real-Time Tracking"""

import logging
import os
from dataclasses import dataclass

import torch
from comet_ml import Experiment
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class NextTrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 100000
    total_steps: int = 100000
    max_grad_norm: float = 1.0
    output_dir: str = "./checkpoints_next"
    save_steps: int = 1000
    log_steps: int = 1


# Allow NextTrainingConfig to be loaded via torch.load (PyTorch 2.6+ security)
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([NextTrainingConfig])


class TernaryTrainer:
    def __init__(self, model, train_dataloader, config=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config or NextTrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.total_steps
        )

        # Real-time Tracking
        self.experiment = None
        if os.getenv("COMET_API_KEY"):
            try:
                self.experiment = Experiment(
                    api_key=os.getenv("COMET_API_KEY"),
                    project_name="finai-next",
                    workspace="meridianalgo",
                )
                self.experiment.log_parameters(self.config.__dict__)
                print("[INFO] Comet ML initialized for real-time tracking.")
            except Exception as e:
                print(f"[WARN] Failed to initialize Comet ML: {e}")

        self.global_step = 0

    def train(self):
        self.model.train()
        train_iter = iter(self.train_dataloader)
        print(f"Starting Ternary Training for {self.config.max_steps} steps...")
        print(f"Device: {self.device}")

        # Initialize GitHub Step Summary
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as f:
                f.write("### Training Progress\n")
                f.write("| Step | Loss | Learning Rate |\n")
                f.write("| --- | --- | --- |\n")

        import gc

        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        for step in range(
            self.config.max_steps * self.config.gradient_accumulation_steps
        ):
            try:
                batch = next(train_iter)
            except StopIteration:
                print(
                    f"[INFO] Dataset exhausted or slice limit reached at step {step}. Stopping training."
                )
                break

            # Extract metadata and move tensors to device
            batch.pop("processed_idx", None)  # Remove metadata
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps

            # Check for NaN loss
            if torch.isnan(loss):
                print(
                    f"\n[WARN] NaN loss detected at step {self.global_step + 1}, skipping batch..."
                )
                self.optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Show micro-progress
            accumulation_idx = (step % self.config.gradient_accumulation_steps) + 1
            progress_bar.set_description(
                f"Batch {accumulation_idx}/{self.config.gradient_accumulation_steps}"
            )
            if accumulation_idx == 1:
                print(f"\n[WORK] Starting optimization step {self.global_step + 1}...")
            print(
                f"  > Processing batch {accumulation_idx}/{self.config.gradient_accumulation_steps}...",
                end="\r",
            )

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Tracking
                actual_loss = loss.item() * self.config.gradient_accumulation_steps
                lr = self.scheduler.get_last_lr()[0]

                # GitHub Step Summary
                if os.getenv("GITHUB_STEP_SUMMARY"):
                    with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as f:
                        f.write(
                            f"| {self.global_step} | {actual_loss:.4f} | {lr:.2e} |\n"
                        )

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {"loss": f"{actual_loss:.4f}", "lr": f"{lr:.2e}"}
                )
                progress_bar.set_description("Training")

                if self.experiment:
                    self.experiment.log_metric(
                        "loss", actual_loss, step=self.global_step
                    )
                    self.experiment.log_metric("lr", lr, step=self.global_step)

                # Cleanup
                if self.global_step % 10 == 0:
                    gc.collect()

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

    def save_checkpoint(self, path=None):
        save_path = path or os.path.join(
            self.config.output_dir, f"step-{self.global_step}"
        )
        os.makedirs(save_path, exist_ok=True)

        print(f"\n[INFO] Saving trainer state to {save_path}...")

        # Move model to CPU to release handles and save memory
        self.model.cpu()
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save model
        self.model.save_pretrained(save_path, safe_serialization=True)

        # Save optimizer, scheduler and global_step
        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(save_path, "trainer_state.pt"))
        print("[INFO] Checkpoint saved successfully.")

    def load_checkpoint(self, load_path):
        if not os.path.exists(load_path):
            print(f"[WARN] Checkpoint path {load_path} does not exist.")
            return False

        print(f"[INFO] Loading trainer state from {load_path}...")

        # Load trainer state
        state_file = os.path.join(load_path, "trainer_state.pt")
        if os.path.exists(state_file):
            checkpoint = torch.load(state_file, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint["global_step"]
            print(
                f"[INFO] Loaded optimizer, scheduler, and global_step ({self.global_step})."
            )
        else:
            print(
                f"[WARN] No trainer_state.pt found in {load_path}. Only model weights will be used."
            )

        return True
