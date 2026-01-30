"""Specialized Ternary Trainer with Real-Time Tracking"""

from comet_ml import Experiment
import os
import torch
import math
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

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
    max_grad_norm: float = 1.0
    output_dir: str = "./checkpoints_next"
    save_steps: int = 1000
    log_steps: int = 1

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
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.max_steps
        )
        
        # Real-time Tracking
        self.experiment = None
        if os.getenv("COMET_API_KEY"):
            try:
                self.experiment = Experiment(
                    api_key=os.getenv("COMET_API_KEY"),
                    project_name="finai-next",
                    workspace="meridianalgo"
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
        
        import gc
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        for step in range(self.config.max_steps * self.config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Show micro-progress
            accumulation_idx = (step % self.config.gradient_accumulation_steps) + 1
            progress_bar.set_description(f"Batch {accumulation_idx}/{self.config.gradient_accumulation_steps}")
            if accumulation_idx == 1:
                print(f"\n[WORK] Starting optimization step {self.global_step + 1}...")
            print(f"  > Processing batch {accumulation_idx}/{self.config.gradient_accumulation_steps}...", end="\r")

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Tracking
                actual_loss = loss.item() * self.config.gradient_accumulation_steps
                lr = self.scheduler.get_last_lr()[0]
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{actual_loss:.4f}", "lr": f"{lr:.2e}"})
                progress_bar.set_description("Training")
                
                if self.experiment:
                    self.experiment.log_metric("loss", actual_loss, step=self.global_step)
                    self.experiment.log_metric("lr", lr, step=self.global_step)

                # Cleanup
                if self.global_step % 10 == 0:
                    gc.collect()

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, f"step-{self.global_step}")
        self.model.save_pretrained(save_path)
        print(f"\n[INFO] Saved ternary checkpoint to {save_path}")
