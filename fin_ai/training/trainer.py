"""Simplified trainer for Fin.AI"""

import os
import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
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
    def __init__(self, datasets_config_path: str, state_file: str = "checkpoints/dataset_state.json"):
        self.config_path = datasets_config_path
        self.state_file = state_file
        with open(datasets_config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.datasets = self.config.get("datasets", [])
        self.current_dataset_idx = 0
        self._load_state()
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self.current_dataset_idx = state.get("current_dataset_idx", 0)
            except:
                pass
    
    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({"current_dataset_idx": self.current_dataset_idx}, f)
    
    def get_current_dataset(self):
        return self.datasets[self.current_dataset_idx] if self.datasets else {}
    
    @property
    def current_dataset_name(self):
        ds = self.get_current_dataset()
        return ds.get("name", "unknown")


class FinAITrainer:
    def __init__(self, model, train_dataloader, eval_dataloader=None, config=None, dataset_cycler=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.dataset_cycler = dataset_cycler
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.config.fp16 and self.device.type == "cuda" else None
        
        self.global_step = 0
        self.epoch = 0
        
        # Wandb
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(project=self.config.wandb_project, config=self.config.__dict__)
            except:
                logger.warning("Wandb not available")
    
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
        
        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.learning_rate, betas=(self.config.adam_beta1, self.config.adam_beta2))
    
    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        print(f"\n🚀 Starting Fin.AI training on {self.device}")
        print(f"📊 Model: {self.model.config.num_parameters:,} parameters")
        print(f"🎯 Target steps: {self.config.max_steps:,}")
        print(f"💾 Checkpoints: {self.config.output_dir}\n")
        
        if self.config.resume_from_checkpoint:
            self._load_checkpoint()
        
        self.model.train()
        train_iter = iter(self.train_dataloader)
        accumulation_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(total=self.config.max_steps, initial=self.global_step, desc="Training")
        
        while self.global_step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', enabled=self.config.fp16 and self.device.type == "cuda"):
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
            
            accumulation_loss += loss.item()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            self.global_step += 1
            pbar.update(1)
            
            if self.global_step % self.config.log_steps == 0:
                elapsed = time.time() - start_time
                current_lr = self.scheduler.get_last_lr()[0]
                
                pbar.set_postfix({
                    "loss": f"{accumulation_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })
                
                try:
                    import wandb
                    wandb.log({
                        "train/loss": accumulation_loss,
                        "train/learning_rate": current_lr,
                    }, step=self.global_step)
                except:
                    pass
                
                accumulation_loss = 0.0
                start_time = time.time()
            
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()
        
        pbar.close()
        self._save_checkpoint()
        print(f"\n✅ Training complete! Final step: {self.global_step}")
    
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
        
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.model.save_pretrained(os.path.join(self.config.output_dir, "model"))
        if self.global_step % (self.config.save_steps * 5) == 0:  # Only log every 5th save
            print(f"💾 Checkpoint saved at step {self.global_step}")
    
    def _load_checkpoint(self):
        if not os.path.exists(self.config.output_dir):
            return
        checkpoints = [f for f in os.listdir(self.config.output_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
        if not checkpoints:
            return
        checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        latest = checkpoints[-1]
        checkpoint_path = os.path.join(self.config.output_dir, latest)
        print(f"📂 Resuming from checkpoint: {latest}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint["global_step"]
            self.epoch = checkpoint["epoch"]
            if self.scaler and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"✅ Resumed from step {self.global_step}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            self.global_step = 0
            self.epoch = 0
