
import sys
import os
import torch
import shutil

# Add current dir to path
sys.path.append(os.getcwd())

from fin_ai.model import FinAIConfig, FinAIForCausalLM
from fin_ai.training import FinAITrainer, TrainingConfig, DatasetCycler
from fin_ai.data import load_datasets_from_config, create_dataloader
from transformers import AutoTokenizer

def test_training():
    print("Testing training fix...")
    
    # 1. Config
    config_path = "config/model_config.yaml"
    dataset_config = "config/datasets.yaml"
    
    model_config = FinAIConfig.from_yaml(config_path)
    # Use micro model for speed
    model_config.size_preset = "micro"
    model_config.n_layers = 2
    model_config.n_heads = 4
    model_config.embed_dim = 128
    
    training_config = TrainingConfig(
        output_dir="./test_checkpoints",
        max_steps=20,
        batch_size=2,
        log_steps=1,
        use_comet=False,
        push_to_hub=False
    )
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_config.vocab_size = len(tokenizer)
    
    # 3. Model
    model = FinAIForCausalLM(model_config)
    print(f"Model params: {model.num_parameters()}")
    
    # 4. Data
    # Use dummy data to avoid downloading
    # Create dummy dataset bypassing loader if possible, or use loader with small subset
    # Let's just use wikitext fallback logic by pointing to non-existent config or similar, 
    # but easier to just mock the dataset
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, length=100):
            self.data = torch.randint(0, 50257, (length, 32))
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            input_ids = self.data[idx]
            # Add some padding to test mask
            input_ids[-5:] = tokenizer.pad_token_id
            
            attention_mask = torch.ones_like(input_ids)
            attention_mask[-5:] = 0
            
            labels = input_ids.clone()
            labels[-5:] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
    dataset = MockDataset(100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # 5. Train
    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=training_config
    )
    
    print("Starting training step...")
    trainer.train()
    
    # Clean up
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
        
if __name__ == "__main__":
    test_training()
