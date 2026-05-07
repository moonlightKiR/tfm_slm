import logging
from pathlib import Path

import torch
import torch.nn as nn
from app.config import settings
from app.model.architecture import HybridConfig, HybridModel
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service responsible for the training loop of the SLM, optimized for CUDA.
    """

    def __init__(self, dataset_path: str = ".datasets/mixed_dataset"):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
        """
        Trains the hybrid architecture from scratch.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Run processing first.")

        # 1. Load and Tokenize
        logger.info(f"Loading mixed dataset from {self.dataset_path}...")
        dataset = load_from_disk(str(self.dataset_path))

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        tokenized_ds = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        tokenized_ds.set_format("torch")

        # 2. Optimized DataLoader for CUDA
        dataloader = DataLoader(
            tokenized_ds, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=2 if torch.cuda.is_available() else 0
        )

        # 3. Model Initialization (From Scratch)
        config = HybridConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=512,
            hidden_size=512,  
            num_layers=6,
            num_heads=8,
            intermediate_size=2048
        )
        
        logger.info("Initializing Hybrid Transformer-GRU Model from scratch...")
        model = HybridModel(config).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 4. Mixed Precision Scaler for CUDA
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        logger.info(f"Training started: {len(tokenized_ds)} samples.")
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                
                # Causal language modeling: input_ids as labels
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # 5. Export weights and config
        output_dir = Path("output/nexus_slm_v1")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Final model saved in: {output_dir}")
