import logging
import os
from pathlib import Path

import boto3
import torch
import torch.nn as nn
from app.config import settings
from app.model.architecture import HybridConfig, HybridModel
from botocore.exceptions import ClientError
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Optimizations for NVIDIA L4 (Ada Lovelace) on g6.xlarge
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')  # Enables TF32 for faster matmuls

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service responsible for the training loop of the SLM,
    highly optimized for NVIDIA L4 (g6.xlarge).
    """

    def __init__(self, dataset_path: str = ".datasets/mixed_dataset"):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s3_client = boto3.client("s3")
        self.bucket_name = settings.checkpoint_bucket

        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
        self,
        epochs: int = 1,
        batch_size: int = 32,
        grad_accum_steps: int = 1,
        lr: float = 5e-5,
    ):
        """
        Trains the hybrid architecture from scratch.
        Leverages bfloat16 and TF32 for NVIDIA L4 GPUs.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. Run processing first."
            )

        # 1. Load and Tokenize
        logger.info(f"Loading mixed dataset from {self.dataset_path}...")
        dataset = load_from_disk(str(self.dataset_path))

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )

        tokenized_ds = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )
        tokenized_ds.set_format("torch")

        # 2. Optimized DataLoader for g6.xlarge (4 vCPUs)
        dataloader = DataLoader(
            tokenized_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Matches vCPUs of g6.xlarge
            prefetch_factor=2,
        )

        # 3. Model Initialization (Hybrid Transformer-GRU)
        config = HybridConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=1024,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
        )

        logger.info("Initializing Hybrid Transformer-GRU Model...")
        model = HybridModel(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # 4. Checkpoint Resuming
        checkpoint_dir = Path("output")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        start_epoch = 0

        if checkpoint_path.exists():
            logger.info(f"Checkpoint found at {checkpoint_path}. Resuming...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            # Try to download from S3 if not local
            try:
                logger.info(f"Checking S3 bucket {self.bucket_name} for checkpoints...")
                self.s3_client.download_file(
                    self.bucket_name, "checkpoint.pt", str(checkpoint_path)
                )
                logger.info("Checkpoint downloaded from S3. Resuming...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
            except ClientError:
                logger.info("No checkpoint found in S3. Starting from scratch.")

        # L4 supports bfloat16, which is more stable than float16
        precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using mixed precision: {precision}")

        # GradScaler is only needed for float16. bfloat16 doesn't need scaling.
        scaler = torch.amp.GradScaler("cuda", enabled=(precision == torch.float16))

        logger.info(
            f"Training started. Batch size: {batch_size}, Grad Accum: {grad_accum_steps}"
        )
        model.train()

        for epoch in range(start_epoch, epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for i, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)

                # Autocast to bfloat16/float16
                with torch.amp.autocast("cuda", dtype=precision):
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs["loss"] / grad_accum_steps

                if precision == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum_steps == 0:
                    if precision == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * grad_accum_steps
                progress_bar.set_postfix(
                    {"loss": f"{loss.item() * grad_accum_steps:.4f}"}
                )

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

            # Save Checkpoint at the end of each epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Sync Checkpoint to S3
            try:
                self.s3_client.upload_file(
                    str(checkpoint_path), self.bucket_name, "checkpoint.pt"
                )
                logger.info(f"Checkpoint synced to S3: s3://{self.bucket_name}/checkpoint.pt")
            except ClientError as e:
                logger.error(f"Failed to sync checkpoint to S3: {e}")

        # 5. Export Final Model
        output_dir = checkpoint_dir / "nexus_slm_v1"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Final model saved in: {output_dir}")
