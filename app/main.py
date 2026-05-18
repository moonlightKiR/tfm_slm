import logging
import os

from app.chat import ChatService
from app.dataset.downloader import DatasetDownloader
from app.dataset.processor import DatasetProcessor
from app.training.trainer import TrainingService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the SLM pipeline.
    Orchestrates the different services following SOLID principles.
    """
    logger.info("Starting SLM Dataset Pipeline")

    # 1. Dataset Downloading Service
    downloader = DatasetDownloader()
    downloader.download_all()

    # 2. Dataset Processing/Mixing Service
    # Aumentado a 200k para un entrenamiento completo
    processor = DatasetProcessor()
    processor.process(total_samples=200000)

    # 3. Training Service (Hybrid Model)
    logger.info("Phase 3: Training Hybrid Transformer-GRU Model from scratch...")
    trainer = TrainingService()
    # Optimized for G7e (RTX PRO 6000 Blackwell - 96GB):
    # Physical batch size = 64, Accumulation = 4 => Effective batch size = 256
    trainer.train(epochs=5, batch_size=64, grad_accum_steps=4)

    logger.info("Pipeline execution finished successfully.")

    # 4. Interactive Chat Session
    logger.info("Starting interactive chat session...")
    chat_service = ChatService()
    chat_service.run()

    logger.info("Pipeline complete. Container ready for docker exec.")
    os.execv("/usr/bin/tail", ["tail", "-f", "/dev/null"])


if __name__ == "__main__":
    main()
