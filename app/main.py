import logging

from app.chat import run_chat
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
    # Optimized for G6e (L40S - 48GB): Physical batch size = 32, Accumulation = 2 => Effective batch size = 64
    trainer.train(epochs=5, batch_size=32, grad_accum_steps=2)

    logger.info("Pipeline execution finished successfully.")

    # 4. Interactive Chat Session
    logger.info("Starting interactive chat session...")
    run_chat()


if __name__ == "__main__":
    main()
