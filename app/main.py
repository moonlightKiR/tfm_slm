import logging

from app.dataset.downloader import DatasetDownloader
from app.dataset.processor import DatasetProcessor

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
    processor = DatasetProcessor()
    processor.process(total_samples=100000)

    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()
