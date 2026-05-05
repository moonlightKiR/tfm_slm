import logging

from app.dataset import download_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting TFM Small Language Model Pipeline")

    # Phase 1: Download Datasets
    logger.info("Phase 1: Downloading general-purpose datasets...")
    download_datasets()

    logger.info("Pipeline step completed successfully.")


if __name__ == "__main__":
    main()
