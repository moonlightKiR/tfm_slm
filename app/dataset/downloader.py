import logging
from pathlib import Path

from app.config import settings
from datasets import load_dataset
from pydantic import BaseModel

from .constants import DATASETS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """
    Schema for individual dataset configuration.
    """

    path: str
    name: str | None = None
    data_dir: str | None = None


def download_datasets(output_dir: str = settings.datasets_dir) -> None:
    """
    Downloads predefined datasets from Hugging Face and saves them locally.
    """
    _ensure_directory_exists(Path(output_dir))

    for dataset_key, raw_config in DATASETS_CONFIG.items():
        # Validate raw constant with Pydantic model
        config = DatasetConfig(**raw_config)
        _download_single_dataset(dataset_key, config, output_dir)


def _ensure_directory_exists(directory_path: Path) -> None:
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")


def _download_single_dataset(key: str, config: DatasetConfig, output_dir: str) -> None:
    logger.info(f"Starting download for: {key} ({config.path})")

    try:
        load_dataset(
            config.path,
            name=config.name,
            cache_dir=output_dir,
            data_dir=config.data_dir,
            trust_remote_code=settings.trust_remote_code,
        )
        logger.info(f"Successfully downloaded {key}")
    except Exception as error:
        logger.error(f"Failed to download {key}: {error}")


if __name__ == "__main__":
    download_datasets()
