import logging
import os

from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASETS_CONFIG = {
    "open_assistant": {"path": "OpenAssistant/oasst1", "name": None},
    "sharegpt": {"path": "anon8231489123/ShareGPT_Vicuna_unfiltered", "name": None},
    "alpaca": {"path": "tatsu-lab/alpaca", "name": None},
    "ultrachat": {"path": "stingning/ultrachat", "name": None},
    "the_stack_yaml": {
        "path": "bigcode/the-stack",
        "name": None,
        "data_dir": "data/yaml",  # Example filter for YAML in The Stack
    },
}


def download_datasets(output_dir: str = ".datasets"):
    """
    Downloads the predefined datasets from Hugging Face and saves them locally.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    for key, config in DATASETS_CONFIG.items():
        logger.info(f"Starting download for: {key} ({config['path']})")
        try:
            # We use load_dataset which handles caching and local storage
            # By default it saves to ~/.cache/huggingface/datasets
            # But we can specify a cache_dir to move it to .datasets
            load_dataset(
                config["path"],
                name=config.get("name"),
                cache_dir=output_dir,
                data_dir=config.get("data_dir"),
                trust_remote_code=True,
            )
            logger.info(f"Successfully downloaded {key}")
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")


if __name__ == "__main__":
    download_datasets()
