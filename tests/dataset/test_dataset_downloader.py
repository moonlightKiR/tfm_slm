import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.dataset.constants import DATASETS_CONFIG
from app.dataset.downloader import (
    DatasetConfig,
    _download_single_dataset,
    _ensure_directory_exists,
    download_datasets,
)


def test_dataset_config_model():
    """Test the Pydantic model for dataset configuration validation."""
    config = DatasetConfig(path="tatsu-lab/alpaca")
    assert config.path == "tatsu-lab/alpaca"
    assert config.name is None
    assert config.data_dir is None

    config_with_opts = DatasetConfig(
        path="bigcode/the-stack", name="yaml", data_dir="data/yaml"
    )
    assert config_with_opts.path == "bigcode/the-stack"
    assert config_with_opts.name == "yaml"
    assert config_with_opts.data_dir == "data/yaml"


def test_ensure_directory_exists(tmp_path: Path):
    """Test that the directory is created if it does not exist."""
    # tmp_path is a built-in pytest fixture providing a temporary directory
    target_dir = tmp_path / "new_datasets_dir"
    
    assert not target_dir.exists()
    
    _ensure_directory_exists(target_dir)
    
    assert target_dir.exists()
    assert target_dir.is_dir()


@patch("app.dataset.downloader.load_dataset")
def test_download_single_dataset_success(mock_load_dataset: MagicMock):
    """Test the successful downloading of a single dataset using mocks."""
    config = DatasetConfig(
        path="OpenAssistant/oasst1", name=None, data_dir=None
    )
    output_dir = ".test_datasets"
    
    _download_single_dataset("open_assistant", config, output_dir)
    
    # Assert that the Hugging Face load_dataset was called with correct parameters
    mock_load_dataset.assert_called_once_with(
        "OpenAssistant/oasst1",
        name=None,
        cache_dir=output_dir,
        data_dir=None,
        trust_remote_code=True,
    )


@patch("app.dataset.downloader.load_dataset")
def test_download_single_dataset_failure(mock_load_dataset: MagicMock, caplog):
    """Test that a failure in load_dataset is caught and logged."""
    # Simulate an error during the dataset download
    mock_load_dataset.side_effect = Exception("Hugging Face API Error")
    
    config = DatasetConfig(path="fake/dataset")
    
    with caplog.at_level(logging.ERROR):
        _download_single_dataset("fake_key", config, ".test_datasets")
    
    # Check if the error was properly caught and logged
    assert "Failed to download fake_key: Hugging Face API Error" in caplog.text


@patch("app.dataset.downloader._download_single_dataset")
@patch("app.dataset.downloader._ensure_directory_exists")
def test_download_datasets(
    mock_ensure_dir: MagicMock, mock_download_single: MagicMock
):
    """Test the main entry point to ensure it iterates over all configs."""
    download_datasets(output_dir=".custom_output")
    
    # Ensure directory creation was called
    mock_ensure_dir.assert_called_once()
    
    # Ensure individual download was called for each configuration in the constants
    expected_calls = len(DATASETS_CONFIG)
    assert mock_download_single.call_count == expected_calls
