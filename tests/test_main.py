import logging
from unittest.mock import MagicMock, patch

from app.main import main


@patch("app.main.download_datasets")
def test_main_calls_download_datasets(mock_download_datasets: MagicMock, caplog):
    """
    Test that the main function correctly starts the pipeline
    and calls the dataset download phase.
    """
    with caplog.at_level(logging.INFO):
        main()

    # Verify that the download function was called
    mock_download_datasets.assert_called_once()

    # Verify that logs are generated correctly
    assert "Starting TFM Small Language Model Pipeline" in caplog.text
    assert "Phase 1: Downloading general-purpose datasets..." in caplog.text
    assert "Pipeline step completed successfully." in caplog.text
