import logging
from unittest.mock import MagicMock, patch

from app.main import main


@patch("app.main.DatasetProcessor")
@patch("app.main.DatasetDownloader")
def test_main_calls_services(
    mock_downloader_cls: MagicMock, mock_processor_cls: MagicMock, caplog
):
    """
    Test that the main function correctly orchestrates the services.
    """
    # Create instances returned by the mocked classes
    mock_downloader = mock_downloader_cls.return_value
    mock_processor = mock_processor_cls.return_value

    with caplog.at_level(logging.INFO):
        main()

    # Verify that the services were instantiated and called
    mock_downloader_cls.assert_called_once()
    mock_downloader.download_all.assert_called_once()

    mock_processor_cls.assert_called_once()
    mock_processor.process.assert_called_once_with(total_samples=100000)

    # Verify that logs are generated correctly
    assert "Starting SLM Dataset Pipeline" in caplog.text
    assert "Pipeline execution finished." in caplog.text
