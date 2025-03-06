import os
import logging
from .file_utils import ensure_directory_exists

def setup_logger(name: str, log_file: str, level = logging.INFO):
    """Sets up a logger with a specific name and log file."""
    ensure_directory_exists(os.path.dirname(log_file))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

app_logger = setup_logger("AppLogger", "logs/application.log")

def log_info(message: str):
    """Logs an info-level message."""
    app_logger.info(message)

def log_warning(message: str):
    """Logs a warning-level message."""
    app_logger.warning(message)

def log_error(message: str):
    """Logs an error-level message."""
    app_logger.error(message)

def log_debug(message: str):
    """Logs an debug-level message."""
    app_logger.debug(message)

