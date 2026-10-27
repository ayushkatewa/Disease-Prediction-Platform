"""
Production Logging Setup
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from src.config import LOGS_DIR, config


def get_logger(name: str = "DiabetesClusterML") -> logging.Logger:
    """
    Creates or retrieves a standardized logger with both console and rotating file output.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Console Handler with clear formatting
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating: 5MB per file, up to 3 backups)
    try:
        log_file = os.path.join(LOGS_DIR, "app.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logger: {e}")

    logger.propagate = False
    return logger

