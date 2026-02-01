"""
Centralized Logger Configuration
================================

Provides a unified logging configuration for all modules.
Import this module to get a configured logger.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message here")
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


# =============================================================================
# LOGGER FACTORY
# =============================================================================
def get_logger(
    name: str,
    level: int = DEFAULT_LEVEL,
    log_to_file: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to file
        log_file: Custom log file path (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            # Use module name as log file
            safe_name = name.replace(".", "_").replace("/", "_")
            log_file = LOG_DIR / f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            log_file = Path(log_file)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def setup_root_logger(level: int = DEFAULT_LEVEL) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
