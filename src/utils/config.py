"""
Configuration loader.
Loads general settings from config/settings.yaml and secrets from .env.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Try to load python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_settings() -> Dict[str, Any]:
    """Load settings from YAML file."""
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    try:
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load settings.yaml: {e}")
    return {}

settings = load_settings()

class Config:
    """Application configuration."""
    
    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    CONFIG_DIR: Path = PROJECT_ROOT / "config"

    # API Keys (From ENV - Secrets only)
    USDA_API_KEY: str = os.environ.get("USDA_API_KEY", "")
    
    # Database (From YAML)
    DATABASE_PATH: str = settings.get("database", {}).get("path", "database/market_data.db")
    
    # Logging (From YAML)
    LOG_LEVEL: str = settings.get("logging", {}).get("level", "INFO")
    
    # Scraper settings (From YAML)
    _scraper = settings.get("scraper", {})
    RATE_LIMIT_MIN: int = _scraper.get("rate_limit_min", 2)
    RATE_LIMIT_MAX: int = _scraper.get("rate_limit_max", 5)
    MAX_RETRIES: int = _scraper.get("max_retries", 3)

config = Config()
