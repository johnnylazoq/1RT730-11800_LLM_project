"""Load configuration from configs/default.yaml and environment variables."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML file.

    Args:
        config_path: Path to config file. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config


def get_api_key(provider: str, env_var: str | None = None) -> str:
    """Get API key from environment.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic').
        env_var: Environment variable name. Defaults to {PROVIDER}_API_KEY.

    Returns:
        API key string.

    Raises:
        ValueError: If API key not found in environment.
    """
    if env_var is None:
        env_var = f"{provider.upper()}_API_KEY"

    key = os.getenv(env_var)
    if not key:
        raise ValueError(f"API key '{env_var}' not found in environment")

    return key
