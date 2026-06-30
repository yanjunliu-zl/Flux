"""Configuration loading using OmegaConf."""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load and merge YAML configuration files.

    Args:
        config_path: Path to main config YAML file.
        overrides: List of CLI-style overrides (e.g., ["training.batch_size=8"]).

    Returns:
        OmegaConf DictConfig object.
    """
    config = OmegaConf.load(config_path)

    # Apply CLI overrides
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    # Resolve interpolations
    OmegaConf.resolve(config)

    return config


def save_config(config: DictConfig, output_path: str | Path):
    """Save configuration to YAML file."""
    OmegaConf.save(config, output_path)
