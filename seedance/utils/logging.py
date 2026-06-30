"""Logging utilities for training and evaluation."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console logging.

    Args:
        level: Logging level.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("seedance")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def log_metrics(metrics: dict[str, float], step: int, logger: logging.Logger | None = None):
    """Log training metrics.

    Args:
        metrics: Dict of metric_name -> value.
        step: Current training step.
        logger: Logger instance (creates one if None).
    """
    if logger is None:
        logger = logging.getLogger("seedance")
    msg = f"[Step {step}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(msg)
