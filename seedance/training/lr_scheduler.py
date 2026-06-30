"""Learning rate schedulers."""

import math
import torch


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 5000,
    max_steps: int = 500000,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with linear warmup.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr: Minimum learning rate (relative to base lr, reached at max_steps).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup from 0 to 1
            return step / max(1, warmup_steps)
        elif step >= max_steps:
            return min_lr
        else:
            # Cosine decay from 1 to min_lr
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
