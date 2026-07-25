"""Optimizer factory with sensible defaults for transformer training."""

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """Create optimizer with separate weight decay groups.

    Biases and LayerNorm parameters get no weight decay.
    All other parameters get the specified weight decay.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        betas: Adam betas.
        weight_decay: Weight decay coefficient.
        eps: Adam epsilon.
        optimizer_type: "adamw" or "lion".

    Returns:
        Configured optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln_" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type == "adamw":
        # Try to use fused AdamW if available (PyTorch 2.0+)
        try:
            return torch.optim.AdamW(
                param_groups, lr=lr, betas=betas, eps=eps, fused=True
            )
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(
                param_groups, lr=lr, betas=betas, eps=eps
            )
    elif optimizer_type == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            return torch.optim.AdamW(
                param_groups, lr=lr, betas=betas, eps=eps
            )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
