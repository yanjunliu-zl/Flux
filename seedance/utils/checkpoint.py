"""Checkpoint save/load with FSDP state dict handling."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

from seedance.training.trainer import TrainingState


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    path: str,
):
    """Save training checkpoint.

    Args:
        model: Model (may be FSDP-wrapped).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        state: TrainingState with step/epoch info.
        path: Output file path.
    """
    # Get raw state dict (handle FSDP wrapping)
    if hasattr(model, "_fsdp_wrapped_module"):
        model_state = model.state_dict()
    elif hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": {
            "step": state.step,
            "epoch": state.epoch,
            "best_loss": state.best_loss,
        },
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    path: str,
    device: torch.device,
) -> TrainingState:
    """Load training checkpoint.

    Args:
        model: Model to load weights into.
        optimizer: Optimizer to load state (optional).
        scheduler: LR scheduler to load state.
        path: Checkpoint file path.
        device: Target device.

    Returns:
        TrainingState with restored step/epoch.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Load model weights
    if hasattr(model, "_fsdp_wrapped_module"):
        model.load_state_dict(checkpoint["model"])
    elif hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    # Move model to target device
    model.to(device)

    # Load optimizer
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer state tensors to target device
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if p in optimizer.state:
                    for k, v in optimizer.state[p].items():
                        if isinstance(v, torch.Tensor):
                            optimizer.state[p][k] = v.to(device)

    # Load scheduler
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Restore training state
    state = TrainingState(
        step=checkpoint["state"]["step"],
        epoch=checkpoint["state"]["epoch"],
        best_loss=checkpoint["state"]["best_loss"],
    )

    print(f"[Checkpoint] Loaded from {path} (step={state.step})")
    return state
