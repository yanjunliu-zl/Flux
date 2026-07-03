"""Checkpoint save/load with FSDP/DDP state dict handling."""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seedance.training.trainer import TrainingState


def _is_fsdp(model: nn.Module) -> bool:
    """Check if model is wrapped with FSDP."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return isinstance(model, FSDP)


def _get_model_state(model: nn.Module) -> dict:
    """Get raw model state dict, handling FSDP/DDP wrappers.

    For FSDP: uses FSDP.state_dict() which consolidates across shards.
    For DDP: accesses model.module.
    For unwrapped: direct state_dict().
    """
    if _is_fsdp(model):
        # FSDP.state_dict() gathers full unsharded weights
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            return model.state_dict()
    elif hasattr(model, "module"):
        # DDP wrapper
        return model.module.state_dict()
    else:
        return model.state_dict()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: "TrainingState",
    path: str,
):
    """Save training checkpoint.

    Handles FSDP (full state dict consolidation), DDP, and unwrapped models.
    Only call from the main process.

    Args:
        model: Model (may be FSDP/DDP-wrapped).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        state: TrainingState with step/epoch info.
        path: Output file path.
    """
    model_state = _get_model_state(model)

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
) -> "TrainingState":
    """Load training checkpoint.

    Handles FSDP (full state dict distribution), DDP, and unwrapped models.

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
    model_state = checkpoint["model"]

    # Load model weights (FSDP-aware)
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.state_dict_type(model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            model.load_state_dict(model_state)
    elif hasattr(model, "module"):
        # DDP wrapper
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

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

    # Restore training state (lazy import to break circular dependency)
    from seedance.training.trainer import TrainingState
    state = TrainingState(
        step=checkpoint["state"]["step"],
        epoch=checkpoint["state"]["epoch"],
        best_loss=checkpoint["state"]["best_loss"],
    )

    print(f"[Checkpoint] Loaded from {path} (step={state.step})")
    return state
