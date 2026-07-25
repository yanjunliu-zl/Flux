"""Exponential Moving Average (EMA) for model weights."""

import copy
import torch
import torch.nn as nn


class EMA(nn.Module):
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters updated with:
        shadow = decay * shadow + (1 - decay) * param

    Args:
        model: The model to track.
        decay: EMA decay rate (default: 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.shadow = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA shadow parameters.

        Args:
            model: Current model (parameters have just been updated).
        """
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            if name in self.shadow:
                # Move shadow to correct device if needed
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to model (for inference/evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def store(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Store current model parameters (for restoring after copying EMA)."""
        stored = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                stored[name] = param.data.clone()
        return stored

    def restore(self, model: nn.Module, stored: dict[str, torch.Tensor]):
        """Restore model parameters from stored state."""
        for name, param in model.named_parameters():
            if name in stored:
                param.data.copy_(stored[name])
