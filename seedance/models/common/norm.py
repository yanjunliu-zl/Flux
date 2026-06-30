"""Normalization layers used across the model."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard Layer Normalization with optional elementwise affine."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, SD3, Flux)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * rms
        if self.weight is not None:
            x_normed = x_normed * self.weight
        return x_normed


class GroupNorm(nn.Module):
    """Group Normalization wrapper with configurable groups."""

    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, num_channels),
            num_channels=num_channels,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
