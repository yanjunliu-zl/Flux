"""Adaptive modulation helpers: FiLM, AdaLN modulation.

These are shared between VAE and DB-DiT modules.
"""

import torch
import torch.nn as nn


class AdaLNModulation(nn.Module):
    """Adaptive LayerNorm modulation: produces shift, scale, gate from conditioning.

    Used in DB-DiT blocks to modulate normalization based on timestep.

    Args:
        dim: Feature dimension.
        cond_dim: Conditioning dimension (timestep embedding dim).
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )
        # Zero-initialize the last layer for stable training start
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (shift, scale, gate) triple, each of shape (..., dim)."""
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        return shift, scale, gate


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift.

    Args:
        x: Input tensor.
        shift: Shift tensor (same shape as x or broadcastable).
        scale: Scale tensor (same shape as x or broadcastable).

    Returns:
        Modulated tensor.
    """
    return x * (1 + scale) + shift
