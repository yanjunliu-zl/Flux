"""Adaptive LayerNorm (AdaLN) for diffusion transformer blocks.

Conditions normalization on timestep embeddings.
"""

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """Adaptive LayerNorm: modulates feature normalization based on conditioning.

    Given conditioning c (e.g., timestep embedding), produces per-channel
    shift, scale, and gate parameters:
        output = gate * (LayerNorm(x) * (1 + scale) + shift)

    Args:
        dim: Feature dimension.
        cond_dim: Conditioning dimension (timestep embedding dim).
        eps: LayerNorm epsilon.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Project conditioning to (shift, scale, gate)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )

        # Zero-initialize for stable training
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply AdaLN.

        Args:
            x: Input tensor (B, N, D).
            cond: Conditioning tensor (B, cond_dim).

        Returns:
            Tuple of (normalized_x, shift, scale, gate).
        """
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x_mod, shift, scale, gate
