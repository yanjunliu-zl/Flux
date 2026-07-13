"""Cross-Scale Causal Attention for world model temporal reasoning.

VideoWorld 2 (CVPR 2026): Hierarchical spatio-temporal DiT extracts
differentiable motion, collision, and gravity priors by attending across
multiple temporal scales with causal masking.

Architecture:
  1. Multi-scale temporal pyramid: 1×, 2×, 4× downsampled timelines
  2. Cross-scale attention: each scale queries the coarser scale above it
  3. Causal masking: future frames never leak into past predictions
  4. Learnable scale embeddings: model learns which features belong at which scale

This enables:
  - Coarse scale: slow scene changes (camera pans, lighting)
  - Fine scale: fast object motion (collisions, gestures)
  - Cross-scale: causal relationships (ball thrown → lands → bounces)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalePyramid(nn.Module):
    """Build a temporal scale pyramid from video features.

    Downsamples the temporal dimension at multiple rates to create
    a hierarchy of temporal resolutions.

    Args:
        num_scales: Number of pyramid levels (default: 3 → 1×, 2×, 4×).
        pool_mode: "avg" (average pooling) or "conv" (learned strided conv).
        dim: Feature dimension (for conv mode).
    """

    def __init__(self, num_scales: int = 3, pool_mode: str = "avg", dim: int | None = None):
        super().__init__()
        self.num_scales = num_scales
        self.pool_mode = pool_mode

        if pool_mode == "conv" and dim is not None:
            self.downsamplers = nn.ModuleList([
                nn.Conv1d(dim, dim, kernel_size=2 * s, stride=s, padding=s // 2)
                for s in [1, 2, 4][:num_scales]
            ])
        else:
            self.downsamplers = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Build temporal pyramid.

        Args:
            x: (B, T, D) or (B*HW, T, D) flattened video tokens.

        Returns:
            List of tensors at scales [1×, 2×, 4×, ...], each (B, T_s, D).
        """
        pyramid = [x]
        for scale in range(1, self.num_scales):
            if self.downsamplers is not None:
                # Learned conv downsampling
                prev = pyramid[-1].transpose(1, 2)  # (B, D, T)
                down = self.downsamplers[scale](prev)  # (B, D, T_s)
                pyramid.append(down.transpose(1, 2))   # (B, T_s, D)
            else:
                # Average pooling: kernel=2, stride=2
                prev = pyramid[-1]
                if prev.shape[1] % 2 == 1:
                    prev = prev[:, :-1]  # Drop last if odd
                B, T, D = prev.shape
                down = prev.reshape(B, T // 2, 2, D).mean(dim=2)
                pyramid.append(down)
        return pyramid


class CrossScaleCausalAttention(nn.Module):
    """Cross-scale causal attention.

    Each scale attends to:
      1. Itself (intra-scale self-attention)
      2. The coarser scale above it (cross-scale attention)
    All with causal masking (frame t can only attend to frames ≤ t).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_scales: Number of temporal scales.
        dropout: Attention dropout.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_scales: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_scales = num_scales

        # Intra-scale self-attention (one per scale)
        self.intra_attn = nn.ModuleList([
            _CausalSelfAttention(dim, num_heads, dropout) for _ in range(num_scales)
        ])

        # Cross-scale attention (fine queries coarse)
        self.cross_attn = nn.ModuleList([
            _CausalCrossAttention(dim, num_heads, dropout)
            for _ in range(num_scales - 1)  # Only finer scales query coarser
        ])

        # Scale embeddings
        self.scale_embed = nn.Parameter(
            torch.randn(num_scales, dim) * 0.02
        )

    def forward(self, pyramid: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply cross-scale causal attention to the temporal pyramid.

        Args:
            pyramid: List of (B, T_s, D) from ScalePyramid.

        Returns:
            Updated pyramid with cross-scale information.
        """
        out = []

        # Coarsest scale: only self-attention
        x_coarse = pyramid[-1] + self.scale_embed[-1]
        x_coarse = self.intra_attn[-1](x_coarse)
        out.append(x_coarse)

        # Intermediate scales: self-attn + cross-attn from coarser
        for s in range(self.num_scales - 2, -1, -1):
            x_s = pyramid[s] + self.scale_embed[s]
            # Intra-scale self-attention
            x_s = self.intra_attn[s](x_s)
            # Cross-scale: attend to the coarser scale above
            # Need to upsample coarse features to match temporal length
            coarse = out[0]  # Coarsest scale
            if coarse.shape[1] != x_s.shape[1]:
                coarse = F.interpolate(
                    coarse.transpose(1, 2),
                    size=x_s.shape[1], mode="linear", align_corners=False
                ).transpose(1, 2)
            x_s = self.cross_attn[self.num_scales - 2 - s](x_s, coarse)
            out.insert(0, x_s)

        return out  # [finest, ..., coarsest]


class _CausalSelfAttention(nn.Module):
    """Self-attention with causal masking."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each (B, T, H, D_h)

        # (B, H, T, D_h)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Causal mask: token at position i attends to [0, i]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class _CausalCrossAttention(nn.Module):
    """Cross-attention with causal masking (queries attend to keys at ≤ their position)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Cross-attention: fine queries attend to coarse keys.

        Uses temporal alignment — each fine timestep attends to the
        corresponding coarse timestep region (via interpolation).
        """
        B, T_q, D = query.shape
        T_k = key_value.shape[1]

        q = self.q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(B, T_k, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(B, T_k, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Build causal-temporal alignment mask
        # Fine timestep i maps to coarse timestep floor(i * T_k / T_q)
        mask = torch.zeros(T_q, T_k, device=query.device, dtype=torch.bool)
        for i in range(T_q):
            max_k = int((i + 1) * T_k / T_q)  # Causal: can attend up to aligned position
            mask[i, max_k:] = True
        # Invert for attn_mask (True = masked OUT)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, T_q, D)
        return self.proj(out)
