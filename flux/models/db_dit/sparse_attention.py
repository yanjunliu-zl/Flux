"""Block Sparse Attention for long video sequences (120+ frames).

Enables 4K 30s generation by reducing temporal attention from O(T²) to
O(T × window_size), where T is the temporal dimension of the latent.

Three attention patterns (configurable per layer):
1. LOCAL_WINDOW   — each frame attends to ±W neighbor frames
2. SHIFTED_WINDOW — offset the window partition to cross boundaries
3. GLOBAL_TOKENS  — sparse global tokens for long-range dependencies

Reference:
    LongCat-Video (Oct 2025): Block Sparse Attention enables minute-long generation
    Open-Sora v1.3 (2025): Shift-Window Attention with 3D relative position bias
    Seedance 2.5 (Jun 2026): Native 30s 4K output
"""

import math
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPattern(Enum):
    LOCAL_WINDOW = "local_window"
    SHIFTED_WINDOW = "shifted_window"
    FULL_ATTENTION = "full_attention"  # fallback for short sequences


class BlockSparseTemporalAttention(nn.Module):
    """Block-sparse temporal attention for long video sequences.

    Replaces the standard temporal attention in VisionBranchBlock when
    num_frames exceeds a threshold (default: 32 frames).

    Each frame attends only to its local temporal window + sparse global
    tokens, reducing FLOPs from O(T²×D) to O(T×W×D).

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        window_size: Temporal window radius (total window = 2*W + 1).
            Default: 8 (each frame attends to ±8 neighbors = 17 frames).
        num_global_tokens: Number of global tokens for long-range connectivity.
            Default: 4 (first 2 and last 2 frames serve as global tokens).
        shift_window: If True, alternate shifted windows across layers.
        layer_idx: Current layer index (0-based) for shift scheduling.
        qk_norm: Apply QK normalization.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        num_global_tokens: int = 4,
        shift_window: bool = True,
        layer_idx: int = 0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.layer_idx = layer_idx

        # Shift windows on even layers (Swin Transformer pattern)
        self.do_shift = shift_window and (layer_idx % 2 == 1)
        self.shift_amount = window_size // 2 if self.do_shift else 0

        # QKV projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # QK normalization
        if qk_norm:
            from flux.models.common.norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Relative position bias for local window (learnable)
        window_len = 2 * window_size + 1
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, window_len))

        self.attn_dropout = dropout

    def _create_window_mask(
        self, T: int, device: torch.device
    ) -> torch.Tensor:
        """Create block-sparse attention mask with local windows + global tokens.

        Returns:
            (T, T) mask where 1 = attend, 0 = masked.
        """
        # Local window: each frame attends to its neighbors
        mask = torch.zeros(T, T, device=device)
        for i in range(T):
            left = max(0, i - self.window_size + self.shift_amount)
            right = min(T, i + self.window_size + 1 + self.shift_amount)
            mask[i, left:right] = 1.0

        # Global tokens: first N and last N frames attend to everything
        if self.num_global_tokens > 0:
            n = self.num_global_tokens // 2
            # First n frames are global (attend to all, all attend to them)
            mask[:n, :] = 1.0
            mask[:, :n] = 1.0
            # Last n frames are global
            mask[-n:, :] = 1.0
            mask[:, -n:] = 1.0

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Block-sparse temporal attention.

        Args:
            x: Video tokens reshaped for temporal attention (B*H*W, T, D).

        Returns:
            Output tensor of same shape.
        """
        B_HW, T, D = x.shape
        H = self.num_heads

        # Project Q, K, V
        q = self.q_proj(x).reshape(B_HW, T, H, self.head_dim)  # (B_HW, T, H, D_h)
        k = self.k_proj(x).reshape(B_HW, T, H, self.head_dim)
        v = self.v_proj(x).reshape(B_HW, T, H, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Use block-sparse if T > 2*window_size, otherwise full attention
        if T > 2 * self.window_size:
            out = self._sparse_attention(q, k, v, T)
        else:
            # Full attention for short sequences
            q = q.transpose(1, 2)  # (B_HW, H, T, D_h)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(B_HW, T, D)

        return self.out_proj(out)

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Compute block-sparse attention with local windows + global tokens.

        For each query frame i, attends to:
        - Local window: [i-W, i+W]
        - Global tokens: first N and last N frames

        Args:
            q, k, v: (B_HW, T, H, D_h)
            T: Temporal length.

        Returns:
            (B_HW, T, D) output.
        """
        B_HW, _, H, D_h = q.shape
        W = self.window_size
        N_g = self.num_global_tokens // 2  # global tokens on each end

        out = torch.zeros(B_HW, T, H, D_h, device=q.device, dtype=q.dtype)

        # Process each frame's local window
        for i in range(T):
            left = max(0, i - W + self.shift_amount)
            right = min(T, i + W + 1 + self.shift_amount)

            # Collect keys/values for local window + global tokens
            k_indices = list(range(left, right))
            # Add global tokens (first N_g + last N_g)
            for g in range(N_g):
                if g not in k_indices:
                    k_indices.append(g)
            for g in range(T - N_g, T):
                if g not in k_indices:
                    k_indices.append(g)

            k_indices = torch.tensor(k_indices, device=q.device)
            k_i = k[:, k_indices]  # (B_HW, L, H, D_h)
            v_i = v[:, k_indices]  # (B_HW, L, H, D_h)
            q_i = q[:, i:i+1]       # (B_HW, 1, H, D_h)

            # Compute attention: q_i @ k_i^T
            attn = torch.einsum("bnhd,bmhd->bhnm", q_i, k_i) * self.scale  # (B_HW, H, 1, L)

            # Add relative position bias for local window
            rel_pos_start = i - left
            rel_pos_end = rel_pos_start + (right - left)
            rel_bias = self.rel_pos_bias[:, rel_pos_start:rel_pos_end]
            # Pad rel_bias for global tokens
            extra_tokens = len(k_indices) - rel_bias.shape[1]
            if extra_tokens > 0:
                rel_bias = torch.cat([
                    rel_bias,
                    torch.zeros(H, extra_tokens, device=q.device, dtype=q.dtype)
                ], dim=1)
            attn = attn + rel_bias[None, :, None, :]

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

            out_i = torch.einsum("bhnm,bmhd->bnhd", attn, v_i)  # (B_HW, 1, H, D_h)
            out[:, i:i+1] = out_i

        return out.reshape(B_HW, T, H * D_h)


class ShiftWindowTemporalAttention(BlockSparseTemporalAttention):
    """Shift-window temporal attention (Open-Sora v1.3 pattern).

    Alternates between regular and shifted window partitions across layers
    to enable cross-window information flow (Swin Transformer design).

    Layer 0, 2, 4, ... : Regular windows  [0,W), [W,2W), ...
    Layer 1, 3, 5, ... : Shifted windows [W/2, 3W/2), [3W/2, 5W/2), ...
    """

    def __init__(self, *args, **kwargs):
        kwargs["shift_window"] = True
        super().__init__(*args, **kwargs)


class LocalWindowTemporalAttention(BlockSparseTemporalAttention):
    """Local window attention only (no shift, no global tokens).

    Simplest sparse pattern: each frame attends to ±W neighbors.
    Best for inference speed.
    """

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8,
                 layer_idx: int = 0, **kwargs):
        super().__init__(
            dim=dim, num_heads=num_heads, window_size=window_size,
            num_global_tokens=0, shift_window=False,
            layer_idx=layer_idx, **kwargs,
        )
