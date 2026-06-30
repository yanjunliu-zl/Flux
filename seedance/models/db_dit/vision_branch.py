"""Vision Branch (STDiT block) — Spatial + Temporal self-attention for video tokens.

Each STDiT block processes video tokens through:
1. Spatial self-attention (within each frame)
2. Temporal self-attention (cross-frame at same spatial position)
3. Cross-attention to text embeddings
4. Feed-forward network

All sub-blocks use AdaLN conditioning on timestep.
"""

import torch
import torch.nn as nn

from seedance.models.db_dit.attention import MultiHeadAttention
from seedance.models.db_dit.adaln import AdaLN
from seedance.models.common.layers import MLP


class VisionBranchBlock(nn.Module):
    """Single STDiT block for the vision branch.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        cond_dim: Timestep conditioning dimension.
        ffn_ratio: FFN hidden dimension ratio (default: 4.0).
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        ffn_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Spatial self-attention
        self.adaln_spatial = AdaLN(dim, cond_dim)
        self.spatial_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout
        )

        # Temporal self-attention
        self.adaln_temporal = AdaLN(dim, cond_dim)
        self.temporal_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout
        )

        # Cross-attention to text
        self.adaln_cross = AdaLN(dim, cond_dim)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout, cross_attn=True
        )

        # Feed-forward network
        self.adaln_ffn = AdaLN(dim, cond_dim)
        ffn_hidden = int(dim * ffn_ratio)
        self.ffn = MLP(dim, ffn_hidden, dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        video_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Forward pass for one vision branch block.

        Args:
            x: Video tokens (B, T*H*W, D).
            t_emb: Timestep embedding (B, cond_dim).
            text_emb: Text embeddings (B, L_text, D).
            video_grid: (T, H, W) grid dimensions for reshape.

        Returns:
            Video tokens (B, T*H*W, D).
        """
        B, N, D = x.shape
        T, H, W = video_grid
        assert N == T * H * W, f"Token count {N} != T*H*W = {T}*{H}*{W} = {T*H*W}"

        # 1. Spatial Self-Attention (within each frame)
        x_norm, shift, scale, gate = self.adaln_spatial(x, t_emb)
        # Reshape: (B, T, H, W, D) -> (B*T, H*W, D)
        x_3d = x_norm.reshape(B, T, H, W, D)
        x_spatial = x_3d.reshape(B * T, H * W, D)
        # Apply spatial attention
        attn_out = self.spatial_attn(x_spatial)
        # Reshape back: (B*T, H*W, D) -> (B, T, H, W, D)
        attn_out = attn_out.reshape(B, T, H, W, D)
        x = x + gate.unsqueeze(1) * attn_out.reshape(B, N, D)

        # 2. Temporal Self-Attention (cross-frame at same spatial position)
        x_norm, shift, scale, gate = self.adaln_temporal(x, t_emb)
        # Reshape: (B, T*H*W, D) -> (B*H*W, T, D)
        x_temporal = x_norm.reshape(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        # Apply temporal attention
        attn_out = self.temporal_attn(x_temporal)
        # Reshape back: (B*H*W, T, D) -> (B, T*H*W, D)
        attn_out = attn_out.reshape(B, H * W, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + gate.unsqueeze(1) * attn_out

        # 3. Cross-Attention to Text
        x_norm, shift, scale, gate = self.adaln_cross(x, t_emb)
        attn_out = self.cross_attn(x_norm, context=text_emb)
        x = x + gate.unsqueeze(1) * attn_out

        # 4. Feed-Forward Network
        x_norm, shift, scale, gate = self.adaln_ffn(x, t_emb)
        ffn_out = self.ffn(x_norm)
        x = x + gate.unsqueeze(1) * ffn_out

        return x
