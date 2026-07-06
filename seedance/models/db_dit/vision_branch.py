"""Vision Branch (STDiT block) — Spatial + Temporal self-attention for video tokens.

Each STDiT block processes video tokens through:
1. Spatial self-attention (within each frame)
2. Temporal self-attention (cross-frame at same spatial position)
3. Cross-attention to text embeddings
4. Feed-forward network (dense MLP or MoE)

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
        context_dim: Dimension of cross-attention text context.
                     Defaults to ``dim`` for same-dim text encoder (e.g. T5-large).
        ffn_ratio: FFN hidden dimension ratio (default: 4.0).
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
        moe_config: If provided, use MoE FFN instead of dense MLP.
                    Dict with keys: num_experts, top_k, expert_dim_ratio.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        context_dim: int | None = None,
        ffn_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        moe_config: dict | None = None,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
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

        # Cross-attention to text (K/V dim = context_dim, e.g. T5-base=768)
        self.adaln_cross = AdaLN(dim, cond_dim)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, context_dim=context_dim,
            qk_norm=qk_norm, dropout=dropout, cross_attn=True,
        )

        # Feed-forward network (dense MLP or MoE)
        self.adaln_ffn = AdaLN(dim, cond_dim)
        if moe_config is not None:
            from seedance.models.db_dit.moe import MoEFFN
            self.ffn = MoEFFN(
                dim=dim,
                num_experts=moe_config.get("num_experts", 32),
                top_k=moe_config.get("top_k", 2),
                expert_dim_ratio=moe_config.get("expert_dim_ratio", 1.0),
                shared_expert=moe_config.get("shared_expert", True),
            )
            self._use_moe = True
        else:
            ffn_hidden = int(dim * ffn_ratio)
            self.ffn = MLP(dim, ffn_hidden, dim, dropout=dropout)
            self._use_moe = False

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        video_grid: tuple[int, int, int],
    ) -> tuple[torch.Tensor, dict | None]:
        """Forward pass for one vision branch block.

        Args:
            x: Video tokens (B, T*H*W, D).
            t_emb: Timestep embedding (B, cond_dim).
            text_emb: Text embeddings (B, L_text, D).
            video_grid: (T, H, W) grid dimensions for reshape.

        Returns:
            Tuple of (video tokens, moe_aux_losses | None).
        """
        B, N, D = x.shape
        T, H, W = video_grid
        assert N == T * H * W, f"Token count {N} != T*H*W = {T}*{H}*{W} = {T*H*W}"
        moe_aux = None

        # 1. Spatial Self-Attention (within each frame)
        x_norm, shift, scale, gate = self.adaln_spatial(x, t_emb)
        x_3d = x_norm.reshape(B, T, H, W, D)
        x_spatial = x_3d.reshape(B * T, H * W, D)
        attn_out = self.spatial_attn(x_spatial)
        attn_out = attn_out.reshape(B, T, H, W, D)
        x = x + gate.unsqueeze(1) * attn_out.reshape(B, N, D)

        # 2. Temporal Self-Attention (cross-frame at same spatial position)
        x_norm, shift, scale, gate = self.adaln_temporal(x, t_emb)
        x_temporal = x_norm.reshape(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        attn_out = self.temporal_attn(x_temporal)
        attn_out = attn_out.reshape(B, H * W, T, D).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + gate.unsqueeze(1) * attn_out

        # 3. Cross-Attention to Text
        x_norm, shift, scale, gate = self.adaln_cross(x, t_emb)
        attn_out = self.cross_attn(x_norm, context=text_emb)
        x = x + gate.unsqueeze(1) * attn_out

        # 4. Feed-Forward Network
        x_norm, shift, scale, gate = self.adaln_ffn(x, t_emb)
        if self._use_moe:
            ffn_out, moe_aux = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
        x = x + gate.unsqueeze(1) * ffn_out

        return x, moe_aux
