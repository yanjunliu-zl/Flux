"""Audio Branch (DiT block) — Self-attention + Cross-attention for audio tokens.

Simpler than the vision branch: audio tokens have a 1D sequence structure
(flattened frequency × time), so only one self-attention is needed.
"""

import torch
import torch.nn as nn

from seedance.models.db_dit.attention import MultiHeadAttention
from seedance.models.db_dit.adaln import AdaLN
from seedance.models.common.layers import MLP


class AudioBranchBlock(nn.Module):
    """Single DiT block for the audio branch.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        cond_dim: Timestep conditioning dimension.
        context_dim: Dimension of cross-attention text context.
        ffn_ratio: FFN hidden dimension ratio.
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
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
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.dim = dim

        # Self-attention
        self.adaln_self = AdaLN(dim, cond_dim)
        self.self_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout
        )

        # Cross-attention to text (K/V dim = context_dim)
        self.adaln_cross = AdaLN(dim, cond_dim)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, context_dim=context_dim,
            qk_norm=qk_norm, dropout=dropout, cross_attn=True,
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
    ) -> torch.Tensor:
        """Forward pass for one audio branch block.

        Args:
            x: Audio tokens (B, N_a, D).
            t_emb: Timestep embedding (B, cond_dim).
            text_emb: Text embeddings (B, L_text, D).

        Returns:
            Audio tokens (B, N_a, D).
        """
        # 1. Self-Attention
        x_norm, shift, scale, gate = self.adaln_self(x, t_emb)
        attn_out = self.self_attn(x_norm)
        x = x + gate.unsqueeze(1) * attn_out

        # 2. Cross-Attention to Text
        x_norm, shift, scale, gate = self.adaln_cross(x, t_emb)
        attn_out = self.cross_attn(x_norm, context=text_emb)
        x = x + gate.unsqueeze(1) * attn_out

        # 3. Feed-Forward Network
        x_norm, shift, scale, gate = self.adaln_ffn(x, t_emb)
        ffn_out = self.ffn(x_norm)
        x = x + gate.unsqueeze(1) * ffn_out

        return x
