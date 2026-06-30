"""Multi-head attention with Flash Attention and QK-norm support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional Flash Attention backend.

    Supports:
    - Self-attention and cross-attention
    - Optional QK-RMSNorm (from SD3/Flux)
    - Flash Attention 2/3 for memory-efficient training
    - Standard scaled dot-product attention as fallback

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        qk_norm: If True, apply RMSNorm to Q and K (default: False).
        dropout: Attention dropout rate (only used without flash-attn).
        cross_attn: If True, supports separate encoder hidden states.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cross_attn = cross_attn

        # QKV projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # QK normalization (from SD3/Flux)
        if qk_norm:
            from seedance.models.common.norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = dropout
        self.use_flash_attn = HAS_FLASH_ATTN and not cross_attn

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query tensor (B, N, D).
            context: Key/Value tensor for cross-attention (B, M, D). If None, self-attention.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor (B, N, D).
        """
        B, N, D = x.shape

        if context is None:
            context = x

        M = context.shape[1]

        # Project Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)

        # Apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_flash_attn and attn_mask is None:
            # Flash Attention expects (B, N, H, D) with contiguous layout
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = flash_attn_func(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, softmax_scale=self.scale)
            out = out.reshape(B, N, D)
        else:
            # Standard scaled dot-product attention
            q = q.transpose(1, 2)  # (B, H, N, D)
            k = k.transpose(1, 2)  # (B, H, M, D)
            v = v.transpose(1, 2)  # (B, H, M, D)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(out)
