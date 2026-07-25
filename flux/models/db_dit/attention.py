"""Multi-head attention with memory-efficient attention backends.

Uses the best available backend in priority order:
  1. xformers memory-efficient attention (Windows/Linux, no build required)
  2. flash-attn (optional, Linux only)
  3. PyTorch F.scaled_dot_product_attention (universal fallback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops
    import torch

    # xformers CUDA kernels only support up to compute capability 9.0 (H100).
    # Blackwell GPUs (10.0+) are not supported yet.
    _gpu_cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    HAS_XFORMERS = _gpu_cap <= 9
    if not HAS_XFORMERS:
        import warnings
        warnings.warn(
            f"xformers does not support GPU compute capability {_gpu_cap}.0 (Blackwell). "
            f"Falling back to PyTorch SDPA."
        )
except ImportError:
    HAS_XFORMERS = False

try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention with memory-efficient backend.

    Supports:
    - Self-attention and cross-attention
    - xformers memory-efficient attention (primary backend)
    - Flash Attention 2/3 (optional, Linux only)
    - PyTorch SDPA (universal fallback)
    - Optional QK-RMSNorm (from SD3/Flux)

    Args:
        dim: Model dimension for query (and key/value when context_dim is None).
        num_heads: Number of attention heads.
        context_dim: Dimension of cross-attention context (key/value).
        qk_norm: If True, apply RMSNorm to Q and K.
        dropout: Attention dropout rate.
        cross_attn: If True, supports separate encoder hidden states.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        context_dim: int | None = None,
        qk_norm: bool = False,
        dropout: float = 0.0,
        cross_attn: bool = False,
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cross_attn = cross_attn

        # QKV projections (K/V use context_dim for cross-attention)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # QK normalization (from SD3/Flux)
        if qk_norm:
            from flux.models.common.norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = dropout

        # Select best available backend.
        # Priority: xformers (Windows workaround) > flash-attn (Linux opt-in) > SDPA.
        # On Linux, PyTorch SDPA natively includes Flash Attention kernels — no
        # extra package is needed. xformers is only relevant on older GPUs where
        # PyTorch's built-in flash kernels are missing (e.g. some Windows builds).
        if HAS_XFORMERS:
            self._backend = "xformers"
        elif HAS_FLASH_ATTN:
            self._backend = "flash_attn"
        else:
            self._backend = "sdpa"

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query tensor (B, N, D).
            context: Key/Value tensor for cross-attention (B, M, D_context).
                     If None, self-attention.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor (B, N, D).
        """
        B, N, D = x.shape

        if context is None:
            context = x

        M = context.shape[1]

        # Project Q, K, V -> (B, H, N, head_dim) or (B, H, M, head_dim)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim)

        # Apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        dropout_p = self.attn_dropout if self.training else 0.0

        if self._backend == "xformers" and attn_mask is None:
            # xformers memory-efficient attention — best on Windows
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = xops.memory_efficient_attention(
                q, k, v, p=dropout_p, scale=self.scale,
            )
            out = out.reshape(B, N, D)

        elif self._backend == "flash_attn" and attn_mask is None and q.dtype in (torch.float16, torch.bfloat16):
            # Flash Attention 2/3 — Linux only, requires fp16 or bf16
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale)
            out = out.reshape(B, N, D)

        else:
            # PyTorch SDPA — universal fallback (handles masks natively)
            q = q.transpose(1, 2)  # (B, H, N, head_dim)
            k = k.transpose(1, 2)  # (B, H, M, head_dim)
            v = v.transpose(1, 2)  # (B, H, M, head_dim)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(out)

