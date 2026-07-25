"""NTK-aware Temporal RoPE scaling for long video sequences.

When training on 16-32 frames but generating 120+ frames (30s),
the RoPE position encodings need to be extended beyond training horizons.

Three interpolation strategies:
1. LINEAR — linearly rescale positions
2. NTK_AWARE — scale frequency base to compress high frequencies
3. YARN — NTK + temperature scaling (best for extreme extension)

Reference:
    NTK-Aware: bloc97, 2023
    YaRN: Peng et al., 2023
    LongCat-Video: NTK RoPE for minute-long generation
"""

import math
import torch
import torch.nn as nn


class NTKTemporalRoPE(nn.Module):
    """NTK-aware temporal RoPE for extending to longer sequences.

    Args:
        head_dim: Per-head dimension.
        max_train_frames: Max frames during training (e.g. 32).
        target_frames: Target inference frames (e.g. 128 for 30s).
        scale_method: "linear", "ntk", or "yarn".
        ntk_alpha: NTK scaling factor (higher = more conservative).
        theta: RoPE base frequency.
    """

    def __init__(
        self,
        head_dim: int = 128,
        max_train_frames: int = 32,
        target_frames: int = 128,
        scale_method: str = "ntk",
        ntk_alpha: float = 8.0,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_train_frames = max_train_frames
        self.target_frames = target_frames
        self.scale_method = scale_method
        self.ntk_alpha = ntk_alpha
        self.theta = theta
        self.scale_ratio = target_frames / max_train_frames

    def _compute_freqs(self, half: int, device, dtype) -> torch.Tensor:
        """Compute frequency basis based on scaling method."""
        if self.scale_method == "linear":
            return 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))

        elif self.scale_method == "ntk":
            scale = self.scale_ratio
            alpha = self.ntk_alpha
            freqs = []
            for i in range(half):
                lam = min(scale, alpha ** (i / max(half - 1, 1)))
                freqs.append(1.0 / (self.theta ** ((i / half) / lam)))
            return torch.tensor(freqs, device=device, dtype=dtype)

        elif self.scale_method == "yarn":
            scale = min(self.scale_ratio, 32.0)
            alpha = self.ntk_alpha
            half_half = half // 2
            freqs = []
            for i in range(half):
                if i < half_half:
                    ramp = i / max(half_half - 1, 1)
                    lam = scale ** (ramp * (alpha - 1))
                else:
                    lam = scale ** alpha
                freqs.append(1.0 / (self.theta ** (i / half) * lam))
            return torch.tensor(freqs, device=device, dtype=dtype)

        else:
            return 1.0 / (self.theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin tables for given sequence length."""
        half = self.head_dim // 2
        positions = torch.arange(seq_len, device=device, dtype=dtype)

        if self.scale_method == "linear":
            positions = positions / self.scale_ratio

        freqs = self._compute_freqs(half, device, dtype)

        angles = positions[:, None] * freqs[None, :]  # (T, half)
        angles = torch.cat([angles, angles], dim=-1)   # (T, head_dim)

        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def apply_rope(
        self, x: torch.Tensor, offset: int = 0,
    ) -> torch.Tensor:
        """Apply NTK-scaled RoPE to temporal attention tensors.

        Args:
            x: (..., T, head_dim) tensor.
            offset: Position offset for chunked processing.

        Returns:
            RoPE-rotated tensor of same shape.
        """
        T = x.shape[-2]
        cos, sin = self.get_cos_sin(T + offset, x.device, x.dtype)
        cos = cos[offset:offset+T]
        sin = sin[offset:offset+T]

        D = x.shape[-1]
        D_rot = D // 2
        x1, x2 = x[..., :D_rot], x[..., D_rot:2*D_rot]

        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_rot_1 = x1 * cos[..., :D_rot] - x2 * sin[..., :D_rot]
        x_rot_2 = x2 * cos[..., :D_rot] + x1 * sin[..., :D_rot]

        return torch.cat([x_rot_1, x_rot_2, x[..., 2*D_rot:]], dim=-1)
