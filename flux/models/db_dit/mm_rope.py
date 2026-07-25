"""Multi-Modal Rotary Position Embedding (MM-RoPE).

Encodes three positional dimensions:
  - Temporal (t): frame index in video
  - Spatial H (h): row position within a frame
  - Spatial W (w): column position within a frame
  - Audio time (t_a): audio event position

Each dimension gets its own frequency subspace via block-diagonal RoPE.
Following the design from Seedance 1.0/1.5 described MMDiT papers.
"""

import math
import torch
import torch.nn as nn


class MMRoPE(nn.Module):
    """Multi-Modal Rotary Position Embedding.

    Applies 3D RoPE for video tokens and 1D RoPE for audio tokens.
    Uses separate frequency bases per axis so the model can distinguish
    temporal shifts from spatial shifts from audio shifts.

    Args:
        dim: Total head dimension.
        rope_dim_t: Dimensions for temporal axis.
        rope_dim_h: Dimensions for height axis.
        rope_dim_w: Dimensions for width axis.
        rope_dim_a: Dimensions for audio axis (defaults to dim).
        theta: Base frequency for RoPE (default: 10000.0).
    """

    def __init__(
        self,
        dim: int,
        rope_dim_t: int | None = None,
        rope_dim_h: int | None = None,
        rope_dim_w: int | None = None,
        rope_dim_a: int | None = None,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta

        # Default: split evenly
        if rope_dim_t is None and rope_dim_h is None and rope_dim_w is None:
            rope_dim_t = dim // 3
            rope_dim_h = dim // 3
            rope_dim_w = dim - rope_dim_t - rope_dim_h

        self.rope_dim_t = rope_dim_t or 0
        self.rope_dim_h = rope_dim_h or 0
        self.rope_dim_w = rope_dim_w or 0

        if rope_dim_a is None:
            rope_dim_a = dim
        self.rope_dim_a = rope_dim_a

        # Precompute frequencies for each axis
        self.register_buffer("freqs_t", self._compute_freqs(self.rope_dim_t, theta), persistent=False)
        self.register_buffer("freqs_h", self._compute_freqs(self.rope_dim_h, theta), persistent=False)
        self.register_buffer("freqs_w", self._compute_freqs(self.rope_dim_w, theta), persistent=False)
        self.register_buffer("freqs_a", self._compute_freqs(rope_dim_a, theta), persistent=False)

    @staticmethod
    def _compute_freqs(dim: int, theta: float) -> torch.Tensor:
        """Compute frequency tensor for RoPE.

        Returns tensor of shape (dim // 2,) with exponentially decreasing frequencies.
        """
        if dim <= 0:
            return torch.zeros(0)
        half = dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        return freqs

    def rope_1d(
        self, pos: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D RoPE cos/sin tables for given positions.

        Args:
            pos: Position indices (N,) or (B, N).
            freqs: Frequency tensor (D//2,).

        Returns:
            (cos, sin) each of shape (N, D) or (B, N, D).
        """
        if freqs.numel() == 0:
            return torch.ones(*pos.shape, 0, device=pos.device), torch.ones(
                *pos.shape, 0, device=pos.device
            )

        # (N, 1) * (D//2,) -> (N, D//2)
        angles = pos.float().unsqueeze(-1) * freqs.to(pos.device)
        # Repeat for full dimension (cos, cos, sin, sin pattern)
        angles = torch.cat([angles, angles], dim=-1)  # (N, D)
        return torch.cos(angles), torch.sin(angles)

    def apply_rope_1d(
        self, x: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        """Apply 1D RoPE to input tensor.

        Args:
            x: (..., N, D) where D is head_dim.
            pos: (N,) position indices.

        Returns:
            Rotated tensor of same shape.
        """
        D = x.shape[-1]
        cos, sin = self.rope_1d(pos, self.freqs_a.to(x.device))

        if cos.shape[-1] < D:
            # Pad: non-RoPE dimensions pass through unchanged
            pad_dim = D - cos.shape[-1]
            cos = torch.cat([cos, torch.ones(*cos.shape[:-1], pad_dim, device=x.device)], dim=-1)
            sin = torch.cat([sin, torch.zeros(*sin.shape[:-1], pad_dim, device=x.device)], dim=-1)

        # Rotate half the dimensions
        D_rot = D // 2
        x1, x2 = x[..., :D_rot], x[..., D_rot : 2 * D_rot]
        cos_rot = cos[..., :D_rot]
        sin_rot = sin[..., :D_rot]

        x_rot_1 = x1 * cos_rot - x2 * sin_rot
        x_rot_2 = x2 * cos_rot + x1 * sin_rot

        return torch.cat([x_rot_1, x_rot_2, x[..., 2 * D_rot:]], dim=-1)

    def apply_rope_3d(
        self,
        x: torch.Tensor,
        t_pos: torch.Tensor,
        h_pos: torch.Tensor,
        w_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 3D RoPE for video tokens.

        Each axis occupies a contiguous subspace of the head dimension:
        [temporal_dim | height_dim | width_dim | remaining]

        Args:
            x: (B, T, H, W, D) video token features.
            t_pos: (T,) temporal positions.
            h_pos: (H,) height positions.
            w_pos: (W,) width positions.

        Returns:
            Rotated tensor of same shape (B, T, H, W, D).
        """
        D = x.shape[-1]
        B, T, H, W, _ = x.shape

        # Temporal RoPE
        if self.rope_dim_t > 0:
            t_cos, t_sin = self.rope_1d(t_pos, self.freqs_t)
            t_cos = t_cos[None, :, None, None, :]  # (1, T, 1, 1, D_t)
            t_sin = t_sin[None, :, None, None, :]
            D_t = self.rope_dim_t
            D_tr = D_t // 2
            x_t = x[..., :D_tr]
            x_t_next = x[..., D_tr : 2 * D_tr]
            x_rot_t1 = x_t * t_cos[..., :D_tr] - x_t_next * t_sin[..., :D_tr]
            x_rot_t2 = x_t_next * t_cos[..., :D_tr] + x_t * t_sin[..., :D_tr]
            x = torch.cat([x_rot_t1, x_rot_t2, x[..., 2 * D_tr:]], dim=-1)

        # Height RoPE
        if self.rope_dim_h > 0:
            offset = self.rope_dim_t
            h_cos, h_sin = self.rope_1d(h_pos, self.freqs_h)
            h_cos = h_cos[None, None, :, None, :]  # (1, 1, H, 1, D_h)
            h_sin = h_sin[None, None, :, None, :]
            D_h = self.rope_dim_h
            D_hr = D_h // 2
            x_h = x[..., offset : offset + D_hr]
            x_h_next = x[..., offset + D_hr : offset + 2 * D_hr]
            x_rot_h1 = x_h * h_cos[..., :D_hr] - x_h_next * h_sin[..., :D_hr]
            x_rot_h2 = x_h_next * h_cos[..., :D_hr] + x_h * h_sin[..., :D_hr]
            x_h_part = x[..., offset : offset + D_h]
            x_h_part = torch.cat(
                [x_rot_h1, x_rot_h2, x_h_part[..., 2 * D_hr:]], dim=-1
            )
            x = torch.cat(
                [x[..., :offset], x_h_part, x[..., offset + D_h:]], dim=-1
            )

        # Width RoPE
        if self.rope_dim_w > 0:
            offset = self.rope_dim_t + self.rope_dim_h
            w_cos, w_sin = self.rope_1d(w_pos, self.freqs_w)
            w_cos = w_cos[None, None, None, :, :]  # (1, 1, 1, W, D_w)
            w_sin = w_sin[None, None, None, :, :]
            D_w = self.rope_dim_w
            D_wr = D_w // 2
            x_w = x[..., offset : offset + D_wr]
            x_w_next = x[..., offset + D_wr : offset + 2 * D_wr]
            x_rot_w1 = x_w * w_cos[..., :D_wr] - x_w_next * w_sin[..., :D_wr]
            x_rot_w2 = x_w_next * w_cos[..., :D_wr] + x_w * w_sin[..., :D_wr]
            x_w_part = x[..., offset : offset + D_w]
            x_w_part = torch.cat(
                [x_rot_w1, x_rot_w2, x_w_part[..., 2 * D_wr:]], dim=-1
            )
            x = torch.cat(
                [x[..., :offset], x_w_part, x[..., offset + D_w:]], dim=-1
            )

        return x

    def get_video_positions(
        self, T: int, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate 3D position indices for video tokens.

        Args:
            T, H, W: Grid dimensions.
            device: Target device.

        Returns:
            Tuple of (t_pos, h_pos, w_pos) each of shape (T,), (H,), (W,).
        """
        t_pos = torch.arange(T, device=device, dtype=torch.float32)
        h_pos = torch.arange(H, device=device, dtype=torch.float32)
        w_pos = torch.arange(W, device=device, dtype=torch.float32)
        return t_pos, h_pos, w_pos

    def get_audio_positions(
        self, N: int, device: torch.device
    ) -> torch.Tensor:
        """Generate 1D position indices for audio tokens.

        Args:
            N: Number of audio tokens.
            device: Target device.

        Returns:
            Position tensor of shape (N,).
        """
        return torch.arange(N, device=device, dtype=torch.float32)
