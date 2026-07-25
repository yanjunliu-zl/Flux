"""Causal 3D Convolution — ensures temporal causality (no future-frame leakage)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding.

    Pads only on the LEFT side of the temporal dimension:
      pad = (0, 0, 0, 0, kernel_t - 1, 0)
    This ensures frame t cannot see information from frame t+1.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: (T, H, W) kernel size.
        stride: (T, H, W) stride.
        padding: Additional spatial padding (handled separately from causal pad).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 1,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        # Spatial padding only — temporal padding is handled via causal pad
        self.spatial_padding = (padding[1], padding[2])

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(0, padding[1], padding[2]),  # No temporal padding yet
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        k_t = self.kernel_size[0]
        # Causal temporal padding: (k_t - 1) zeros on the left side
        p_t_left = k_t - 1
        p_t_right = 0

        # Use F.pad for flexible asymmetric temporal padding
        pad = (0, 0, 0, 0, p_t_left, p_t_right)  # (W_left, W_right, H_left, H_right, T_left, T_right)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)

    def __repr__(self):
        return (
            f"CausalConv3d(in={self.conv.in_channels}, out={self.conv.out_channels}, "
            f"kernel={self.kernel_size}, stride={self.stride})"
        )
