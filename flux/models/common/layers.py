"""Common neural network layer building blocks."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden dimension (default: 4x in_dim).
        out_dim: Output dimension (default: same as in_dim).
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv2dBlock(nn.Module):
    """2D Convolution block: Conv2d -> GroupNorm -> SiLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Conv3dBlock(nn.Module):
    """3D Convolution block: Conv3d -> GroupNorm -> SiLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
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
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
