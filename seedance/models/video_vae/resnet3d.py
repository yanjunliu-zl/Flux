"""3D ResNet blocks for VideoVAE with optional timestep conditioning."""

import torch
import torch.nn as nn

from seedance.models.video_vae.causal_conv3d import CausalConv3d


class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm + SiLU + CausalConv3D.

    Supports optional timestep embedding for conditioning (used in diffusion context
    but in VAE context, timestep_emb is typically None).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Conv kernel size (T, H, W).
        stride: Conv stride (T, H, W). If stride != 1, uses a 1x1x1 conv in skip.
        temb_channels: Timestep embedding channels (None = no conditioning).
        dropout: Dropout rate.
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        temb_channels: int | None = None,
        dropout: float = 0.0,
        norm_groups: int = 32,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Normalization and activation
        self.norm1 = nn.GroupNorm(
            min(norm_groups, in_channels), in_channels
        )
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size, stride, padding=(1, 1, 1)
        )

        self.norm2 = nn.GroupNorm(
            min(norm_groups, out_channels), out_channels
        )
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size, stride=1, padding=(1, 1, 1)
        )

        # Timestep embedding projection
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None

        # Skip connection
        self.use_skip_conv = (
            in_channels != out_channels
            or stride[0] != 1
            or stride[1] != 1
            or stride[2] != 1
        )
        if self.use_skip_conv:
            self.skip = CausalConv3d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.temb_proj is not None and temb is not None:
            # Add timestep embedding as channel-wise bias
            h = h + self.temb_proj(self.act1(temb))[:, :, None, None, None]

        skip = self.skip(x)
        return h + skip
