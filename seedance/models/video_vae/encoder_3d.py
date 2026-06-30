"""3D Video Encoder with CausalConv3D — 8x spatial + 4x temporal compression."""

import torch
import torch.nn as nn

from seedance.models.video_vae.causal_conv3d import CausalConv3d
from seedance.models.video_vae.resnet3d import ResBlock3D


class AttentionBlock3D(nn.Module):
    """3D attention block: applies self-attention spatially within each frame.

    Reshapes the 5D tensor (B, C, T, H, W) to (B*T, H*W, C) for 2D self-attention.
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        h = self.norm(x)
        # Reshape to treat T as batch dim
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Self-attention via 1x1 conv QKV projection + scaled dot-product
        qkv = self.qkv(h)  # (B*T, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Flatten spatial dims for attention
        q = q.reshape(B * T, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B * T, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B * T, self.num_heads, C // self.num_heads, H * W)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(
            (q * scale) @ k.transpose(-2, -1), dim=-1
        )
        out = (attn @ v).reshape(B * T, C, H, W)
        out = self.proj(out)

        # Reshape back
        out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x + out


class Encoder3D(nn.Module):
    """3D Encoder with CausalConv3D backbone.

    Encodes video frames into latent space with:
    - 8x spatial compression (height, width)
    - 4x temporal compression

    Channel progression: 3 -> base_channels -> base_channels*2 -> base_channels*4 -> base_channels*4

    Args:
        in_channels: Input channels (3 for RGB).
        latent_channels: Output latent channels (16).
        base_channels: Base channel count (128).
        channel_multipliers: Multiplier per stage [1, 2, 4, 4].
        spatial_strides: Spatial stride per stage.
        temporal_strides: Temporal stride per stage.
        num_res_blocks: Number of ResBlocks per stage.
        attn_resolutions: Spatial resolutions at which to apply attention.
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: list[int] = None,
        spatial_strides: list[int] = None,
        temporal_strides: list[int] = None,
        num_res_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        norm_groups: int = 32,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]
        if spatial_strides is None:
            spatial_strides = [1, 2, 2, 2]
        if temporal_strides is None:
            temporal_strides = [2, 2, 1, 1]
        if attn_resolutions is None:
            attn_resolutions = [16]

        self.num_stages = len(channel_multipliers)

        # Input projection
        self.conv_in = CausalConv3d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1, 1),
        )

        # Downsampling stages
        current_channels = base_channels
        self.stages = nn.ModuleList()

        for stage in range(self.num_stages):
            stage_channels = base_channels * channel_multipliers[stage]
            s_stride = spatial_strides[stage]
            t_stride = temporal_strides[stage]

            stage_blocks = nn.ModuleList()

            # ResNet blocks
            for i in range(num_res_blocks):
                stride = (t_stride if i == 0 else 1, s_stride if i == 0 else 1, s_stride if i == 0 else 1)
                stage_blocks.append(
                    ResBlock3D(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        stride=stride,
                        norm_groups=norm_groups,
                    )
                )
                current_channels = stage_channels

            # Attention block at specified resolutions
            if any(r >= 16 for r in attn_resolutions):  # Will be checked per resolution
                stage_blocks.append(AttentionBlock3D(current_channels))

            self.stages.append(stage_blocks)

        # Output: project to 2 * latent_channels (mean + logvar)
        self.norm_out = nn.GroupNorm(min(norm_groups, current_channels), current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(
            current_channels,
            2 * latent_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video frames.

        Args:
            x: Video tensor of shape (B, 3, T, H, W).

        Returns:
            Tuple of (mean, logvar) each of shape (B, latent_channels, T//4, H//8, W//8).
        """
        h = self.conv_in(x)

        for stage_blocks in self.stages:
            for block in stage_blocks:
                h = block(h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar
