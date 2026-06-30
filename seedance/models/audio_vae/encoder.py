"""AudioVAE Encoder — 2D Conv stack for mel-spectrogram compression."""

import torch
import torch.nn as nn


class ResBlock2D(nn.Module):
    """2D Residual Block: GroupNorm -> SiLU -> Conv2D -> GroupNorm -> SiLU -> Conv2D.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for first conv (downsampling).
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        norm_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(norm_groups, in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )

        self.norm2 = nn.GroupNorm(min(norm_groups, out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Skip connection
        self.use_skip = (
            in_channels != out_channels or stride[0] != 1 or stride[1] != 1
        )
        if self.use_skip:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AudioEncoder(nn.Module):
    """2D Conv encoder for mel-spectrogram compression.

    Compresses mel-spectrogram (B, 1, n_mels, T_frames) to latent (B, 2*z_dim, F, T').

    Args:
        in_channels: Input channels (1 for mel-spectrogram).
        latent_channels: Latent dimension (before mean/logvar split).
        base_channels: Base convolution channels.
        channel_multipliers: Channel multiplier per stage.
        strides: Stride per stage in (freq, time) format.
        num_res_blocks: Number of ResBlocks per stage.
        attn_resolutions: Reserved for future attention blocks.
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        strides: list[list[int]] | None = None,
        num_res_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        norm_groups: int = 32,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]
        if strides is None:
            strides = [[1, 2], [2, 2], [2, 1], [2, 1]]

        self.conv_in = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )

        current_channels = base_channels
        self.stages = nn.ModuleList()

        for stage_idx in range(len(channel_multipliers)):
            stage_channels = base_channels * channel_multipliers[stage_idx]
            stride = tuple(strides[stage_idx])

            stage_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                s = stride if i == 0 else (1, 1)
                stage_blocks.append(
                    ResBlock2D(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        stride=s,
                        norm_groups=norm_groups,
                    )
                )
                current_channels = stage_channels

            self.stages.append(stage_blocks)

        self.norm_out = nn.GroupNorm(min(norm_groups, current_channels), current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            current_channels,
            2 * latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mel-spectrogram.

        Args:
            x: Mel-spectrogram (B, 1, n_mels, T_frames).

        Returns:
            Tuple of (mean, logvar) each (B, latent_channels, F, T').
        """
        h = self.conv_in(x)

        for stage_blocks in self.stages:
            for block in stage_blocks:
                h = block(h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar
