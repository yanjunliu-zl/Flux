"""AudioVAE Decoder — 2D Conv stack for mel-spectrogram reconstruction."""

import torch
import torch.nn as nn

from seedance.models.audio_vae.encoder import ResBlock2D


class AudioDecoder(nn.Module):
    """2D Conv decoder for mel-spectrogram reconstruction.

    Args:
        latent_channels: Latent channels.
        out_channels: Output channels (1 for mel-spectrogram).
        base_channels: Base convolution channels (at highest resolution).
        channel_multipliers: Channel multiplier per stage (reversed from encoder).
        strides: Upsample stride per stage in (freq, time).
        num_res_blocks: Number of ResBlocks per stage.
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        strides: list[list[int]] | None = None,
        num_res_blocks: int = 2,
        norm_groups: int = 32,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [4, 4, 2, 1]  # Reversed from encoder
        if strides is None:
            strides = [[2, 1], [2, 1], [2, 2], [1, 2]]

        first_channels = base_channels * channel_multipliers[0]

        self.conv_in = nn.Conv2d(
            latent_channels, first_channels, kernel_size=3, stride=1, padding=1
        )

        current_channels = first_channels
        self.stages = nn.ModuleList()

        for stage_idx in range(len(channel_multipliers)):
            stage_channels = base_channels * channel_multipliers[stage_idx]
            stride = strides[stage_idx]
            do_upsample = stride[0] > 1 or stride[1] > 1

            stage_blocks = nn.ModuleList()

            if do_upsample:
                stage_blocks.append(
                    nn.Upsample(scale_factor=tuple(stride), mode="nearest")
                )
                stage_blocks.append(
                    nn.Conv2d(
                        current_channels, stage_channels,
                        kernel_size=3, stride=1, padding=1,
                    )
                )
                current_channels = stage_channels

            for i in range(num_res_blocks):
                stage_blocks.append(
                    ResBlock2D(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        norm_groups=norm_groups,
                    )
                )
                current_channels = stage_channels

            self.stages.append(stage_blocks)

        self.norm_out = nn.GroupNorm(min(norm_groups, current_channels), current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            current_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to mel-spectrogram.

        Args:
            z: Latent tensor (B, latent_channels, F, T').

        Returns:
            Mel-spectrogram (B, 1, n_mels, T_frames).
        """
        h = self.conv_in(z)

        for stage_blocks in self.stages:
            for block in stage_blocks:
                h = block(h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h
