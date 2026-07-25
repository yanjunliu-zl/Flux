"""3D Video Decoder — reconstructs video frames from latent space."""

import torch
import torch.nn as nn

from flux.models.video_vae.causal_conv3d import CausalConv3d
from flux.models.video_vae.resnet3d import ResBlock3D


class Decoder3D(nn.Module):
    """3D Decoder with CausalConv3D backbone.

    Decodes latent space back to video frames:
    - 8x spatial upsampling (nearest + conv)
    - 4x temporal upsampling (nearest + conv)

    Args:
        latent_channels: Input latent channels (16).
        out_channels: Output channels (3 for RGB).
        base_channels: Base channel count at highest resolution.
        channel_multipliers: Channel multiplier per stage (reversed from encoder).
        spatial_strides: Spatial stride per stage (reversed from encoder).
        temporal_strides: Temporal stride per stage (reversed from encoder).
        num_res_blocks: Number of ResBlocks per stage.
        attn_resolutions: Spatial resolutions at which to apply attention.
        norm_groups: Number of GroupNorm groups.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: list[int] | None = None,
        spatial_strides: list[int] | None = None,
        temporal_strides: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        norm_groups: int = 32,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [4, 4, 2, 1]  # Reversed from encoder
        if spatial_strides is None:
            spatial_strides = [2, 2, 2, 1]
        if temporal_strides is None:
            temporal_strides = [1, 1, 2, 2]
        if attn_resolutions is None:
            attn_resolutions = [16]

        self.num_stages = len(channel_multipliers)

        # Input: project latent to base channels of first stage
        first_channels = base_channels * channel_multipliers[0]
        current_channels = first_channels

        self.conv_in = CausalConv3d(
            latent_channels,
            first_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1, 1),
        )

        # Upsampling stages
        self.stages = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            stage_channels = base_channels * channel_multipliers[stage_idx]
            s_stride = spatial_strides[stage_idx]
            t_stride = temporal_strides[stage_idx]

            stage_blocks = nn.ModuleList()

            # Attention block (applied before upsampling at specified resolutions)
            if attn_resolutions:
                from flux.models.video_vae.encoder_3d import AttentionBlock3D
                stage_blocks.append(AttentionBlock3D(current_channels))

            # Upsample if needed
            if s_stride > 1 or t_stride > 1:
                upsample = nn.Upsample(
                    scale_factor=(t_stride, s_stride, s_stride),
                    mode="nearest",
                )
                stage_blocks.append(upsample)

                # Conv after upsample to refine
                stage_blocks.append(
                    CausalConv3d(
                        current_channels,
                        stage_channels,
                        kernel_size=3,
                        stride=1,
                        padding=(1, 1, 1),
                    )
                )
                current_channels = stage_channels

            # ResNet blocks
            for i in range(num_res_blocks):
                next_channels = base_channels * channel_multipliers[min(stage_idx, self.num_stages - 1)]
                if i == 0 and (s_stride <= 1 and t_stride <= 1):
                    # If no upsample happened, we may still need channel transition
                    next_channels = stage_channels

                stage_blocks.append(
                    ResBlock3D(
                        in_channels=current_channels,
                        out_channels=stage_channels,
                        norm_groups=norm_groups,
                    )
                )
                current_channels = stage_channels

            self.stages.append(stage_blocks)

        # Output
        self.norm_out = nn.GroupNorm(min(norm_groups, current_channels), current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(
            current_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video frames.

        Args:
            z: Latent tensor (B, latent_channels, T_latent, H_latent, W_latent).

        Returns:
            Video tensor (B, 3, T, H, W) with temporal upsampling applied.
        """
        h = self.conv_in(z)

        for stage_blocks in self.stages:
            for block in stage_blocks:
                h = block(h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h
