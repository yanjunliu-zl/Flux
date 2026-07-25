"""VideoVAE: 3D Autoencoder for video compression.

8x spatial + 4x temporal compression using CausalConv3D backbone.
Supports loading pretrained SDXL VAE 2D weights with zero-init temporal layers.
"""

import torch
import torch.nn as nn

from flux.models.video_vae.encoder_3d import Encoder3D
from flux.models.video_vae.decoder_3d import Decoder3D


class VideoVAE(nn.Module):
    """3D Video Autoencoder with KL regularization.

    Encodes video (B, 3, T, H, W) to latent (B, 16, T/4, H/8, W/8) and back.

    Uses a diagonal Gaussian posterior with KL regularization.
    Supports loading 2D SDXL VAE weights and initializing temporal layers to zero.

    Args:
        in_channels: Input channels (3 for RGB).
        latent_channels: Latent space channels (16).
        base_channels: Base convolution channels.
        channel_multipliers: Channel multiplier per encoder stage.
        spatial_strides: Spatial stride per encoder stage.
        temporal_strides: Temporal stride per encoder stage.
        num_res_blocks: Number of ResBlocks per stage.
        attn_resolutions: Spatial resolutions for attention blocks.
        norm_groups: Number of GroupNorm groups.
        kl_weight: KL divergence weight.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: list[int] | None = None,
        spatial_strides: list[int] | None = None,
        temporal_strides: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
        norm_groups: int = 32,
        kl_weight: float = 1e-6,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.kl_weight = kl_weight

        channel_multipliers = channel_multipliers or [1, 2, 4, 4]
        spatial_strides = spatial_strides or [1, 2, 2, 2]
        temporal_strides = temporal_strides or [2, 2, 1, 1]
        attn_resolutions = attn_resolutions or [16]

        self.encoder = Encoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            spatial_strides=spatial_strides,
            temporal_strides=temporal_strides,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            norm_groups=norm_groups,
        )

        # Decoder: reverse the strides and channel progression
        decoder_channel_multipliers = list(reversed(channel_multipliers))
        decoder_spatial_strides = list(reversed(spatial_strides))
        decoder_temporal_strides = list(reversed(temporal_strides))

        self.decoder = Decoder3D(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=decoder_channel_multipliers,
            spatial_strides=decoder_spatial_strides,
            temporal_strides=decoder_temporal_strides,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            norm_groups=norm_groups,
        )

    def encode(
        self, x: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Encode video to latent representation.

        Args:
            x: Video tensor (B, 3, T, H, W).
            sample: If True, sample from posterior. If False, return mean.

        Returns:
            Latent tensor (B, latent_channels, T//4, H//8, W//8).
        """
        mean, logvar = self.encoder(x)

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video frames.

        Args:
            z: Latent tensor (B, latent_channels, T_latent, H_latent, W_latent).

        Returns:
            Video tensor (B, 3, T, H, W).
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for training.

        Args:
            x: Video tensor (B, 3, T, H, W).
            sample: If True, sample from posterior during training.

        Returns:
            Tuple of (reconstruction, posterior_sample, kl_loss).
        """
        mean, logvar = self.encoder(x)

        std = torch.exp(0.5 * logvar)
        if sample:
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        recon = self.decoder(z)

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3, 4])
        kl = kl.mean() * self.kl_weight

        return recon, z, kl

    def init_from_sdxl_vae(self, sdxl_vae_state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize 2D spatial weights from SDXL VAE checkpoint.

        Maps 2D Conv weights to the 3D Conv backbone. Temporal layers
        (which don't exist in SDXL VAE) remain at their initialization.

        Args:
            sdxl_vae_state_dict: State dict from stabilityai/sdxl-vae.
        """
        # Build a mapping from 2D conv keys to 3D conv keys
        own_state = self.state_dict()
        matched = 0
        skipped = 0

        for key_2d, weight_2d in sdxl_vae_state_dict.items():
            # Map 2D encoder/decoder keys to our 3D encoder/decoder keys
            if "encoder" in key_2d:
                key_3d = key_2d.replace("encoder.", "encoder.")
            elif "decoder" in key_2d:
                key_3d = key_2d.replace("decoder.", "decoder.")
            else:
                key_3d = key_2d

            # Handle Conv2d -> Conv3d weight expansion
            if key_3d in own_state and weight_2d.dim() == 4 and own_state[key_3d].dim() == 5:
                # Conv2d weight: (out_c, in_c, kH, kW)
                # Conv3d weight: (out_c, in_c, kT, kH, kW)
                # Center the 2D kernel in the 3D weight at temporal position kT//2
                w_3d = own_state[key_3d]
                kT = w_3d.shape[2]

                if weight_2d.shape[2] == w_3d.shape[3] and weight_2d.shape[3] == w_3d.shape[4]:
                    # Spatial kernel sizes match, expand temporally
                    expanded = torch.zeros_like(w_3d)
                    expanded[:, :, kT // 2, :, :] = weight_2d
                    own_state[key_3d] = expanded
                    matched += 1
                else:
                    skipped += 1
            elif key_3d in own_state and weight_2d.shape == own_state[key_3d].shape:
                # Direct weight assignment (norms, etc.)
                own_state[key_3d] = weight_2d
                matched += 1
            else:
                skipped += 1

        self.load_state_dict(own_state)
        print(
            f"[VideoVAE] Initialized from SDXL VAE: {matched} params matched, "
            f"{skipped} skipped (temporal layers zero-init)"
        )
