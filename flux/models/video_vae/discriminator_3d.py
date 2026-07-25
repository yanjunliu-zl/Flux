"""3D PatchGAN Discriminator for adversarial VAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator with spectral normalization.

    Operates on 5D video tensors (B, C, T, H, W).
    Outputs a patch-based real/fake prediction.

    Args:
        in_channels: Input channels (3 for RGB video).
        ndf: Base number of discriminator filters (default: 64).
        num_layers: Number of downsampling layers (default: 4).
        use_spectral_norm: Whether to apply spectral norm (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        conv_layer = (
            lambda *args, **kwargs: nn.utils.spectral_norm(
                nn.Conv3d(*args, **kwargs)
            )
            if use_spectral_norm
            else nn.Conv3d
        )

        layers = []

        # First layer: no normalization
        layers.append(
            conv_layer(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.append(
                conv_layer(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2 if n < num_layers - 1 else 1,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm3d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final layer: project to 1 channel (real/fake score)
        layers.append(
            conv_layer(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass, returns intermediate features for feature matching loss.

        Args:
            x: Video tensor (B, 3, T, H, W).

        Returns:
            List of feature maps from each layer, ending with logits.
        """
        features = []
        for layer in self.main:
            x = layer(x)
            features.append(x)
        return features
