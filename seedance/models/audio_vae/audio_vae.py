"""AudioVAE: 2D Conv Autoencoder for mel-spectrogram compression.

Compresses 80-bin mel-spectrograms to a compact latent space.
Uses the same KL-regularized VAE framework as VideoVAE.
"""

import torch
import torch.nn as nn

from seedance.models.audio_vae.encoder import AudioEncoder
from seedance.models.audio_vae.decoder import AudioDecoder
from seedance.models.audio_vae.mel_transform import MelTransform


class AudioVAE(nn.Module):
    """Audio mel-spectrogram VAE.

    Full pipeline: waveform -> mel -> encode -> latent -> decode -> mel -> waveform.

    Args:
        sample_rate: Audio sample rate (Hz).
        n_mels: Number of mel bins (80).
        hop_length: STFT hop length.
        latent_channels: Latent space channels.
        base_channels: Base convolution channels.
        channel_multipliers: Channel multiplier per stage.
        strides: Stride per stage (freq, time).
        num_res_blocks: Number of ResBlocks per stage.
        norm_groups: Number of GroupNorm groups.
        kl_weight: KL divergence weight.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 256,
        latent_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: list[int] | None = None,
        strides: list[list[int]] | None = None,
        num_res_blocks: int = 2,
        norm_groups: int = 32,
        kl_weight: float = 1e-6,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.kl_weight = kl_weight

        # Mel transform (not used during VAE training directly, but available)
        self.mel_transform = MelTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        )

        self.encoder = AudioEncoder(
            in_channels=1,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            strides=strides,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
        )

        decoder_channel_multipliers = (
            list(reversed(channel_multipliers)) if channel_multipliers else None
        )
        decoder_strides = (
            [list(reversed(s)) for s in reversed(strides)] if strides else None
        )

        self.decoder = AudioDecoder(
            latent_channels=latent_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_multipliers=decoder_channel_multipliers,
            strides=decoder_strides,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
        )

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode mel-spectrogram to latent.

        Args:
            x: Mel-spectrogram (B, 1, n_mels, T_frames).
            sample: If True, sample from posterior.

        Returns:
            Latent tensor (B, latent_channels, F, T').
        """
        mean, logvar = self.encoder(x)

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to mel-spectrogram.

        Args:
            z: Latent tensor (B, latent_channels, F, T').

        Returns:
            Mel-spectrogram (B, 1, n_mels, T_frames).
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for training.

        Args:
            x: Mel-spectrogram (B, 1, n_mels, T_frames).
            sample: If True, sample from posterior.

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

        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl = kl.mean() * self.kl_weight

        return recon, z, kl

    def waveform_to_latent(
        self, waveform: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Full pipeline: waveform -> mel -> encode -> latent.

        Args:
            waveform: Audio waveform (B, 1, T_samples).
            sample: If True, sample from posterior.

        Returns:
            Latent tensor.
        """
        mel = self.mel_transform.to_mel(waveform)
        return self.encode(mel, sample=sample)

    def latent_to_waveform(self, z: torch.Tensor) -> torch.Tensor:
        """Full pipeline: latent -> decode -> mel -> waveform.

        Args:
            z: Latent tensor.

        Returns:
            Audio waveform (B, 1, T_samples).
        """
        mel = self.decode(z)
        return self.mel_transform.to_waveform(mel)
