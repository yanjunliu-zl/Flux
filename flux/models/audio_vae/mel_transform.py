"""Mel-spectrogram transformations for audio processing.

Converts between waveform and mel-spectrogram representations.
Uses torchaudio for GPU-accelerated STFT/mel conversion.
"""

import torch
import torch.nn as nn
import torchaudio


class MelTransform(nn.Module):
    """Convert audio waveform to mel-spectrogram and back.

    Forward: waveform -> mel-spectrogram (in dB)
    Inverse: mel-spectrogram -> waveform (via Griffin-Lim or pretrained vocoder)

    Args:
        sample_rate: Audio sample rate (Hz).
        n_mels: Number of mel bins.
        hop_length: STFT hop length.
        win_length: STFT window length.
        n_fft: FFT size.
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz), None = sample_rate/2.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

        f_max = f_max or sample_rate / 2

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
        )

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,  # Magnitude spectrogram
        )

        self.inverse_mel_scale = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,
            n_iter=32,
        )

    def to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram.

        Args:
            waveform: Audio tensor (B, 1, T_samples) or (B, T_samples).

        Returns:
            Mel-spectrogram (B, 1, n_mels, T_frames).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        spec = self.spectrogram(waveform)  # (B, 1, n_fft//2+1, T_frames)
        mel = self.mel_scale(spec)  # (B, 1, n_mels, T_frames)
        # Log scale with small epsilon to avoid log(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram back to waveform via Griffin-Lim.

        Args:
            mel: Mel-spectrogram (B, 1, n_mels, T_frames), in log scale.

        Returns:
            Waveform (B, 1, T_samples).
        """
        # Convert back from log scale
        mel_linear = torch.exp(torch.clamp(mel, max=20))  # Avoid overflow
        spec = self.inverse_mel_scale(mel_linear)
        waveform = self.griffin_lim(spec)
        return waveform

    def get_output_length(self, input_samples: int) -> int:
        """Get the number of mel frames for a given number of audio samples.

        Args:
            input_samples: Number of audio samples.

        Returns:
            Number of mel-spectrogram time frames.
        """
        return (input_samples // self.hop_length) + 1
