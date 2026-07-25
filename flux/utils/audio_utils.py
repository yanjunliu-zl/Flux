"""Audio I/O utilities."""

import torch
import torchaudio


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    duration_s: float | None = None,
) -> torch.Tensor:
    """Load audio file and resample.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate.
        duration_s: If set, truncate to this duration.

    Returns:
        Waveform tensor (1, T_samples).
    """
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Truncate
    if duration_s is not None:
        max_samples = int(sample_rate * duration_s)
        waveform = waveform[:, :max_samples]

    return waveform


def save_audio(
    waveform: torch.Tensor,
    output_path: str,
    sample_rate: int = 16000,
):
    """Save audio waveform to file.

    Args:
        waveform: Audio tensor (1, T_samples) or (T_samples,).
        output_path: Output file path.
        sample_rate: Sample rate for output file.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.detach().cpu().clamp(-1, 1)
    torchaudio.save(output_path, waveform, sample_rate)
