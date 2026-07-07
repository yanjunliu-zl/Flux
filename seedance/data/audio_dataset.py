"""Audio dataset for Stage 2: Audio Pretraining."""

import csv
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Audio dataset for Stage 2 pretraining.

    Reads a CSV manifest with columns:
        audio_path, audio_sample_rate, audio_duration_s,
        caption_short, caption_long

    Args:
        manifest_path: Path to CSV manifest.
        sample_rate: Target sample rate (Hz).
        n_mels: Number of mel bins.
        hop_length: STFT hop length.
        max_duration_s: Maximum audio duration in seconds.
        caption_dropout_prob: CFG caption dropout probability.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 256,
        max_duration_s: float = 10.0,
        caption_dropout_prob: float = 0.1,
        use_short_caption: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.max_duration_s = max_duration_s
        self.caption_dropout_prob = caption_dropout_prob
        self.use_short_caption = use_short_caption
        self.max_samples = int(sample_rate * max_duration_s)

        # Load manifest
        self.samples = []
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=1024,
            win_length=1024,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate or pad
        if waveform.shape[1] > self.max_samples:
            start = random.randint(0, waveform.shape[1] - self.max_samples)
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_samples - waveform.shape[1])
            )

        # Compute mel-spectrogram
        mel = self.mel_transform(waveform)  # (1, n_mels, T_frames)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Caption
        caption = sample.get(
            "caption_short" if self.use_short_caption else "caption_long", ""
        )
        if random.random() < self.caption_dropout_prob:
            caption = ""

        return {
            "mel": mel,
            "caption": caption,
        }
