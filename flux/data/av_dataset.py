"""Audio-Visual paired dataset for Stage 3/4: Joint Training."""

import csv
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import cv2

from flux.data.transforms import VideoTransforms


class AVDataset(Dataset):
    """Paired video-audio dataset for joint training.

    Reads a CSV manifest with columns:
        video_path, num_frames, height, width, fps, duration_s,
        audio_path, audio_sample_rate, audio_duration_s,
        caption_short, caption_long

    Args:
        manifest_path: Path to CSV manifest.
        video_resolution: Target video spatial resolution.
        video_num_frames: Number of frames to sample.
        video_frame_stride: Stride between frames.
        audio_sample_rate: Target audio sample rate.
        audio_n_mels: Number of mel bins.
        audio_hop_length: STFT hop length.
        caption_dropout_prob: CFG caption dropout.
        first_frame_condition_prob: Probability of first-frame conditioning (I2VA).
    """

    def __init__(
        self,
        manifest_path: str,
        video_resolution: int = 256,
        video_num_frames: int = 32,
        video_frame_stride: int = 1,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 80,
        audio_hop_length: int = 256,
        caption_dropout_prob: float = 0.1,
        first_frame_condition_prob: float = 0.3,
        use_short_caption: bool = True,
    ):
        super().__init__()
        self.video_resolution = video_resolution
        self.video_num_frames = video_num_frames
        self.video_frame_stride = video_frame_stride
        self.audio_sample_rate = audio_sample_rate
        self.audio_hop_length = audio_hop_length
        self.caption_dropout_prob = caption_dropout_prob
        self.first_frame_condition_prob = first_frame_condition_prob
        self.use_short_caption = use_short_caption
        self.max_audio_samples = int(audio_sample_rate * (video_num_frames / 16.0) + 0.5)

        self.samples = []
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        self.video_transforms = VideoTransforms(
            resolution=video_resolution,
            num_frames=video_num_frames,
            training=True,
        )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sample_rate,
            n_mels=audio_n_mels,
            hop_length=audio_hop_length,
            n_fft=1024,
            win_length=1024,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: str) -> tuple[torch.Tensor, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        needed = self.video_num_frames * self.video_frame_stride
        start_idx = random.randint(0, max(0, total_frames - needed)) if total_frames >= needed else 0

        frames = []
        frame_idx = start_idx
        while len(frames) < self.video_num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx = start_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += self.video_frame_stride

        cap.release()
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2)
        return video, fps

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] > self.max_audio_samples:
            waveform = waveform[:, :self.max_audio_samples]
        elif waveform.shape[1] < self.max_audio_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_audio_samples - waveform.shape[1])
            )

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        try:
            video, fps = self._load_video(sample["video_path"])
            mel = self._load_audio(sample.get("audio_path", sample["video_path"]))
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        video = self.video_transforms(video)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        caption = sample.get(
            "caption_short" if self.use_short_caption else "caption_long", ""
        )
        if random.random() < self.caption_dropout_prob:
            caption = ""

        # I2VA: first frame conditioning
        is_i2va = random.random() < self.first_frame_condition_prob
        first_frame = None
        if is_i2va:
            first_frame = video[:, 0:1, :, :].clone()

        return {
            "video": video,
            "mel": mel,
            "caption": caption,
            "fps": fps,
            "first_frame": first_frame,  # None for T2VA
        }
