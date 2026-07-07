"""Video dataset for Stage 1: Video Pretraining.

Loads video clips from a CSV manifest, applies augmentations,
and returns processed video tensors with captions.
"""

import csv
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from seedance.data.transforms import VideoTransforms


class VideoDataset(Dataset):
    """Video-only dataset for Stage 1 pretraining.

    Reads a CSV manifest with columns:
        video_path, num_frames, height, width, fps, duration_s,
        caption_short, caption_long

    Args:
        manifest_path: Path to CSV manifest file.
        resolution: Target spatial resolution (square).
        num_frames: Number of frames to sample.
        frame_stride: Stride between sampled frames.
        fps_condition: If True, include FPS as conditioning info.
        caption_dropout_prob: Probability of dropping caption (for CFG training).
        use_short_caption: If True, use short captions; else use long.
    """

    def __init__(
        self,
        manifest_path: str,
        resolution: int = 256,
        num_frames: int = 32,
        frame_stride: int = 1,
        fps_condition: bool = True,
        caption_dropout_prob: float = 0.1,
        use_short_caption: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.fps_condition = fps_condition
        self.caption_dropout_prob = caption_dropout_prob
        self.use_short_caption = use_short_caption

        # Load manifest
        self.samples = []
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        self.transforms = VideoTransforms(
            resolution=resolution,
            num_frames=num_frames,
            training=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames from file.

        Returns tensor of shape (T, C, H, W).
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Determine which frames to read
        needed = self.num_frames * self.frame_stride
        if total_frames < needed:
            # Loop or pad if video is too short
            start_idx = 0
        else:
            start_idx = random.randint(0, total_frames - needed)

        frames = []
        frame_idx = start_idx
        while len(frames) < self.num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Loop back if needed
                frame_idx = start_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += self.frame_stride

        cap.release()

        # Stack and convert to tensor
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

        return video, fps

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_path = sample["video_path"]

        try:
            video, fps = self._load_video(video_path)
        except Exception as e:
            # Fallback: try another random sample
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        # Apply augmentations
        video = self.transforms(video)  # (T, C, H, W) -> (T, C, res, res)

        # Convert to (C, T, H, W) format expected by VideoVAE
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Caption
        caption = sample.get(
            "caption_short" if self.use_short_caption else "caption_long", ""
        )

        # CFG caption dropout
        if random.random() < self.caption_dropout_prob:
            caption = ""

        return {
            "video": video,
            "caption": caption,
            "fps": fps if self.fps_condition else 0,
        }
