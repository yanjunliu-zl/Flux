"""Video data augmentation transforms."""

import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class VideoTransforms(nn.Module):
    """Video augmentation pipeline.

    Args:
        resolution: Target spatial resolution (square crop).
        num_frames: Number of frames to output.
        training: If True, apply random augmentations.
    """

    def __init__(
        self,
        resolution: int = 256,
        num_frames: int = 32,
        training: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_frames = num_frames
        self.training = training

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to video.

        Args:
            video: Video tensor (T, C, H, W).

        Returns:
            Augmented video tensor (T, C, resolution, resolution).
        """
        T, C, H, W = video.shape

        # Resize shorter side to resolution
        if H < W:
            new_h = self.resolution
            new_w = int(W * self.resolution / H)
        else:
            new_w = self.resolution
            new_h = int(H * self.resolution / W)

        video_resized = torch.zeros(T, C, new_h, new_w)
        for t in range(T):
            video_resized[t] = TF.resize(video[t], [new_h, new_w], antialias=True)

        # Random crop to resolution x resolution
        if self.training:
            top = random.randint(0, max(0, new_h - self.resolution))
            left = random.randint(0, max(0, new_w - self.resolution))
        else:
            top = (new_h - self.resolution) // 2
            left = (new_w - self.resolution) // 2

        video = video_resized[:, :, top:top + self.resolution, left:left + self.resolution]

        # Random horizontal flip
        if self.training and random.random() < 0.5:
            video = torch.flip(video, dims=[-1])

        # Color jitter (subtle)
        if self.training:
            brightness = 1.0 + random.uniform(-0.1, 0.1)
            contrast = 1.0 + random.uniform(-0.1, 0.1)
            saturation = 1.0 + random.uniform(-0.1, 0.1)
            for t in range(T):
                video[t] = TF.adjust_brightness(video[t], brightness)
                video[t] = TF.adjust_contrast(video[t], contrast)
                video[t] = TF.adjust_saturation(video[t], saturation)

        # Normalize to [-1, 1]
        video = 2.0 * video - 1.0

        return video.clamp(-1.0, 1.0)
