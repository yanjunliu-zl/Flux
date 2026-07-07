"""Batch collation for variable-length video and audio data."""

import torch


def collate_video_batch(batch: list[dict]) -> dict:
    """Collate a batch of video-only samples.

    Args:
        batch: List of dicts with keys "video", "caption", "fps".

    Returns:
        Collated dict with batched tensors.
    """
    videos = torch.stack([item["video"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    fps = torch.tensor([item["fps"] for item in batch])
    return {"video": videos, "caption": captions, "fps": fps}


def collate_audio_batch(batch: list[dict]) -> dict:
    """Collate a batch of audio-only samples."""
    mels = torch.stack([item["mel"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    return {"mel": mels, "caption": captions}


def collate_av_batch(batch: list[dict]) -> dict:
    """Collate a batch of audio-video paired samples.

    Handles first-frame conditioning (I2VA) by creating
    a mask that marks which frames should be denoised.
    """
    videos = torch.stack([item["video"] for item in batch], dim=0)
    mels = torch.stack([item["mel"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    fps = torch.tensor([item.get("fps", 0) for item in batch])

    # First-frame conditioning
    has_first_frame = any(item.get("first_frame") is not None for item in batch)
    first_frames = None
    first_frame_mask = None
    if has_first_frame:
        first_frames = []
        for item in batch:
            ff = item.get("first_frame")
            if ff is not None:
                first_frames.append(ff)
            else:
                first_frames.append(torch.zeros_like(item["video"][:, :1, :, :]))
        first_frames = torch.stack(first_frames, dim=0)

    return {
        "video": videos,
        "mel": mels,
        "caption": captions,
        "fps": fps,
        "first_frame": first_frames,
    }
