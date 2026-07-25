"""Video I/O utilities."""

import torch
import numpy as np


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 16,
    normalize: bool = True,
):
    """Save video frames to MP4 file.

    Args:
        frames: Video tensor (T, C, H, W) or (C, T, H, W), values in [-1, 1] or [0, 1].
        output_path: Output file path.
        fps: Frames per second.
        normalize: If True, scale from [-1, 1] to [0, 255].
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required. Install with: pip install imageio[ffmpeg]")

    # Normalize format
    if frames.dim() == 4:
        if frames.shape[1] == 3:  # (T, C, H, W)
            frames = frames
        elif frames.shape[0] == 3:  # (C, T, H, W)
            frames = frames.permute(1, 0, 2, 3)
        else:
            frames = frames.unsqueeze(1)  # (T, H, W)

    frames = frames.detach().cpu()

    # Convert to uint8
    if normalize:
        frames = (frames + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    frames = (frames.clamp(0, 1) * 255).to(torch.uint8)

    # (T, C, H, W) -> list of (H, W, C) numpy arrays
    frames_np = [f.permute(1, 2, 0).numpy() for f in frames]

    writer = imageio.get_writer(output_path, fps=fps, format="FFMPEG", codec="libx264")
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def load_video_frames(
    video_path: str,
    num_frames: int = 32,
    resolution: int = 256,
) -> torch.Tensor:
    """Load and preprocess video frames.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to extract.
        resolution: Target spatial resolution.

    Returns:
        Video tensor (T, C, H, W), values in [0, 1].
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resolution, resolution))
        frames.append(frame)

    cap.release()

    video = torch.from_numpy(np.stack(frames)).float() / 255.0
    return video  # (T, H, W, C) -> need permute for (T, C, H, W)
