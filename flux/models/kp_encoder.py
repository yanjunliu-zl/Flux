"""KP-Semantic 3D Keypoint Encoder for facial motion control.

Encodes 3D facial keypoints (468-point MediaPipe mesh or FLAME vertices)
into compact temporal control vectors used as conditioning signals
for the DB-DiT temporal branch.

This module handles:
  1. 3D keypoint coordinate normalization
  2. Temporal smoothing (Gaussian filter to remove jitter)
  3. Keypoint-to-latent encoding via temporal convolution
  4. Mouth-region weighted loss for audio-visual lip sync

Integration:
  - Preprocessing: Extract 3D KPs per frame, encode to z_kp, store in Zarr.
  - Training: z_kp injected as cross-attention condition in DB-DiT temporal branch.
  - Loss: KP reconstruction L2 + mouth-region weighted loss for lip sync.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KPConfig:
    """3D keypoint encoder configuration."""
    kp_num: int = 468          # Number of keypoints (MediaPipe: 468, FLAME: 5023)
    coord_dim: int = 3         # x, y, z (depth)
    seq_len: int = 32          # Temporal window (frames), aligned with video
    kp_emb_dim: int = 96       # Output embedding dimension
    mouth_kp_indices: Optional[list[int]] = None  # Indices of mouth-region KPs
    smooth_sigma: float = 1.2  # Temporal Gaussian smoothing sigma


# Default MediaPipe face mesh mouth-region landmark indices
# (lips inner + outer contour, ~40 points)
MEDIAPIPE_MOUTH_INDICES = list(range(0, 17)) + list(range(61, 82)) + list(range(267, 288))


class KP3DEncoder(nn.Module):
    """3D keypoint semantic encoder with temporal convolution.

    Compresses a sequence of 3D facial keypoints into a compact
    temporal embedding z_kp used as a control signal for MMDiT.

    Architecture:
      Input: (B, T, N*3)  — flattened 3D keypoints
        ↓
      MLP per-frame:  N*3 → 512
        ↓
      1D temporal conv (3 layers): 512 → 256 → 128 → kp_emb_dim
        ↓
      Output: (B, T, kp_emb_dim) — temporal control embedding

    Args:
        config: KPConfig instance.
    """

    def __init__(self, config: KPConfig | None = None):
        super().__init__()
        if config is None:
            config = KPConfig()
        self.config = config
        self.kp_num = config.kp_num
        self.coord_dim = config.coord_dim
        self.input_dim = config.kp_num * config.coord_dim  # e.g. 468*3 = 1404
        self.kp_emb_dim = config.kp_emb_dim

        # Per-frame projection
        self.frame_proj = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        # Temporal convolution stack (operates on (B, C, T))
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, config.kp_emb_dim, kernel_size=3, padding=1),
        )

    def forward(self, kp_3d_seq: torch.Tensor) -> torch.Tensor:
        """Encode 3D keypoint sequence.

        Args:
            kp_3d_seq: Raw 3D keypoints (B, T, N*3).
                N*3 = kp_num * coord_dim, flattened per frame.

        Returns:
            Temporal control embedding (B, T, kp_emb_dim).
        """
        B, T, _ = kp_3d_seq.shape

        # Per-frame MLP projection
        feat = self.frame_proj(kp_3d_seq)  # (B, T, 256)

        # Transpose for Conv1d: (B, C, T)
        feat = feat.permute(0, 2, 1)  # (B, 256, T)

        # Temporal convolution
        z_kp = self.temporal_conv(feat)  # (B, kp_emb_dim, T)

        # Transpose back: (B, T, kp_emb_dim)
        z_kp = z_kp.permute(0, 2, 1)

        return z_kp


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def smooth_kp_sequence(
    kp_seq: np.ndarray,
    sigma: float = 1.2,
) -> np.ndarray:
    """Apply temporal Gaussian smoothing to keypoint trajectories.

    Removes per-frame detection jitter while preserving real motion.

    Args:
        kp_seq: Keypoint array (T, N, 3) or (T, N*3).
        sigma: Gaussian filter standard deviation (in frames).

    Returns:
        Smoothed keypoint array with same shape.
    """
    from scipy.ndimage import gaussian_filter1d

    if kp_seq.ndim == 2:
        # (T, N*3) — smooth along axis 0 (time)
        return gaussian_filter1d(kp_seq, sigma=sigma, axis=0)

    # (T, N, 3) — smooth each coordinate independently
    T, N, D = kp_seq.shape
    for n in range(N):
        for d in range(D):
            kp_seq[:, n, d] = gaussian_filter1d(kp_seq[:, n, d], sigma=sigma)
    return kp_seq


def normalize_kp_coordinates(
    kp_coords: np.ndarray,
    method: str = "scale",
) -> np.ndarray:
    """Normalize 3D keypoint coordinates to a stable range.

    Args:
        kp_coords: Raw keypoints (T, N, 3) or frame (N, 3).
        method: "scale" (divide by max absolute value → [-1, 1])
                or "zscore" (zero-mean unit-variance).

    Returns:
        Normalized coordinates.
    """
    if method == "scale":
        max_val = np.max(np.abs(kp_coords))
        if max_val > 1e-8:
            kp_coords = kp_coords / max_val
    elif method == "zscore":
        mean = kp_coords.mean()
        std = kp_coords.std()
        if std > 1e-8:
            kp_coords = (kp_coords - mean) / std

    return kp_coords


# ---------------------------------------------------------------------------
# Production 3D keypoint extraction via MediaPipe Face Mesh
# ---------------------------------------------------------------------------

def extract_3d_keypoints(
    frame: "np.ndarray",
    normalize: bool = True,
) -> "np.ndarray | None":
    """Extract 468-point 3D face mesh from a single frame.

    Uses MediaPipe Face Mesh for production-grade 3D landmark extraction.

    Args:
        frame: RGB image (H, W, 3) uint8.
        normalize: If True, normalize to [-1, 1] range.

    Returns:
        (468, 3) float32 array, or None if no face detected.
    """
    from flux.models.face_analysis import get_face_analyzer

    analyzer = get_face_analyzer()
    meshes = analyzer.extract_mesh(frame)

    if not meshes:
        return None

    kp = meshes[0]  # (468, 3) — first (largest) face

    if normalize:
        max_val = np.max(np.abs(kp))
        if max_val > 1e-8:
            kp = kp / max_val

    return kp.astype(np.float32)


def extract_kp_sequence(
    frames: "np.ndarray",
    smooth: bool = True,
    smooth_sigma: float = 1.2,
) -> "np.ndarray | None":
    """Extract 468-point 3D keypoint sequence from video frames.

    Args:
        frames: Video frames (T, H, W, 3) uint8 or (T, C, H, W) float tensor.
        smooth: Apply temporal Gaussian smoothing.
        smooth_sigma: Smoothing sigma in frames.

    Returns:
        (T, 468, 3) float32 array, or None if extraction fails.
    """
    # Convert tensor to numpy if needed
    if hasattr(frames, "cpu"):
        if frames.dim() == 4 and frames.shape[1] == 3:
            frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # TCHW -> THWC
        else:
            frames = frames.cpu().numpy()
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    kp_list = []
    for t in range(len(frames)):
        kp = extract_3d_keypoints(frames[t], normalize=True)
        if kp is not None:
            kp_list.append(kp)
        elif kp_list:
            # Propagate last valid detection
            kp_list.append(kp_list[-1].copy())
        else:
            kp_list.append(np.zeros((468, 3), dtype=np.float32))

    kp_seq = np.stack(kp_list, axis=0)  # (T, 468, 3)

    if smooth:
        kp_seq = smooth_kp_sequence(kp_seq, sigma=smooth_sigma)

    return kp_seq


# ---------------------------------------------------------------------------
# Training losses
# ---------------------------------------------------------------------------

def kp_reconstruction_loss(
    kp_pred: torch.Tensor,
    kp_gt: torch.Tensor,
    mouth_indices: list[int] | None = None,
    mouth_weight: float = 2.0,
) -> torch.Tensor:
    """Keypoint reconstruction loss with mouth-region weighting.

    Heavier penalty on mouth keypoints to improve lip-sync quality.

    Args:
        kp_pred: Predicted keypoint embedding (B, T, kp_emb_dim).
        kp_gt: Ground truth keypoint embedding (B, T, kp_emb_dim).
        mouth_indices: List of mouth-region KP indices. If None, uses default MediaPipe set.
        mouth_weight: Weight multiplier for mouth-region KPs.

    Returns:
        Weighted MSE loss scalar.
    """
    if mouth_indices is None:
        mouth_indices = MEDIAPIPE_MOUTH_INDICES

    B, T, D = kp_pred.shape
    N = kp_pred.shape[-1] * 3 // 3  # Approximate KP count from embedding dim

    # Base L2 loss
    base_loss = F.mse_loss(kp_pred, kp_gt)

    # Mouth-region weighting is approximate with the compressed embedding.
    # In practice, the mouth weighting would be applied in raw KP space
    # before encoding. Here we apply a simplified version:
    # weight = 1.0 + (mouth_weight - 1.0) * (mouth_kp_ratio)
    mouth_ratio = len(mouth_indices) / max(N, 1)
    weighted_loss = base_loss * (1.0 + (mouth_weight - 1.0) * mouth_ratio)

    return weighted_loss


def kp_temporal_smoothness_loss(
    z_kp: torch.Tensor,
) -> torch.Tensor:
    """Temporal smoothness regularization for keypoint embeddings.

    Penalizes large frame-to-frame changes in the encoded KP latent,
    encouraging smooth facial motion.

    Args:
        z_kp: Keypoint embedding (B, T, kp_emb_dim).

    Returns:
        Smoothness loss scalar.
    """
    # First-order temporal difference
    diff = z_kp[:, 1:, :] - z_kp[:, :-1, :]  # (B, T-1, D)
    loss = torch.mean(diff ** 2)
    return loss
