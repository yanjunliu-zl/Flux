"""LFA: Lightweight Feature Anchor Encoder for character consistency.

LFA solves the character identity drift problem in video generation by
extracting a compact, temporally-invariant identity anchor vector that
remains fixed across all frames of a video clip.

Architecture (dual-branch):
  - Identity branch: Frozen ArcFace/ResNet backbone → L2-normalized 128-dim anchor.
    Captures invariant facial features (bone structure, skin tone, hair).
  - Motion branch: Lightweight trainable CNN → 128-dim expression/pose features.
    Captures variable attributes (expression, head pose, gaze).
  - Fusion gate: Learnable weighted combination → final 128-dim condition vector.

Usage:
    # Extract anchor from reference image
    encoder = LFAEncoder()
    z_id = encoder.extract_anchor(reference_face)  # (1, 128) global anchor
    z_motion = encoder.extract_motion(face_frame)  # (1, 128) per-frame motion

    # Training loss: force all generated frames close to anchor
    loss = lfa_consistency_loss(z_id_anchor, frame_z_fuse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LFAConfig:
    """LFA encoder configuration."""
    emb_dim: int = 128
    face_size: int = 224
    id_backbone: str = "resnet18"   # "resnet18" | "resnet50" | "arcface_ir50"
    freeze_id_branch: bool = True
    motion_channels: int = 64
    pretrained_id: bool = True


class IDBackbone(nn.Module):
    """Identity feature extractor using lightweight ResNet backbone.

    Extracts facial identity features that are invariant to pose/expression.
    Frozen during training to preserve identity discrimination.
    """

    def __init__(self, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = resnet18(weights=None)
        except (ImportError, AttributeError):
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-2])  # (B, 512, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract identity features.

        Args:
            x: Face image (B, 3, 224, 224), normalized to [0, 1].

        Returns:
            L2-normalized identity embedding (B, emb_dim).
        """
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)  # (B, 512)
        z_id = self.proj(feat)
        return F.normalize(z_id, p=2, dim=-1)  # Unit sphere for cosine similarity


class MotionEncoder(nn.Module):
    """Lightweight motion/expression encoder.

    Captures per-frame variable attributes (expression, head pose, gaze)
    using a compact CNN. Trainable to adapt to the target distribution.
    """

    def __init__(self, emb_dim: int = 128, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels * 2, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract motion/expression features.

        Args:
            x: Face image (B, 3, 224, 224).

        Returns:
            Motion embedding (B, emb_dim).
        """
        feat = self.conv(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)


class LFAEncoder(nn.Module):
    """Lightweight Feature Anchor encoder for character consistency.

    Dual-branch architecture:
      - Frozen identity branch: Extracts invariant face anchor z_id.
      - Trainable motion branch: Extracts per-frame pose/expression features.
      - Gate: Learns to fuse both branches for conditioning MMDiT.

    The identity anchor z_id is computed once per video/clip and reused
    across all frames, ensuring the generated character stays consistent.

    Args:
        config: LFAConfig or dict with LFA hyperparameters.
    """

    def __init__(self, config: LFAConfig | None = None):
        super().__init__()
        if config is None:
            config = LFAConfig()
        self.config = config
        self.emb_dim = config.emb_dim

        # Identity branch (frozen)
        self.id_backbone = IDBackbone(
            emb_dim=config.emb_dim,
            pretrained=config.pretrained_id,
        )
        if config.freeze_id_branch:
            for param in self.id_backbone.parameters():
                param.requires_grad = False

        # Motion/expression branch (trainable)
        self.motion_encoder = MotionEncoder(
            emb_dim=config.emb_dim,
            channels=config.motion_channels,
        )

        # Fusion gate: learn how to combine identity and motion
        self.gate = nn.Sequential(
            nn.Linear(config.emb_dim * 2, config.emb_dim),
            nn.Sigmoid(),  # Output in [0, 1] as soft gate
        )

    def extract_anchor(self, face_img: torch.Tensor) -> torch.Tensor:
        """Extract global identity anchor from reference face.

        This should be called ONCE per character and shared across
        all frames of a video or all shots of a short drama.

        Args:
            face_img: Reference face image (1, 3, H, W) or (B, 3, H, W).

        Returns:
            L2-normalized identity anchor (1, emb_dim) or (B, emb_dim).
        """
        return self.id_backbone(face_img)

    def extract_motion(self, face_img: torch.Tensor) -> torch.Tensor:
        """Extract per-frame motion/expression features.

        Args:
            face_img: Face image (B, 3, H, W).

        Returns:
            Motion embedding (B, emb_dim).
        """
        return self.motion_encoder(face_img)

    def forward(self, face_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            face_img: Face image (B, 3, H, W).

        Returns:
            Tuple of:
              - z_id: L2-normalized identity anchor (B, emb_dim)
              - z_fuse: Gated fusion of identity + motion (B, emb_dim)
        """
        z_id = self.id_backbone(face_img)           # (B, emb_dim), L2-normalized
        z_motion = self.motion_encoder(face_img)      # (B, emb_dim)

        # Gated fusion
        concat = torch.cat([z_id, z_motion], dim=-1)  # (B, 2*emb_dim)
        gate = self.gate(concat)                       # (B, emb_dim), in [0,1]
        z_fuse = gate * z_id + (1 - gate) * z_motion   # Soft combination

        return z_id, z_fuse


def lfa_consistency_loss(
    z_id_anchor: torch.Tensor,
    frame_z_fuse: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Compute character consistency loss.

    Penalizes deviation of per-frame generated features from the
    global identity anchor. This is added to the Flow Matching loss
    during SFT/RLHF training phases.

    Args:
        z_id_anchor: Global identity anchor (B, emb_dim).
            Computed once from reference face, shared across all frames.
        frame_z_fuse: Per-frame fused features (B, T, emb_dim).
            From LFA encoder, one per generated frame.
        loss_type: "cosine" for cosine similarity loss,
                   "l2" for MSE between features.

    Returns:
        Scalar consistency loss.
    """
    B, T, D = frame_z_fuse.shape
    # Expand anchor to match temporal dimension
    anchor_exp = z_id_anchor.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

    if loss_type == "cosine":
        # Maximize cosine similarity between each frame and anchor
        frame_norm = F.normalize(frame_z_fuse, p=2, dim=-1)
        anchor_norm = F.normalize(anchor_exp, p=2, dim=-1)
        cosine_sim = (frame_norm * anchor_norm).sum(dim=-1)  # (B, T)
        loss = 1.0 - cosine_sim.mean()
    elif loss_type == "l2":
        loss = F.mse_loss(frame_z_fuse, anchor_exp)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


# ---------------------------------------------------------------------------
# Production face detection via SCRFD (InsightFace)
# ---------------------------------------------------------------------------

def detect_and_crop_face(
    frame: torch.Tensor,
    face_size: int = 224,
    expand_ratio: float = 0.2,
) -> torch.Tensor | None:
    """Detect and crop the largest face using SCRFD.

    Production-grade detection via InsightFace SCRFD-10G.

    Args:
        frame: RGB frame tensor (C, H, W) or (H, W, C) [0, 1] or [0, 255].
        face_size: Output face crop size (square).
        expand_ratio: Ratio to expand the face bounding box.

    Returns:
        Cropped face tensor (1, 3, face_size, face_size), or None if no face found.
    """
    import numpy as np

    # Convert to numpy HWC uint8
    if frame.dim() == 3 and frame.shape[0] == 3:
        frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    else:
        frame_np = frame.cpu().numpy()

    if frame_np.max() <= 1.0:
        frame_np = (frame_np * 255).astype(np.uint8)
    else:
        frame_np = frame_np.astype(np.uint8)

    H, W = frame_np.shape[:2]

    # Use InsightFace SCRFD
    try:
        from seedance.models.face_analysis import get_face_analyzer
        analyzer = get_face_analyzer()
        face = analyzer.detect_largest(frame_np)
    except Exception:
        return None

    if face is None:
        return None

    x1, y1, x2, y2 = face.bbox
    fw, fh = x2 - x1, y2 - y1

    # Expand
    pad_w = int(fw * expand_ratio)
    pad_h = int(fh * expand_ratio)
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(W, int(x2 + pad_w))
    y2 = min(H, int(y2 + pad_h))

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = frame_np[y1:y2, x1:x2]

    # Resize to target
    import cv2
    face_resized = cv2.resize(face_crop, (face_size, face_size))

    face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0)  # (1, 3, face_size, face_size)
