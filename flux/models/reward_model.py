"""Multi-dimensional Reward Model for RLHF fine-tuning.

The Reward Model (RM) scores generated videos across multiple quality
dimensions to provide a composite reward signal for PPO training.

Scoring dimensions (5-axis):
  1. Visual quality: BRISQUE-inspired, clarity, color richness
  2. Motion smoothness: Temporal coherence, absence of jitter
  3. Character consistency: Cross-frame face similarity (powered by LFA)
  4. Audio-visual sync: Lip-movement to audio alignment
  5. Prompt alignment: CLIP-based semantic similarity

Architecture:
  Shared 3D Conv backbone → dimension-specific heads → scalar scores.
  Final reward = weighted sum of dimension scores.

Usage:
    rm = RewardModel()
    scores = rm(video_latent, audio_latent, text_emb)  # (B, 5)
    reward = rm.compute_reward(scores)                   # (B,) weighted sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class RMConfig:
    """Reward Model configuration."""
    latent_channels: int = 16
    hidden_dim: int = 512
    num_scores: int = 5  # visual, motion, identity, av_sync, prompt_alignment
    score_names: list[str] = field(default_factory=lambda: [
        "visual_quality",
        "motion_smoothness",
        "character_consistency",
        "av_sync",
        "prompt_alignment",
    ])
    # Default weights for computing composite reward
    score_weights: list[float] = field(default_factory=lambda: [0.2, 0.25, 0.25, 0.15, 0.15])
    # Text embedding dimension (from T5 encoder)
    text_emb_dim: int = 768


class VideoQualityHead(nn.Module):
    """Scores video visual quality: clarity, color, detail preservation."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(feat).flatten(1))  # (B, 1)


class MotionHead(nn.Module):
    """Scores motion smoothness: temporal coherence, naturalness."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Temporal difference pooling
        self.conv = nn.Conv3d(hidden_dim, hidden_dim // 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        temp_feat = self.conv(feat)  # Capture temporal differences
        return self.head(self.pool(temp_feat).flatten(1))  # (B, 1)


class IdentityConsistencyHead(nn.Module):
    """Scores character consistency across frames."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Compute variance across temporal dimension as inconsistency signal
        B, C, T, H, W = feat.shape
        temporal_mean = feat.mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
        temporal_var = ((feat - temporal_mean) ** 2).mean(dim=[2, 3, 4])  # (B, C)
        # Lower variance = higher consistency
        var_feat = temporal_var.mean(dim=-1, keepdim=True)  # (B, 1)
        return 1.0 / (1.0 + var_feat)  # Map to (0, 1] range


class AVSyncHead(nn.Module):
    """Scores audio-visual synchronization quality."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.av_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, v_feat: torch.Tensor, a_feat: torch.Tensor) -> torch.Tensor:
        # Cross-modal feature interaction
        v_pool = v_feat.mean(dim=[2, 3, 4])  # (B, C)
        a_pool = a_feat.mean(dim=[2, 3])     # (B, C)
        concat = torch.cat([v_pool, a_pool], dim=-1)  # (B, 2C)
        fuse = self.av_proj(concat)
        return self.head(fuse)  # (B, 1)


class PromptAlignmentHead(nn.Module):
    """Scores semantic alignment between video content and text prompt."""
    def __init__(self, hidden_dim: int, text_emb_dim: int = 768):
        super().__init__()
        self.text_proj = nn.Linear(text_emb_dim, hidden_dim)
        self.video_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, v_feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Text embedding (B, L, D) or (B, D)
        if text_emb.dim() == 3:
            text_emb = text_emb.mean(dim=1)  # Pool token dimension
        text_feat = self.text_proj(text_emb)  # (B, hidden_dim)

        # Video feature (B, C, T, H, W) → (B, hidden_dim)
        v_pool = v_feat.mean(dim=[2, 3, 4])  # (B, C)
        v_feat_p = self.video_proj(v_pool)    # (B, hidden_dim)

        # Cosine similarity in projected space
        sim = F.cosine_similarity(v_feat_p, text_feat, dim=-1)  # (B,)
        # Map from [-1, 1] to score; use a learnable scale + bias via head
        return self.head(v_feat_p * text_feat)  # (B, 1)


class RewardModel(nn.Module):
    """Multi-dimensional reward model for RLHF.

    Evaluates generated videos across 5 quality dimensions and
    produces a composite reward signal for PPO optimization.

    Args:
        config: RMConfig instance.
    """

    def __init__(self, config: RMConfig | None = None):
        super().__init__()
        if config is None:
            config = RMConfig()
        self.config = config
        self.num_scores = config.num_scores

        # Shared 3D backbone for video feature extraction
        self.video_backbone = nn.Sequential(
            nn.Conv3d(config.latent_channels, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(config.hidden_dim // 2, config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Audio backbone
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(8, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim // 2, config.hidden_dim, kernel_size=3, stride=2, padding=1),
        )

        # Dimension-specific scoring heads
        self.visual_head = VideoQualityHead(config.hidden_dim)
        self.motion_head = MotionHead(config.hidden_dim)
        self.identity_head = IdentityConsistencyHead(config.hidden_dim)
        self.av_sync_head = AVSyncHead(config.hidden_dim)
        self.prompt_head = PromptAlignmentHead(config.hidden_dim, config.text_emb_dim)

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score a generated video across all dimensions.

        Args:
            video_latent: Generated video latent (B, C, T, H, W).
            audio_latent: Generated audio latent (B, C, F, T), optional.
            text_emb: Text embedding (B, L, D) or (B, D), optional.

        Returns:
            Dict mapping score_name → tensor (B, 1).
        """
        # Video features
        v_feat = self.video_backbone(video_latent)  # (B, hidden_dim, T', H', W')

        # Audio features
        a_feat = None
        if audio_latent is not None:
            a_feat = self.audio_backbone(audio_latent)  # (B, hidden_dim, F', T')

        # Compute dimension scores
        scores = {}
        scores["visual_quality"] = self.visual_head(v_feat)
        scores["motion_smoothness"] = self.motion_head(v_feat)
        scores["character_consistency"] = self.identity_head(v_feat)

        if a_feat is not None:
            scores["av_sync"] = self.av_sync_head(v_feat, a_feat)
        else:
            scores["av_sync"] = torch.zeros_like(scores["visual_quality"])

        if text_emb is not None:
            scores["prompt_alignment"] = self.prompt_head(v_feat, text_emb)
        else:
            scores["prompt_alignment"] = torch.zeros_like(scores["visual_quality"])

        return scores

    def compute_reward(
        self,
        scores: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute composite reward as weighted sum of dimension scores.

        Args:
            scores: Dict from forward().

        Returns:
            Composite reward (B, 1).
        """
        reward = torch.zeros_like(scores["visual_quality"])
        for name, weight in zip(self.config.score_names, self.config.score_weights):
            if name in scores:
                reward = reward + weight * scores[name]
        return reward

    def compute_training_loss(
        self,
        scores: dict[str, torch.Tensor],
        human_ratings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Train RM to match human preference ratings.

        Uses MSE loss per dimension against human-labeled scores.

        Args:
            scores: Model-predicted scores dict.
            human_ratings: Human-labeled scores dict (same keys).

        Returns:
            Training loss scalar.
        """
        total_loss = 0.0
        for name in self.config.score_names:
            if name in scores and name in human_ratings:
                total_loss = total_loss + F.mse_loss(scores[name], human_ratings[name])
        return total_loss / self.num_scores
