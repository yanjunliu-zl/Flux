"""Lip-Sync Losses: phoneme-viseme alignment + mouth-audio sync.

Provides three complementary loss functions for fine-grained lip-sync training:

1. VisemeClassificationLoss
   - Cross-entropy on viseme class prediction from audio features
   - Supervised by pre-computed phoneme→viseme labels per frame
   - Trains the model to map audio phonemes to visual mouth shapes

2. MouthAudioContrastiveLoss
   - Fine-grained contrastive loss restricted to mouth-region features
   - Pushes mouth-region video features closer to corresponding audio features
   - Unlike global AV sync loss, this focuses exclusively on the mouth area

3. LipTemporalSmoothnessLoss
   - Penalizes abrupt changes in mouth-region latent features
   - Encourages natural, smooth lip transitions between frames

Usage:
    from seedance.loss.lip_sync_loss import (
        VisemeClassificationLoss,
        MouthAudioContrastiveLoss,
        compute_lip_sync_losses,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisemeClassificationLoss(nn.Module):
    """Cross-entropy loss for viseme (mouth shape) prediction from audio.

    Maps audio features to one of 14 viseme classes using a learned
    audio-to-viseme classifier. Ground truth viseme labels come from
    preprocessing: audio → Wav2Vec2/HuBERT → phoneme → viseme mapping.

    Args:
        num_visemes: Number of viseme classes (default 14, MPEG-4 set).
        label_smoothing: Label smoothing factor.
        class_weights: Optional per-class weights for imbalanced data.
    """

    def __init__(
        self,
        num_visemes: int = 14,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_visemes = num_visemes
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        viseme_logits: torch.Tensor,
        viseme_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute viseme classification loss.

        Args:
            viseme_logits: Predicted logits (B, T, num_visemes) or (B, N_a, num_visemes).
            viseme_labels: Ground truth viseme class indices (B, T), values in [0, num_visemes-1].
                -1 = ignore index (e.g., non-speech frames).
            mask: Optional valid frame mask (B, T), True = valid.

        Returns:
            Scalar loss.
        """
        B = viseme_logits.shape[0]

        # Flatten to (B*T, num_visemes) and (B*T,)
        logits_flat = viseme_logits.reshape(-1, self.num_visemes)
        labels_flat = viseme_labels.reshape(-1)

        # Ignore invalid frames (label = -1)
        valid = labels_flat >= 0
        if mask is not None:
            valid = valid & mask.reshape(-1)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=viseme_logits.device)

        logits_valid = logits_flat[valid]
        labels_valid = labels_flat[valid]

        # Cross-entropy with optional label smoothing and class weights
        loss = F.cross_entropy(
            logits_valid,
            labels_valid,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        return loss

    def compute_accuracy(
        self,
        viseme_logits: torch.Tensor,
        viseme_labels: torch.Tensor,
    ) -> float:
        """Compute viseme classification accuracy.

        Args:
            viseme_logits: (B, T, num_visemes).
            viseme_labels: (B, T).

        Returns:
            Accuracy as fraction [0, 1].
        """
        valid = viseme_labels >= 0
        if valid.sum() == 0:
            return 0.0

        preds = viseme_logits.argmax(dim=-1)
        correct = (preds == viseme_labels) & valid
        return correct.sum().item() / max(valid.sum().item(), 1)


class MouthAudioContrastiveLoss(nn.Module):
    """Fine-grained contrastive loss restricted to mouth region.

    Unlike the global AV sync loss (which pools over the entire frame),
    this loss focuses on mouth-region features to learn precise
    correlations between lip movements and audio.

    Uses a localized InfoNCE loss where:
      - Positive pairs: mouth features + aligned audio features
      - Negative pairs: mouth features from frame i + audio from frame j (i ≠ j)

    Args:
        temperature: Softmax temperature (default 0.07).
        mouth_mask_threshold: Spatial mask threshold for selecting "mouth" tokens.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mouth_mask_threshold: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.mouth_mask_threshold = mouth_mask_threshold

    def _extract_mouth_features(
        self,
        v_tokens: torch.Tensor,
        mouth_mask: torch.Tensor,
        T_lat: int,
        H_lat: int,
        W_lat: int,
    ) -> torch.Tensor:
        """Extract mouth-region features using spatial mask.

        Args:
            v_tokens: Video tokens (B, N_v, D) where N_v = T_lat * H_lat * W_lat.
            mouth_mask: Spatial mask (B, H_lat, W_lat) or (1, H_lat, W_lat).
            T_lat, H_lat, W_lat: Latent grid dimensions.

        Returns:
            Mouth-region pooled features (B, T_lat, D).
        """
        B, _, D = v_tokens.shape

        # Expand mask to batch and temporal dimensions
        if mouth_mask.dim() == 3:
            if mouth_mask.shape[0] == 1 and B > 1:
                mouth_mask = mouth_mask.expand(B, -1, -1)
        mask_spatial = mouth_mask.reshape(B, 1, H_lat * W_lat)  # (B, 1, H*W)
        mask_spatiotemporal = mask_spatial.unsqueeze(1).expand(-1, T_lat, -1, -1)
        mask_spatiotemporal = mask_spatiotemporal.reshape(B, T_lat, H_lat * W_lat)

        # Reshape video tokens: (B, T_lat*H_lat*W_lat, D) → (B, T_lat, H_lat*W_lat, D)
        v_reshaped = v_tokens.reshape(B, T_lat, H_lat * W_lat, D)

        # Weighted pooling over spatial dims
        weights = F.softmax(mask_spatiotemporal * 10, dim=-1).unsqueeze(-1)  # (B, T_lat, H*W, 1)
        mouth_features = (v_reshaped * weights).sum(dim=2)  # (B, T_lat, D)

        return mouth_features

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        mouth_mask: torch.Tensor,
        T_lat: int,
        H_lat: int,
        W_lat: int,
    ) -> torch.Tensor:
        """Compute mouth-audio contrastive loss.

        Args:
            v_tokens: Video tokens (B, N_v, D).
            a_tokens: Audio tokens (B, N_a, D).
            mouth_mask: Spatial mask (B, H_lat, W_lat).
            T_lat, H_lat, W_lat: Latent grid dims.

        Returns:
            Contrastive loss scalar.
        """
        # Extract mouth features per temporal frame
        mouth_feat = self._extract_mouth_features(v_tokens, mouth_mask, T_lat, H_lat, W_lat)
        # (B, T_lat, D) → (B*T_lat, D)
        B, T, D = mouth_feat.shape
        mouth_feat = mouth_feat.reshape(B * T, D)

        # Pool audio tokens per temporal segment
        # (B, N_a, D) → (B, T_lat, D) via adaptive temporal pooling
        a_feat = a_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
        a_feat = a_feat.expand(-1, T, -1).reshape(B * T, D)

        # Normalize
        mouth_feat = F.normalize(mouth_feat, dim=-1)
        a_feat = F.normalize(a_feat, dim=-1)

        # Cosine similarity
        sim = (mouth_feat @ a_feat.T) / self.temperature  # (B*T, B*T)

        # InfoNCE
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_mouth = F.cross_entropy(sim, labels)
        loss_audio = F.cross_entropy(sim.T, labels)

        return (loss_mouth + loss_audio) / 2


class LipTemporalSmoothnessLoss(nn.Module):
    """Temporal smoothness loss for natural lip motion.

    Penalizes large frame-to-frame changes in mouth-region latent features.
    This prevents jittery/unrealistic lip movements.

    Args:
        order: Order of smoothness (1 = first derivative, 2 = second derivative).
        reduction: "mean" or "sum".
    """

    def __init__(self, order: int = 1, reduction: str = "mean"):
        super().__init__()
        self.order = order
        self.reduction = reduction

    def forward(
        self,
        mouth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            mouth_features: Mouth-region features (B, T, D) or (B, T_lat, D).

        Returns:
            Smoothness loss scalar.
        """
        # First-order difference
        diff = mouth_features[:, 1:, :] - mouth_features[:, :-1, :]  # (B, T-1, D)

        if self.order == 1:
            loss = diff ** 2
        elif self.order == 2:
            # Second-order: penalize acceleration
            diff2 = diff[:, 1:, :] - diff[:, :-1, :]  # (B, T-2, D)
            loss = diff2 ** 2
        else:
            raise ValueError(f"order must be 1 or 2, got {self.order}")

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# Convenience function for computing all lip-sync losses
# ---------------------------------------------------------------------------

def compute_lip_sync_losses(
    viseme_logits: Optional[torch.Tensor],
    viseme_labels: Optional[torch.Tensor],
    v_tokens: Optional[torch.Tensor],
    a_tokens: Optional[torch.Tensor],
    mouth_mask: Optional[torch.Tensor],
    T_lat: int,
    H_lat: int,
    W_lat: int,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, torch.Tensor]:
    """Compute all lip-sync related losses in one call.

    Args:
        viseme_logits: (B, N_a, num_visemes) from LipSyncBridge.
        viseme_labels: (B, T) viseme class indices.
        v_tokens: Video tokens (B, N_v, D).
        a_tokens: Audio tokens (B, N_a, D).
        mouth_mask: (B, H_lat, W_lat).
        T_lat, H_lat, W_lat: Latent grid dims.
        weights: Optional loss weight dict with keys:
            "viseme", "mouth_contrastive", "lip_smooth".

    Returns:
        Dict of loss_name → tensor.
    """
    if weights is None:
        weights = {
            "viseme": 0.5,
            "mouth_contrastive": 0.3,
            "lip_smooth": 0.1,
        }

    losses = {}
    device = v_tokens.device if v_tokens is not None else a_tokens.device

    # 1. Viseme classification loss
    if viseme_logits is not None and viseme_labels is not None:
        viseme_criterion = VisemeClassificationLoss()
        losses["viseme_loss"] = viseme_criterion(viseme_logits, viseme_labels) * weights["viseme"]
    else:
        losses["viseme_loss"] = torch.tensor(0.0, device=device)

    # 2. Mouth-audio contrastive loss
    if v_tokens is not None and a_tokens is not None and mouth_mask is not None:
        mouth_criterion = MouthAudioContrastiveLoss()
        losses["mouth_contrastive_loss"] = mouth_criterion(
            v_tokens, a_tokens, mouth_mask, T_lat, H_lat, W_lat
        ) * weights["mouth_contrastive"]
    else:
        losses["mouth_contrastive_loss"] = torch.tensor(0.0, device=device)

    # 3. Lip temporal smoothness loss (on mouth features)
    if v_tokens is not None and mouth_mask is not None:
        smooth_criterion = LipTemporalSmoothnessLoss(order=1)
        # Extract mouth features for smoothness
        B, _, D = v_tokens.shape
        v_reshaped = v_tokens.reshape(B, T_lat, H_lat * W_lat, D)
        if mouth_mask.dim() == 3:
            mask = mouth_mask.reshape(-1, 1, H_lat * W_lat).unsqueeze(-1)
            mouth_feat = (v_reshaped * mask.softmax(dim=2)).sum(dim=2)
            losses["lip_smooth_loss"] = smooth_criterion(mouth_feat) * weights["lip_smooth"]
        else:
            losses["lip_smooth_loss"] = torch.tensor(0.0, device=device)
    else:
        losses["lip_smooth_loss"] = torch.tensor(0.0, device=device)

    # Total
    losses["lip_sync_total"] = sum(losses.values())

    return losses
