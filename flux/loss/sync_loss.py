"""Audio-Visual synchronization loss.

Contrastive loss that encourages aligned audio-video pairs
to have higher feature similarity than misaligned pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVSyncLoss(nn.Module):
    """Contrastive audio-visual synchronization loss.

    Uses InfoNCE-style loss on mean-pooled features from both modalities.
    Can be applied at intermediate or final layers.

    Args:
        temperature: Softmax temperature (default: 0.07).
        use_intermediate: If True, compute on intermediate layer outputs.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        v_features: torch.Tensor,
        a_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive AV sync loss.

        Args:
            v_features: Video features (B, D_v) — mean-pooled over spatial/temporal.
            a_features: Audio features (B, D_a) — mean-pooled over freq/time.

        Returns:
            Contrastive loss scalar.
        """
        # Normalize features
        v_feat = F.normalize(v_features, dim=-1)
        a_feat = F.normalize(a_features, dim=-1)

        # Cosine similarity matrix: (B, B)
        sim = v_feat @ a_feat.T / self.temperature

        # Positive pairs on diagonal
        labels = torch.arange(sim.shape[0], device=sim.device)

        # Symmetric InfoNCE
        loss_v = F.cross_entropy(sim, labels)
        loss_a = F.cross_entropy(sim.T, labels)

        return (loss_v + loss_a) / 2
