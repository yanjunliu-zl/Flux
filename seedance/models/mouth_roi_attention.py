"""Mouth ROI Attention for fine-grained lip-sync.

Extends the CBGA cross-modal bridge with spatially-focused attention
on the mouth region, enabling the model to learn precise correlations
between audio features and lip movements.

Key components:
  1. MouthRegionMask — Spatial gaussian attention mask focused on mouth area
  2. MouthROIAttention — Cross-attention weighted by mouth proximity
  3. MouthFeatureExtractor — Lightweight mouth-region feature refinement

Integration with DB-DiT:
  - Placed inside DualBranchBlock, after CBGA
  - Receives video tokens + audio tokens + optional mouth mask
  - Outputs lip-refined video tokens

Architecture:
  Video tokens (B, N_v, D)   Audio tokens (B, N_a, D)
         │                          │
         ├── Spatial split ─────────┤
         │                          │
    Mouth region tokens      Audio query tokens
    (positions near mouth)   (temporal slices)
         │                          │
         └──► MouthROIAttention ◄───┘
                    │
            Lip-refined video tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Standard viseme classes (MPEG-4 / Disney viseme set, 14 classes)
# Each viseme maps to a set of phonemes
# ---------------------------------------------------------------------------
VISEME_TO_PHONEMES = {
    0:  ["sil", "sp"],                    # neutral / silence
    1:  ["p", "b", "m"],                  # bilabial closure
    2:  ["f", "v"],                        # labiodental
    3:  ["th", "dh"],                      # dental
    4:  ["t", "d", "s", "z", "n", "l"],   # alveolar
    5:  ["sh", "zh", "ch", "jh"],          # postalveolar
    6:  ["k", "g", "ng"],                  # velar
    7:  ["w", "r"],                        # rounded glide
    8:  ["iy", "ih"],                      # spread lips (high front)
    9:  ["ey", "eh"],                      # mid front
    10: ["ae", "aa"],                      # low front
    11: ["ao", "ow"],                      # rounded mid back
    12: ["uh", "uw"],                      # rounded high back
    13: ["ah", "ax", "er", "ay", "aw", "oy"],  # neutral open
}

NUM_VISEMES = 14

# Reverse mapping: phoneme → viseme
PHONEME_TO_VISEME = {}
for viseme_id, phones in VISEME_TO_PHONEMES.items():
    for phone in phones:
        PHONEME_TO_VISEME[phone] = viseme_id


class MouthRegionMask(nn.Module):
    """Generates spatial attention masks focused on the mouth region.

    Given latent spatial dimensions (H_lat, W_lat) and an (optional)
    mouth bounding box, produces a soft gaussian-weighted attention mask.

    When mouth bbox is not available, uses a heuristic: the lower-center
    portion of the face (roughly bottom 1/3, center 1/2 of the frame).

    Args:
        default_bottom_ratio: Default mouth position (fraction from bottom).
        default_center_ratio: Default mouth width span (fraction of center).
        sigma_scale: Gaussian sigma relative to image size.
    """

    def __init__(
        self,
        default_bottom_ratio: float = 0.35,
        default_center_ratio: float = 0.5,
        sigma_scale: float = 0.08,
    ):
        super().__init__()
        self.default_bottom_ratio = default_bottom_ratio
        self.default_center_ratio = default_center_ratio
        self.sigma_scale = sigma_scale

    def _create_gaussian_mask(
        self,
        H: int,
        W: int,
        center_y: float,
        center_x: float,
        sigma_y: float,
        sigma_x: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a 2D Gaussian attention mask.

        Args:
            H, W: Grid dimensions.
            center_y, center_x: Gaussian center (normalized [0,1]).
            sigma_y, sigma_x: Gaussian standard deviation (in grid units).

        Returns:
            2D mask (H, W).
        """
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Gaussian
        mask = torch.exp(
            -((grid_y - center_y) ** 2) / (2 * sigma_y ** 2)
            -((grid_x - center_x) ** 2) / (2 * sigma_x ** 2)
        )
        return mask / mask.max()  # Normalize to [0, 1]

    def _default_mouth_center(
        self, H: int, W: int
    ) -> Tuple[float, float, float, float]:
        """Default mouth position heuristic.

        Returns:
            (center_y, center_x, sigma_y, sigma_x) in normalized [0,1].
        """
        center_y = 1.0 - self.default_bottom_ratio / 2
        center_x = 0.5
        sigma_y = self.sigma_scale * 1.5
        sigma_x = self.default_center_ratio / 2
        return center_y, center_x, sigma_y, sigma_x

    def forward(
        self,
        H_lat: int,
        W_lat: int,
        mouth_bbox: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate mouth attention mask for latent grid.

        Args:
            H_lat: Latent spatial height.
            W_lat: Latent spatial width.
            mouth_bbox: Optional bounding box (B, 4) in [x1, y1, x2, y2]
                normalized to [0, 1]. If None, uses default heuristic.

        Returns:
            Attention mask (1, H_lat, W_lat) or (B, H_lat, W_lat).
            Values in [0, 1], 1 = center of mouth.
        """
        B = mouth_bbox.shape[0] if mouth_bbox is not None else 1

        if mouth_bbox is not None:
            # Use provided bbox to compute mouth center
            center_y = (mouth_bbox[:, 1] + mouth_bbox[:, 3]) / 2  # (B,)
            center_x = (mouth_bbox[:, 0] + mouth_bbox[:, 2]) / 2
            sigma_y = (mouth_bbox[:, 3] - mouth_bbox[:, 1]) / 2 * 1.5
            sigma_x = (mouth_bbox[:, 2] - mouth_bbox[:, 0]) / 2 * 1.5

            masks = []
            for b in range(B):
                m = self._create_gaussian_mask(
                    H_lat, W_lat,
                    center_y[b].item(), center_x[b].item(),
                    sigma_y[b].item(), sigma_x[b].item(),
                    device=mouth_bbox.device,
                )
                masks.append(m)
            return torch.stack(masks, dim=0)  # (B, H, W)

        # Default heuristic
        cy, cx, sy, sx = self._default_mouth_center(H_lat, W_lat)
        mask = self._create_gaussian_mask(
            H_lat, W_lat, cy, cx, sy, sx,
            device=torch.device("cpu"),
        )
        return mask.unsqueeze(0)  # (1, H, W)


class MouthROIAttention(nn.Module):
    """Mouth-region-focused cross-attention for lip-sync.

    Extends standard cross-attention with a spatial weighting mechanism
    that emphasizes mouth-region video tokens when attending to audio.

    The mouth mask is broadcast to temporal dimension and applied as
    an attention bias, so tokens near the mouth attend more strongly
    to audio features than tokens in the background.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply QK normalization.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections
        self.q_proj = nn.Linear(dim, dim)  # Video tokens (mouth region query)
        self.k_proj = nn.Linear(dim, dim)  # Audio tokens (key)
        self.v_proj = nn.Linear(dim, dim)  # Audio tokens (value)
        self.out_proj = nn.Linear(dim, dim)

        # QK normalization (SD3/Flux-style)
        if qk_norm:
            from seedance.models.common.norm import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Learnable mouth attention temperature
        # (non-trainable scalar — tuned via hyperparameter search)
        self.register_buffer("mouth_temp", torch.ones(1) * 2.0)

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        mouth_mask: Optional[torch.Tensor] = None,
        T_lat: int = 1,
        H_lat: int = 1,
        W_lat: int = 1,
    ) -> torch.Tensor:
        """Mouth-focused cross-attention: video queries, audio keys/values.

        Args:
            v_tokens: Video tokens (B, N_v, D) where N_v = T_lat * H_lat * W_lat.
            a_tokens: Audio tokens (B, N_a, D).
            mouth_mask: Optional spatial mask (B, H_lat, W_lat) or (1, H_lat, W_lat).
                Higher values = more attention to that position.
            T_lat, H_lat, W_lat: Latent video token grid dimensions.

        Returns:
            Mouth-refined video tokens (B, N_v, D).
        """
        B, N_v, D = v_tokens.shape
        _, N_a, _ = a_tokens.shape

        # Q from video, K/V from audio
        q = self.q_proj(v_tokens)
        k = self.k_proj(a_tokens)
        v = self.v_proj(a_tokens)

        # Reshape to multi-head
        q = q.view(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, N_v, D_h)
        k = k.view(B, N_a, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N_a, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention scores
        attn = q @ k.transpose(-2, -1) * self.scale  # (B, H, N_v, N_a)

        # --- Apply mouth spatial bias ---
        if mouth_mask is not None:
            # Reshape mask to match video token layout
            # mask: (B, H_lat, W_lat) → broadcast to (B, T_lat, H_lat, W_lat)
            if mouth_mask.shape[1:] == (H_lat, W_lat):
                spatial_weights = mouth_mask.unsqueeze(1)  # (B, 1, H_lat, W_lat)
                spatial_weights = spatial_weights.expand(-1, T_lat, -1, -1)
                spatial_weights = spatial_weights.reshape(B, -1)  # (B, N_v)

                # Add as attention bias (positive = more attention)
                # Apply per-head with temperature
                spatial_bias = spatial_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, N_v, 1)
                spatial_bias = spatial_bias * self.mouth_temp  # Scale by learnable temp
                attn = attn + spatial_bias

        # Softmax and output
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, N_v, D_h)
        out = out.permute(0, 2, 1, 3).reshape(B, N_v, D)
        return self.out_proj(out)


class LipSyncBridge(nn.Module):
    """Lip-sync cross-modal bridge for fine-grained mouth-audio alignment.

    Combines MouthROIAttention with a gating mechanism (similar to CBGA)
    and adds a viseme embedding table for explicit phoneme-to-viseme mapping.

    Should be placed alongside CBGA in DualBranchBlock for lip-sync
    capable layers (typically layers 6, 12, 18 in the 24-layer model).

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        num_visemes: Number of viseme classes (default 14).
        qk_norm: Whether to use QK normalization.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_visemes: int = NUM_VISEMES,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_visemes = num_visemes

        # Mouth-focused cross-attention
        self.mouth_attn = MouthROIAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            dropout=dropout,
        )
        self.mouth_norm = nn.LayerNorm(dim)

        # Gate initialized to zero (stable training start)
        self.mouth_gate = nn.Parameter(torch.zeros(1))

        # Viseme embedding table: fixed prototypes for each mouth shape.
        # Each viseme (e.g. bilabial, spread lips) has a dim-dimensional embedding.
        # These are NOT directly trained — instead, audio_viseme_proj learns to
        # map audio features toward these fixed prototypes via cosine similarity.
        self.register_buffer(
            "viseme_embeddings",
            torch.randn(num_visemes, dim) * 0.02,
        )

        # Audio-to-viseme projection: maps audio features to viseme embedding space
        self.audio_viseme_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        # Warmup tracking (same as CBGA)
        self.register_buffer("warmup_steps", torch.tensor(50000))
        # Start at 1 so gate.warmup_scale = 1/50000 ≈ 2e-5 > 0,
        # ensuring gradient flow through mouth_gate from step 0.
        self.register_buffer("current_step", torch.tensor(1))

    def set_step(self, step: int):
        """Update training step for gate warmup."""
        self.current_step.fill_(step)

    def get_gate_scale(self) -> float:
        """Get gate scale based on warmup [0, 1]."""
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self.current_step.item() / self.warmup_steps.item())

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        mouth_mask: Optional[torch.Tensor] = None,
        T_lat: int = 1,
        H_lat: int = 1,
        W_lat: int = 1,
    ) -> torch.Tensor:
        """Apply lip-sync refined attention to video tokens.

        Args:
            v_tokens: Video tokens (B, N_v, D).
            a_tokens: Audio tokens (B, N_a, D).
            mouth_mask: Mouth attention mask (B, H_lat, W_lat).
            T_lat, H_lat, W_lat: Latent grid dims.

        Returns:
            Lip-refined video tokens (B, N_v, D).
        """
        warmup_scale = self.get_gate_scale()

        # Compute viseme logits from audio (always, to keep params connected to graph).
        # Scaled by 0 so they don't affect velocity prediction, but gradients flow.
        _viseme_logits = self.compute_viseme_logits(a_tokens)  # (B, N_a, num_visemes)

        v_norm = self.mouth_norm(v_tokens)
        mouth_out = self.mouth_attn(v_norm, a_tokens, mouth_mask, T_lat, H_lat, W_lat)

        gate = warmup_scale * self.mouth_gate
        v_tokens = v_tokens + gate * mouth_out + 0.0 * _viseme_logits.mean()

        return v_tokens

    def compute_viseme_logits(
        self,
        a_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict viseme class logits from audio tokens.

        Projects audio features to viseme space and computes similarity
        with learned viseme prototypes. The viseme_embeddings participate
        in the forward graph via this dot product.

        Args:
            a_tokens: Audio tokens (B, N_a, D).

        Returns:
            Viseme logits (B, N_a, num_visemes).
        """
        # Project audio tokens to viseme embedding space
        a_proj = self.audio_viseme_proj(a_tokens)  # (B, N_a, D)
        a_proj = F.normalize(a_proj, dim=-1)

        # Normalize viseme embeddings
        viseme_norm = F.normalize(self.viseme_embeddings, dim=-1)  # (num_visemes, D)

        # Cosine similarity → logits
        logits = a_proj @ viseme_norm.T  # (B, N_a, num_visemes)
        # Scale by learnable temperature for sharper predictions
        return logits * 10.0
