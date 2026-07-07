"""VPT: Role-aware signals + modality-decoupled denoising.

Zheng et al., July 2026 — "Enhancing Video Physical Consistency via Role-aware
Joint Training and Modality-decoupled Denoising"

Key innovations:
1. Role-aware captioning: tag objects by physical role (agent, controlled,
   passive, background) to give the model causal structure
2. Modality-decoupled noise: visual and auxiliary (flow) modalities get
   independent noise levels, preventing the model from blurring them together

Results: 39.4% relative improvement on VideoPhy benchmark over Wan2.1-T2V-1.3B.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleAwareCaptioner:
    """Augments captions with physical role tags.

    Converts: "A person pushes a box across the floor"
    Into:     "[agent: person] pushes [controlled: box] across [passive: floor]"

    The role tags help the model distinguish:
    - agent:       Entity that initiates force/motion
    - controlled:  Entity being acted upon
    - passive:     Static environment elements
    - background:  Scene context (always present but never interacts)

    Usage:
        captioner = RoleAwareCaptioner()
        tagged = captioner.augment("A person pushes a box")
        # → "agent: person pushes controlled: box across passive: floor"
    """

    ROLE_KEYWORDS = {
        "agent": [
            "person", "man", "woman", "child", "dog", "cat", "bird",
            "athlete", "dancer", "driver", "player", "robot", "hand",
            "car", "truck", "bus", "train",
        ],
        "controlled": [
            "ball", "box", "cup", "book", "chair", "door", "bag",
            "bicycle", "kite", "umbrella", "phone", "bottle",
            "rock", "paper", "toy", "fruit", "vegetable",
        ],
        "passive": [
            "floor", "ground", "wall", "table", "desk", "road", "street",
            "water", "grass", "sand", "snow", "sky", "ceiling", "shelf",
        ],
    }

    def augment(self, caption: str) -> str:
        """Add role tags to a caption.

        Args:
            caption: Raw text caption.

        Returns:
            Role-tagged caption.
        """
        words = caption.lower().split()
        tagged = []

        for word in words:
            # Strip punctuation for matching
            clean = word.strip(".,!?;:\"'")
            tag = None
            for role, keywords in self.ROLE_KEYWORDS.items():
                if clean in keywords:
                    tag = role
                    break

            if tag:
                tagged.append(f"[{tag}: {clean}]")
            else:
                tagged.append(word)

        return " ".join(tagged)

    def augment_batch(self, captions: list[str]) -> list[str]:
        """Apply role tags to a batch."""
        return [self.augment(c) for c in captions]


class VPTNoiseScheduler:
    """Modality-decoupled noise scheduling.

    Standard flow matching applies the SAME noise level to all modalities
    (visual pixels, optical flow if used). This causes the model to treat
    appearance and motion as equally noisy at each step, blurring causal
    physical structure.

    VPT applies INDEPENDENT noise:
    - Visual modality: standard logit-normal schedule
    - Flow modality:   less noise (biased toward t=1) to preserve motion signal

    Args:
        flow_noise_shift: How much less noise to apply to flow modality (default: 0.3).
            Higher = more flow signal preserved, better physics but less diverse.
        flow_noise_std: Standard deviation of flow noise (default: 0.5).
    """

    def __init__(
        self,
        flow_noise_shift: float = 0.3,
        flow_noise_std: float = 0.5,
    ):
        self.flow_noise_shift = flow_noise_shift
        self.flow_noise_std = flow_noise_std

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample decoupled timesteps for visual and flow modalities.

        Returns:
            (t_visual, t_flow): Visual timesteps in [0,1], flow timesteps shifted
            toward lower noise levels (closer to 1).
        """
        # Visual: standard logit-normal
        eps_v = torch.randn(batch_size, device=device)
        t_visual = torch.sigmoid(eps_v)

        # Flow: shifted toward t=1 (cleaner) to preserve motion signal
        eps_f = self.flow_noise_std * torch.randn(batch_size, device=device)
        t_flow = torch.sigmoid(eps_f + self.flow_noise_shift)
        t_flow = t_flow.clamp(0.0, 1.0)

        return t_visual, t_flow

    def add_noise(
        self,
        visual_clean: torch.Tensor,
        flow_clean: torch.Tensor | None,
        t_visual: torch.Tensor,
        t_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Add modality-decoupled noise.

        Args:
            visual_clean: Clean video latent (B, C_v, T, H, W).
            flow_clean: Clean optical flow (B, 2, T, H, W) or None.
            t_visual: Visual timesteps (B,).
            t_flow: Flow timesteps (B,).

        Returns:
            (visual_noisy, flow_noisy).
        """
        # Visual noise
        v_noise = torch.randn_like(visual_clean)
        t_v = t_visual[:, None, None, None, None]
        visual_noisy = (1 - t_v) * v_noise + t_v * visual_clean

        # Flow noise (less noisy — preserves motion)
        if flow_clean is not None:
            f_noise = torch.randn_like(flow_clean)
            t_f = t_flow[:, None, None, None, None]
            flow_noisy = (1 - t_f) * f_noise + t_f * flow_clean
        else:
            flow_noisy = None

        return visual_noisy, flow_noisy
