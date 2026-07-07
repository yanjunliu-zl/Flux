"""PhaseLock: Lock motion priors from 2-step generation before visual refinement.

Han et al., ICML 2026 — "Physics in 2-Steps: Locking Motion Priors Before
Visual Refinement Erases Them"

Key insight: 2-step denoising produces more physically accurate motion than
50-step, because later denoising steps corrupt the motion-phase of the latent.
PhaseLock extracts a motion prior from 2 steps and enforces it via Latent Delta
Guidance throughout the remaining steps.

Overhead: ~1.06× (one extra 2-step forward pass + guidance projection per step).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseLockSampler:
    """Inference-time sampler that locks physical motion priors.

    Usage (replaces standard ODE sampler):
        sampler = PhaseLockSampler(model)
        video, audio = sampler.sample(
            v_shape, a_shape, text_emb, num_steps=30,
            phase_steps=2, lock_strength=0.5,
        )

    Args:
        model: DB-DiT model (eval mode).
        phase_steps: Number of coarse steps for motion prior extraction (default: 2).
        lock_strength: Strength of phase constraint (0=off, 1=hard lock).
            Higher values preserve motion better but may reduce visual sharpness.
        lock_schedule: How lock_strength decays over steps.
            "constant" (always same), "linear_decay" (fade to 0), "cosine_decay".
    """

    def __init__(
        self,
        model: nn.Module,
        phase_steps: int = 2,
        lock_strength: float = 0.5,
        lock_schedule: str = "linear_decay",
    ):
        self.model = model
        self.phase_steps = phase_steps
        self.lock_strength = lock_strength
        self.lock_schedule = lock_schedule

    def _get_lock_weight(self, step: int, total_steps: int) -> float:
        """Get current lock strength based on schedule."""
        if self.lock_schedule == "constant":
            return self.lock_strength
        progress = step / max(total_steps - 1, 1)
        if self.lock_schedule == "linear_decay":
            return self.lock_strength * (1.0 - progress)
        elif self.lock_schedule == "cosine_decay":
            import math
            return self.lock_strength * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lock_strength

    @torch.no_grad()
    def extract_motion_prior(
        self,
        v_latent: torch.Tensor,
        a_latent: torch.Tensor,
        text_emb: torch.Tensor,
        null_text_emb: torch.Tensor | None = None,
        cfg_video: float = 5.0,
        cfg_audio: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract motion prior via coarse 2-step denoising.

        Returns the 2-step trajectory (sequence of latent states) which encodes
        temporally-coherent motion but lacks visual detail.
        """
        z_v = v_latent.clone()
        z_a = a_latent.clone()
        dt = 1.0 / self.phase_steps

        for step in range(self.phase_steps):
            t_val = step * dt
            B = z_v.shape[0]
            t = torch.full((B,), t_val, device=z_v.device, dtype=z_v.dtype)

            v_pred, a_pred, *_ = self.model(z_v, z_a, t, text_emb)

            if null_text_emb is not None:
                v_uncond, a_uncond, *_ = self.model(z_v, z_a, t, null_text_emb)
                v_pred = v_uncond + cfg_video * (v_pred - v_uncond)
                a_pred = a_uncond + cfg_audio * (a_pred - a_uncond)

            z_v = z_v + v_pred * dt
            z_a = z_a + a_pred * dt

        return z_v, z_a

    @torch.no_grad()
    def compute_phase_guidance(
        self,
        current_v: torch.Tensor,
        current_a: torch.Tensor,
        motion_v: torch.Tensor,
        motion_a: torch.Tensor,
        lock_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Latent Delta Guidance: steer current latents toward motion prior.

        Uses cosine similarity in latent space as the guidance target.
        The motion prior captures low-frequency temporal structure; we nudge
        the current latents to preserve this structure while refining details.
        """
        # Compute delta (drift) from motion prior
        delta_v = motion_v - current_v
        delta_a = motion_a - current_a

        # Low-pass filter: keep only the motion-scale components
        # Apply temporal blur to isolate low-frequency motion
        if delta_v.shape[2] >= 3:  # Need at least 3 temporal frames
            kernel = torch.tensor([0.25, 0.5, 0.25], device=delta_v.device, dtype=delta_v.dtype)
            kernel = kernel.view(1, 1, -1, 1, 1)
            delta_v_low = F.conv3d(
                F.pad(delta_v, (0, 0, 0, 0, 1, 1), mode="reflect"),
                kernel
            )
            # Blend: keep 80% low-freq, 20% full delta
            delta_v = 0.8 * delta_v_low + 0.2 * delta_v

        # Apply guidance with lock weight
        current_v = current_v + lock_weight * delta_v
        current_a = current_a + lock_weight * delta_a

        return current_v, current_a

    @torch.no_grad()
    def sample(
        self,
        v_shape: tuple,
        a_shape: tuple,
        text_emb: torch.Tensor,
        null_text_emb: torch.Tensor | None = None,
        num_steps: int = 30,
        cfg_video: float = 5.0,
        cfg_audio: float = 4.0,
        first_frame_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PhaseLock sampling: coarse motion → guided refinement.

        Args:
            v_shape, a_shape: Latent shapes.
            text_emb: Text embeddings.
            null_text_emb: Null text for CFG.
            num_steps: Total ODE steps.
            cfg_video, cfg_audio: CFG scales.
            first_frame_latent: Optional I2VA first frame.

        Returns:
            (video_latent, audio_latent).
        """
        device = text_emb.device
        dtype = text_emb.dtype
        B = text_emb.shape[0]
        use_cfg = null_text_emb is not None

        # Initialize noise
        z_v = torch.randn(v_shape, device=device, dtype=dtype)
        z_a = torch.randn(a_shape, device=device, dtype=dtype)

        if first_frame_latent is not None:
            z_v[:, :, 0:1, :, :] = first_frame_latent

        # Step 1: Extract motion prior via coarse 2-step denoising
        motion_v, motion_a = self.extract_motion_prior(
            z_v, z_a, text_emb, null_text_emb, cfg_video, cfg_audio
        )

        # Reset to noise for the full denoising trajectory
        z_v = torch.randn(v_shape, device=device, dtype=dtype)
        z_a = torch.randn(a_shape, device=device, dtype=dtype)
        if first_frame_latent is not None:
            z_v[:, :, 0:1, :, :] = first_frame_latent

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)

            # Get lock weight for this step
            lock_w = self._get_lock_weight(step, num_steps)

            # Apply phase guidance (nudge toward motion prior)
            if lock_w > 0.01:
                z_v, z_a = self.compute_phase_guidance(
                    z_v, z_a, motion_v, motion_a, lock_w
                )

            # Standard velocity prediction
            if use_cfg:
                v_cond, a_cond, *_ = self.model(z_v, z_a, t, text_emb)
                v_uncond, a_uncond, *_ = self.model(z_v, z_a, t, null_text_emb)
                v_pred = v_uncond + cfg_video * (v_cond - v_uncond)
                a_pred = a_uncond + cfg_audio * (a_cond - a_uncond)
            else:
                v_pred, a_pred, *_ = self.model(z_v, z_a, t, text_emb)

            z_v = z_v + v_pred * dt
            z_a = z_a + a_pred * dt

        return z_v, z_a
