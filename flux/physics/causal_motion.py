"""CausalMotion: VLM-guided keyframe trajectory injection.

Zhuang et al., June 2026 — "CausalMotion: Structured Physical Reasoning as
Keyframe and Trajectory Guidance for Training-Free Video Generation"

Key insight: A VLM decomposes a prompt into causally consistent keyframes +
object-centric trajectories, which are injected as soft constraints into a
pretrained DiT — no training required.

This module provides:
  - CausalMotionGuide: inference-time guidance using VLM-decomposed trajectories
  - TrajectoryConstraint: soft constraint on object motion trajectories
  - KeyframeSchedule: temporal schedule for keyframe injection

Overhead: ~1–2 extra VLM calls per generation (amortized over all frames).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyframeSchedule:
    """Temporal schedule for keyframe injection.

    Keyframes are injected at specific temporal positions with
    Gaussian window soft constraints.

    Args:
        num_frames: Total video frames in latent space (T_latent).
        keyframe_positions: List of temporal positions (0.0 to 1.0).
            Default: [0.0, 0.5, 1.0] (start, middle, end).
        window_width: Gaussian window sigma for soft constraint blending.
    """

    def __init__(
        self,
        num_frames: int,
        keyframe_positions: list[float] | None = None,
        window_width: float = 0.15,
    ):
        self.num_frames = num_frames
        self.keyframe_positions = keyframe_positions or [0.0, 0.5, 1.0]
        self.window_width = window_width
        self._compute_windows()

    def _compute_windows(self):
        """Pre-compute Gaussian windows for each keyframe position."""
        T = self.num_frames
        positions = torch.arange(T, dtype=torch.float32)
        self.windows = []  # List of (T,) tensors

        for pos in self.keyframe_positions:
            center = pos * (T - 1)
            sigma = self.window_width * T
            window = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
            self.windows.append(window)

        # Normalize so windows sum to at most 1.0 at each position
        stacked = torch.stack(self.windows, dim=0)  # (K, T)
        max_val = stacked.sum(dim=0).max()
        self.weights = stacked / max(max_val, 1.0)  # (K, T)

    def get_mask(self, device: torch.device) -> torch.Tensor:
        """Get keyframe injection mask.

        Returns:
            (1, K, T, 1, 1) mask for soft keyframe constraint.
        """
        return self.weights.to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


class TrajectoryConstraint:
    """Soft constraint on object motion trajectories.

    Represents a 2D trajectory for a single object over time.
    During generation, the latent at each frame is nudged to match
    the object position specified by the trajectory.

    Args:
        object_name: Name of the tracked object.
        positions: List of (x, y) positions in normalized coordinates [0, 1].
            Length should equal num_frames. Use None for frames without constraint.
        spatial_sigma: Gaussian window size for spatial soft constraint.
    """

    def __init__(
        self,
        object_name: str,
        positions: list[tuple[float, float] | None],
        spatial_sigma: float = 0.1,
    ):
        self.object_name = object_name
        self.positions = positions
        self.spatial_sigma = spatial_sigma

    def to_spatial_mask(
        self, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Convert trajectory to spatial attention mask.

        Returns:
            (T, H, W) mask where high values = object presence.
        """
        T = len(self.positions)
        mask = torch.zeros(T, H, W, device=device)

        for t, pos in enumerate(self.positions):
            if pos is None:
                continue
            x, y = pos
            # Create Gaussian blob at (x, y)
            h_coords = torch.arange(H, device=device, dtype=torch.float32) / H
            w_coords = torch.arange(W, device=device, dtype=torch.float32) / W
            h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
            dist = (h_grid - y) ** 2 + (w_grid - x) ** 2
            mask[t] = torch.exp(-dist / (2 * self.spatial_sigma ** 2))

        return mask


class CausalMotionGuide:
    """VLM-guided keyframe + trajectory injection for training-free physics.

    Usage:
        guide = CausalMotionGuide(model)
        # Decompose prompt into keyframes + trajectories (requires VLM)
        keyframes, trajectories = guide.decompose_prompt(
            "A ball bounces off the wall and rolls to a stop"
        )
        # Inject constraints during sampling
        video = guide.sample_with_guidance(
            v_shape, a_shape, text_emb, keyframes, trajectories
        )

    Args:
        model: DB-DiT model.
        guidance_strength: Overall strength of trajectory guidance.
        keyframe_strength: Strength of keyframe constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_strength: float = 0.3,
        keyframe_strength: float = 0.5,
    ):
        self.model = model
        self.guidance_strength = guidance_strength
        self.keyframe_strength = keyframe_strength

    @staticmethod
    def decompose_prompt(
        prompt: str,
        num_frames: int = 8,
    ) -> tuple[KeyframeSchedule, list[TrajectoryConstraint]]:
        """Decompose a prompt into keyframe schedule + trajectory constraints.

        Uses simple heuristics (in production, replace with VLM call).
        The VLM would identify: objects in motion, their trajectories,
        and causally significant keyframes.

        Args:
            prompt: Text description of the scene + action.
            num_frames: Number of latent frames for the schedule.

        Returns:
            (KeyframeSchedule, list of TrajectoryConstraint).

        Note:
            Falls back to heuristics when VLM is not available.
            In production, call CogVLM2/Video-LLaVA for structured decomposition.
        """
        # Heuristic: detect motion verbs → generate plausible trajectories
        prompt_lower = prompt.lower()
        has_motion = any(
            verb in prompt_lower
            for verb in ["run", "walk", "jump", "fall", "throw", "bounce",
                        "fly", "roll", "slide", "swing", "push", "pull",
                        "spin", "turn", "move", "drop", "lift"]
        )

        # Heuristic keyframe positions based on action structure
        if has_motion:
            keyframe_positions = [0.0, 0.3, 0.6, 1.0]  # More keyframes for motion
        else:
            keyframe_positions = [0.0, 0.5, 1.0]

        keyframes = KeyframeSchedule(
            num_frames=num_frames,
            keyframe_positions=keyframe_positions,
        )

        # Simple trajectory: horizontal motion across the frame
        trajectories = []
        if has_motion:
            # Default trajectory: left-to-right with slight arc
            positions = []
            for t in range(num_frames):
                frac = t / max(num_frames - 1, 1)
                # Parabolic arc trajectory
                x = 0.2 + 0.6 * frac
                y = 0.5 - 0.1 * math.sin(math.pi * frac)  # Slight bounce
                positions.append((x, y))
            trajectories.append(
                TrajectoryConstraint("moving_object", positions)
            )

        return keyframes, trajectories

    @torch.no_grad()
    def apply_spatial_guidance(
        self,
        v_latent: torch.Tensor,
        trajectories: list[TrajectoryConstraint],
        current_t: float,
    ) -> torch.Tensor:
        """Apply soft spatial constraints from trajectories.

        Nudges the video latent to encourage objects at trajectory positions.
        The guidance strength decays over time (stronger at keyframes).

        Args:
            v_latent: Video latent (B, C, T, H, W).
            trajectories: List of trajectory constraints.
            current_t: Current denoising timestep (0 = noise, 1 = clean).

        Returns:
            Guided video latent (same shape).
        """
        if not trajectories:
            return v_latent

        B, C, T, H, W = v_latent.shape
        guidance = torch.zeros_like(v_latent)

        for traj in trajectories:
            spatial_mask = traj.to_spatial_mask(H, W, v_latent.device)  # (T, H, W)

            # Apply mask as soft spatial constraint
            # We add a small perturbation to nudge latent toward trajectory regions
            # This is intentionally very soft — it suggests, not forces
            mask = spatial_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, H, W)

            # The trajectory mask nudges the latent distribution
            # Higher at trajectory positions, lower elsewhere
            guidance = guidance + mask * self.guidance_strength * v_latent

        # Soft blend
        v_latent = (1 - self.guidance_strength) * v_latent + guidance
        return v_latent

    @torch.no_grad()
    def sample_with_guidance(
        self,
        v_shape: tuple,
        a_shape: tuple,
        text_emb: torch.Tensor,
        keyframes: KeyframeSchedule | None = None,
        trajectories: list[TrajectoryConstraint] | None = None,
        null_text_emb: torch.Tensor | None = None,
        num_steps: int = 30,
        cfg_video: float = 5.0,
        cfg_audio: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ODE sampling with keyframe + trajectory guidance.

        Args:
            v_shape, a_shape: Latent shapes.
            text_emb, null_text_emb: Text embeddings for CFG.
            keyframes: Keyframe schedule for temporal constraints.
            trajectories: Object trajectory constraints.
            num_steps: ODE steps.
            cfg_video, cfg_audio: CFG scales.

        Returns:
            (video_latent, audio_latent).
        """
        device = text_emb.device
        dtype = text_emb.dtype
        B = text_emb.shape[0]
        use_cfg = null_text_emb is not None

        z_v = torch.randn(v_shape, device=device, dtype=dtype)
        z_a = torch.randn(a_shape, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        kf_mask = None
        if keyframes is not None:
            kf_mask = keyframes.get_mask(device)  # (1, K, T, 1, 1)

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)

            # Apply spatial trajectory guidance (stronger early, weaker late)
            if trajectories and t_val < 0.5:
                guidance_factor = 1.0 - t_val / 0.5  # Decay from 1→0
                z_v = self.apply_spatial_guidance(z_v, trajectories, t_val)
                # Scale by decay factor
                z_v = z_v * (1 - guidance_factor * self.guidance_strength) + \
                      z_v * (guidance_factor * self.guidance_strength)

            # Velocity prediction
            if use_cfg:
                v_cond, a_cond, *_ = self.model(z_v, z_a, t, text_emb)
                v_uncond, a_uncond, *_ = self.model(z_v, z_a, t, null_text_emb)
                v_pred = v_uncond + cfg_video * (v_cond - v_uncond)
                a_pred = a_uncond + cfg_audio * (a_cond - a_uncond)
            else:
                v_pred, a_pred, *_ = self.model(z_v, z_a, t, text_emb)

            # Euler step
            z_v = z_v + v_pred * dt
            z_a = z_a + a_pred * dt

            # Inject keyframe constraints at early steps
            if kf_mask is not None and t_val < 0.3:
                # Soft constraint: nudge toward temporal consistency
                # Keyframes at positions [0, 0.3, 0.6, 1.0]
                # For early denoising (t < 0.3), only the start keyframe matters
                kf_weight = self.keyframe_strength * (1.0 - t_val / 0.3)
                for kf_idx, kf_pos in enumerate(keyframes.keyframe_positions):
                    if abs(t_val - kf_pos) < 0.1:
                        # At keyframe boundaries, strengthen the latent
                        kf_t = int(kf_pos * v_shape[2])
                        if 0 <= kf_t < v_shape[2]:
                            z_v[:, :, kf_t] = z_v[:, :, kf_t] * (1 + kf_weight)

        return z_v, z_a
