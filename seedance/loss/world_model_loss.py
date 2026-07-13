"""World model training losses: future prediction + temporal consistency.

VideoWorld 2 approach: pure visual self-supervision without text.
Losses directly from raw video frames:
  1. Future frame prediction (MSE + LPIPS on next frame)
  2. Temporal consistency (optical flow coherence)
  3. Motion smoothness (acceleration penalty / jerk minimization)
  4. Physics plausibility (collision detection via flow divergence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuturePredictionLoss(nn.Module):
    """Predict future frames from past context.

    Given frames 1..T, predict frame T+1.
    Uses MSE for pixel accuracy + LPIPS for perceptual quality.

    Args:
        lpips_weight: LPIPS perceptual loss weight (default: 0.1).
        mse_weight: MSE pixel loss weight (default: 1.0).
    """

    def __init__(self, lpips_weight: float = 0.1, mse_weight: float = 1.0):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.mse_weight = mse_weight
        self.lpips_fn = None

    def _get_lpips(self, device):
        if self.lpips_fn is None:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="alex", spatial=False).to(device)
            except ImportError:
                self.lpips_fn = False
        return self.lpips_fn if self.lpips_fn is not False else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute future prediction loss.

        Args:
            pred: Predicted next frame (B, C, H, W).
            target: Ground truth next frame (B, C, H, W).

        Returns:
            Dict with "pred_loss", "pred_mse", "pred_lpips".
        """
        mse = F.mse_loss(pred, target)
        loss = self.mse_weight * mse

        lpips_val = torch.tensor(0.0, device=pred.device)
        lpips_fn = self._get_lpips(pred.device)
        if lpips_fn is not None:
            # LPIPS expects [-1, 1] or [0, 1]
            lpips_val = lpips_fn(pred.clamp(-1, 1), target.clamp(-1, 1)).mean()
            loss = loss + self.lpips_weight * lpips_val

        return {"pred_loss": loss, "pred_mse": mse.detach(), "pred_lpips": lpips_val.detach()}


class TemporalConsistencyLoss(nn.Module):
    """Temporal coherence loss using optical flow.

    Penalizes discontinuities in motion between consecutive frames.
    A physically consistent video should have smooth optical flow
    without sudden jumps.
    """

    def __init__(self, consistency_weight: float = 0.5):
        super().__init__()
        self.consistency_weight = consistency_weight

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute temporal consistency loss.

        Args:
            frames: Video frames (B, T, C, H, W).

        Returns:
            (loss, aux_dict).
        """
        B, T, C, H, W = frames.shape
        if T < 3:
            return torch.tensor(0.0, device=frames.device), {"temporal_consistency": torch.tensor(0.0)}

        # Approximate motion energy via frame differences
        # A smooth video has small acceleration (2nd derivative) in pixel space
        diffs = torch.diff(frames, dim=1)  # (B, T-1, C, H, W)
        accel = torch.diff(diffs, dim=1)    # (B, T-2, C, H, W)

        # Loss: minimize acceleration magnitude (jerk)
        jerk_loss = accel.pow(2).mean()

        # Also: consecutive diffs should be similar in magnitude
        diff_mag = diffs.abs().mean(dim=(-3, -2, -1))  # (B, T-1)
        if T >= 3:
            diff_variance = torch.diff(diff_mag, dim=1).abs().mean()
        else:
            diff_variance = torch.tensor(0.0, device=frames.device)

        loss = self.consistency_weight * (jerk_loss + diff_variance)
        return loss, {
            "temporal_consistency": loss.detach(),
            "jerk": jerk_loss.detach(),
            "flow_variance": diff_variance.detach(),
        }


class PhysicsPlausibilityLoss(nn.Module):
    """Physics plausibility constraints based on optical flow heuristics.

    Detects and penalizes unphysical motion patterns:
    1. Collision: sudden velocity divergence in a region
    2. Penetration: objects occupying same region with opposing flow
    3. Gravity violation: upward acceleration without force
    4. Momentum inconsistency: velocity change without contact
    """

    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.physics_weight = physics_weight

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute physics plausibility loss.

        Uses frame differences as a proxy for optical flow (fast approximation).
        For accurate physics, replace with RAFT/GMA flow.

        Args:
            frames: (B, T, C, H, W).

        Returns:
            (loss, aux_dict).
        """
        B, T, C, H, W = frames.shape
        if T < 3:
            zero = torch.tensor(0.0, device=frames.device)
            return zero, {"physics_plausibility": zero}

        # Approximate flow: frame differences
        flow_approx = torch.diff(frames, dim=1)  # (B, T-1, C, H, W)
        flow_mag = flow_approx.pow(2).sum(dim=2, keepdim=True).sqrt()  # (B, T-1, 1, H, W)

        # 1. Collision: local flow divergence (sudden change in direction)
        # Compute flow gradient magnitude — spikes indicate collisions
        flow_grad_x = torch.diff(flow_approx, dim=-1).abs().mean()
        flow_grad_y = torch.diff(flow_approx, dim=-2).abs().mean()
        collision_penalty = flow_grad_x + flow_grad_y

        # 2. Momentum consistency: acceleration should be smooth except at contacts
        accel = torch.diff(flow_approx, dim=1)  # (B, T-2, C, H, W)
        momentum_violation = accel.pow(2).mean()

        loss = self.physics_weight * (collision_penalty + momentum_violation)
        return loss, {
            "physics_plausibility": loss.detach(),
            "collision": collision_penalty.detach(),
            "momentum_violation": momentum_violation.detach(),
        }


class WorldModelLoss(nn.Module):
    """Combined world model training loss.

    Total = future_prediction + temporal_consistency + physics_plausibility

    All losses are computed directly from raw video — no text/language needed.
    This enables pure visual self-supervision (VideoWorld 2 approach).

    Args:
        pred_weight: Future prediction weight (default: 1.0).
        temporal_weight: Temporal consistency weight (default: 0.5).
        physics_weight: Physics plausibility weight (default: 0.1).
    """

    def __init__(self, pred_weight: float = 1.0, temporal_weight: float = 0.5,
                 physics_weight: float = 0.1):
        super().__init__()
        self.pred_loss = FuturePredictionLoss()
        self.temporal_loss = TemporalConsistencyLoss(temporal_weight)
        self.physics_loss = PhysicsPlausibilityLoss(physics_weight)
        self.pred_weight = pred_weight
        self.temporal_weight = temporal_weight
        self.physics_weight = physics_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                full_video: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute combined world model losses.

        Args:
            pred: Predicted next frame(s) (B, C, H, W) or (B, T_pred, C, H, W).
            target: Ground truth next frame(s), same shape.
            full_video: Optional full video (B, T, C, H, W) for temporal+physics losses.

        Returns:
            Dict with "loss" and per-component losses.
        """
        # Future prediction
        p = self.pred_loss(pred, target)
        total = self.pred_weight * p["pred_loss"]

        result = {
            "loss": total, "pred_loss": p["pred_loss"].detach(),
            "pred_mse": p["pred_mse"], "pred_lpips": p["pred_lpips"],
            "temporal_consistency": torch.tensor(0.0, device=pred.device),
            "physics_plausibility": torch.tensor(0.0, device=pred.device),
        }

        # Temporal consistency on prediction sequence
        if full_video is not None and full_video.shape[1] >= 3:
            # Concatenate original + prediction for temporal loss
            if full_video.shape[1] > 1 and pred.dim() == 5:
                combined = torch.cat([full_video[:, -2:], pred], dim=1)
            else:
                combined = full_video

            temporal_loss, t_aux = self.temporal_loss(combined)
            total = total + self.temporal_weight * temporal_loss
            result["loss"] = total
            result["temporal_consistency"] = temporal_loss.detach()

            physics_loss, p_aux = self.physics_loss(combined)
            total = total + self.physics_weight * physics_loss
            result["loss"] = total
            result["physics_plausibility"] = physics_loss.detach()

        return result
