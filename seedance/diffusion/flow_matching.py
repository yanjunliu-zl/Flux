"""Flow Matching framework for training and sampling.

Implements conditional flow matching with velocity prediction:
  - Training: predict velocity field v = x_1 - x_0
  - Sampling: ODE integration from t=0 (noise) to t=1 (data)

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
           "Scaling Rectified Flow Transformers" (Esser et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from seedance.diffusion.noise_schedule import LogitNormalSchedule


class FlowMatching(nn.Module):
    """Conditional Flow Matching with velocity prediction.

    Args:
        model: The DB-DiT model that predicts velocity.
        schedule: Timestep schedule (default: LogitNormalSchedule).
        video_weight: Weight for video velocity loss.
        audio_weight: Weight for audio velocity loss.
        sync_weight: Weight for AV sync contrastive loss (Stage 3 only).
    """

    def __init__(
        self,
        schedule: LogitNormalSchedule | None = None,
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
        sync_weight: float = 0.0,
    ):
        super().__init__()
        self.schedule = schedule or LogitNormalSchedule()
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.sync_weight = sync_weight

    def get_training_loss(
        self,
        model: nn.Module,
        v_latent_clean: torch.Tensor,
        a_latent_clean: torch.Tensor,
        text_emb: torch.Tensor,
        first_frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute flow matching training loss.

        Args:
            model: DB-DiT model.
            v_latent_clean: Clean video latent (B, C_v, T_v, H_v, W_v).
            a_latent_clean: Clean audio latent (B, C_a, F_a, T_a).
            text_emb: Text embeddings (B, L_text, D).
            first_frame_mask: Optional I2VA first-frame mask.

        Returns:
            Dict with loss components: "loss", "video_loss", "audio_loss", "sync_loss".
        """
        B = v_latent_clean.shape[0]
        device = v_latent_clean.device

        # Sample timesteps
        t = self.schedule.sample(B, device)  # (B,)

        # Sample noise
        v_noise = torch.randn_like(v_latent_clean)
        a_noise = torch.randn_like(a_latent_clean)

        # Interpolate: x_t = (1 - t) * x_0_noise + t * x_1
        # Where x_0_noise = pure noise, x_1 = clean data
        t_v = t[:, None, None, None, None]
        t_a = t[:, None, None, None]

        v_latent_t = (1 - t_v) * v_noise + t_v * v_latent_clean
        a_latent_t = (1 - t_a) * a_noise + t_a * a_latent_clean

        # If first_frame_mask is provided, keep first frame clean
        if first_frame_mask is not None:
            v_latent_t = v_latent_t * first_frame_mask + v_latent_clean * (1 - first_frame_mask)

        # Predict velocity
        v_pred, a_pred = model(
            v_latent_t, a_latent_t, t, text_emb,
            first_frame_mask=first_frame_mask,
        )

        # Target velocity: v = x_1 - x_0 = clean - noise
        v_target = v_latent_clean - v_noise
        a_target = a_latent_clean - a_noise

        # MSE losses
        video_loss = F.mse_loss(v_pred, v_target)
        audio_loss = F.mse_loss(a_pred, a_target)

        loss = self.video_weight * video_loss + self.audio_weight * audio_loss

        result = {
            "loss": loss,
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "sync_loss": torch.tensor(0.0, device=device),
        }

        # Optional AV sync loss
        if self.sync_weight > 0:
            sync_loss = self._compute_sync_loss(v_pred, a_pred)
            loss = loss + self.sync_weight * sync_loss
            result["loss"] = loss
            result["sync_loss"] = sync_loss.detach()

        return result

    @staticmethod
    def _compute_sync_loss(
        v_pred: torch.Tensor, a_pred: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive AV sync loss using mean-pooled features.

        Encourages aligned video-audio pairs to have higher similarity
        than misaligned pairs (achieved by shifting within batch).

        Args:
            v_pred: Video velocity (B, C_v, T_v, H_v, W_v).
            a_pred: Audio velocity (B, C_a, F_a, T_a).

        Returns:
            Contrastive sync loss scalar.
        """
        # Global mean pooling
        v_feat = v_pred.mean(dim=[2, 3, 4])  # (B, C_v)
        a_feat = a_pred.mean(dim=[2, 3])     # (B, C_a)

        # Normalize
        v_feat = F.normalize(v_feat, dim=-1)
        a_feat = F.normalize(a_feat, dim=-1)

        # Cosine similarity matrix
        sim = v_feat @ a_feat.T  # (B, B)

        # Diagonal: aligned pairs (should be high)
        # Off-diagonal: misaligned pairs (should be low)
        labels = torch.arange(sim.shape[0], device=sim.device)

        # InfoNCE-style loss
        temperature = 0.07
        loss_v = F.cross_entropy(sim / temperature, labels)
        loss_a = F.cross_entropy(sim.T / temperature, labels)

        return (loss_v + loss_a) / 2

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        v_shape: tuple[int, ...],
        a_shape: tuple[int, ...],
        text_emb: torch.Tensor,
        null_text_emb: torch.Tensor | None = None,
        num_steps: int = 30,
        cfg_video: float = 5.0,
        cfg_audio: float = 4.0,
        sampler: str = "heun",
        first_frame_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the flow matching model via ODE integration.

        Args:
            model: DB-DiT model (in eval mode).
            v_shape: Video latent shape (B, C_v, T_v, H_v, W_v).
            a_shape: Audio latent shape (B, C_a, F_a, T_a).
            text_emb: Text embeddings (B, L_text, D).
            null_text_emb: Null text embeddings for CFG.
            num_steps: Number of integration steps.
            cfg_video: CFG scale for video.
            cfg_audio: CFG scale for audio.
            sampler: "euler" or "heun".
            first_frame_latent: Optional first frame latent for I2VA.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        device = text_emb.device
        B = text_emb.shape[0]

        # Initialize with noise at t=0
        z_v = torch.randn(v_shape, device=device, dtype=text_emb.dtype)
        z_a = torch.randn(a_shape, device=device, dtype=text_emb.dtype)

        # CFG setup
        use_cfg = null_text_emb is not None and (cfg_video > 1.0 or cfg_audio > 1.0)

        # Prepare first frame mask for I2VA
        first_frame_mask = None
        if first_frame_latent is not None:
            # Insert clean first frame into noise
            z_v[:, :, 0:1, :, :] = first_frame_latent
            # Create mask: 1 for noise frames, 0 for clean first frame
            T = v_shape[2]
            first_frame_mask = torch.ones(1, 1, T, 1, 1, device=device)
            first_frame_mask[:, :, 0:1, :, :] = 0.0

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.full((B,), t, device=device)

            if use_cfg:
                # Conditional prediction
                v_cond, a_cond = model(
                    z_v, z_a, t_tensor, text_emb,
                    first_frame_mask=first_frame_mask,
                )
                # Unconditional prediction
                v_uncond, a_uncond = model(
                    z_v, z_a, t_tensor, null_text_emb,
                    first_frame_mask=first_frame_mask,
                )
                # Apply CFG
                v_pred = v_uncond + cfg_video * (v_cond - v_uncond)
                a_pred = a_uncond + cfg_audio * (a_cond - a_uncond)
            else:
                v_pred, a_pred = model(
                    z_v, z_a, t_tensor, text_emb,
                    first_frame_mask=first_frame_mask,
                )

            if sampler == "euler":
                # Euler step: x_{t+dt} = x_t + v * dt
                z_v = z_v + v_pred * dt
                z_a = z_a + a_pred * dt

            elif sampler == "heun":
                # Heun 2nd order: predict at midpoint
                z_v_half = z_v + v_pred * dt
                z_a_half = z_a + a_pred * dt

                t_half = t + dt
                t_half_tensor = torch.full((B,), t_half, device=device)

                if use_cfg:
                    v_cond2, a_cond2 = model(
                        z_v_half, z_a_half, t_half_tensor, text_emb,
                        first_frame_mask=first_frame_mask,
                    )
                    v_uncond2, a_uncond2 = model(
                        z_v_half, z_a_half, t_half_tensor, null_text_emb,
                        first_frame_mask=first_frame_mask,
                    )
                    v_pred2 = v_uncond2 + cfg_video * (v_cond2 - v_uncond2)
                    a_pred2 = a_uncond2 + cfg_audio * (a_cond2 - a_uncond2)
                else:
                    v_pred2, a_pred2 = model(
                        z_v_half, z_a_half, t_half_tensor, text_emb,
                        first_frame_mask=first_frame_mask,
                    )

                # Trapezoidal rule
                z_v = z_v + 0.5 * (v_pred + v_pred2) * dt
                z_a = z_a + 0.5 * (a_pred + a_pred2) * dt

        return z_v, z_a
