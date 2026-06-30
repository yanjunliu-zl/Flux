"""Main training loop for Seedance 2.0 multi-stage training."""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import defaultdict

from seedance.training.optimizer import get_optimizer
from seedance.training.lr_scheduler import get_lr_scheduler
from seedance.training.ema import EMA
from seedance.training.distributed import is_main_process
from seedance.diffusion.flow_matching import FlowMatching
import seedance.utils.checkpoint as ckpt_utils


@dataclass
class TrainingState:
    """Tracks training progress for checkpointing and resumption."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)


class Trainer:
    """Training loop orchestrator for Seedance 2.0.

    Handles:
    - Multi-stage curriculum (Video pretrain → Audio pretrain → AV joint → Hi-res)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - EMA parameter tracking
    - Checkpointing and resumption
    - Logging (console + WandB)

    Args:
        model: DB-DiT model (or VAE model for VAE training phases).
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        config: Training configuration dict.
        device: Target device.
        text_encoder: T5 text encoder for encoding captions.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: dict,
        device: torch.device,
        text_encoder: nn.Module | None = None,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.text_encoder = text_encoder

        # Extract config values
        self.max_steps = config.get("max_steps", 500000)
        self.batch_size = config.get("batch_size", 4)
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = config.get("mixed_precision", "bf16")
        self.log_every = config.get("logging", {}).get("log_every", 100)
        self.sample_every = config.get("logging", {}).get("sample_every", 5000)
        self.checkpoint_every = config.get("logging", {}).get("checkpoint_every", 10000)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.use_wandb = config.get("logging", {}).get("wandb", False)

        # Setup optimizer
        opt_cfg = config.get("optimizer", {})
        self.optimizer = get_optimizer(
            model,
            lr=opt_cfg.get("lr", 2e-4),
            betas=opt_cfg.get("betas", (0.9, 0.999)),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            optimizer_type=opt_cfg.get("type", "adamw"),
        )

        # Setup scheduler
        sched_cfg = config.get("scheduler", {})
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            warmup_steps=sched_cfg.get("warmup_steps", 5000),
            max_steps=self.max_steps,
            min_lr=sched_cfg.get("min_lr", 1e-5),
        )

        # Setup EMA
        ema_cfg = config.get("ema", {})
        self.ema = EMA(model, decay=ema_cfg.get("decay", 0.9999)) if ema_cfg else None

        # Setup AMP scaler
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=(self.mixed_precision == "fp16")
        )

        # Setup flow matching
        loss_cfg = config.get("loss", {})
        self.flow_matching = FlowMatching(
            video_weight=loss_cfg.get("video_weight", 1.0),
            audio_weight=loss_cfg.get("audio_weight", 1.0),
            sync_weight=loss_cfg.get("sync_weight", 0.0),
        )

        # State
        self.state = TrainingState()

        # WandB
        self.wandb_run = None
        if self.use_wandb and is_main_process():
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="seedance",
                    config=config,
                    name=config.get("description", "training"),
                )
            except ImportError:
                self.use_wandb = False

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(
        self, batch: dict
    ) -> dict[str, float]:
        """Single training step (with gradient accumulation).

        Args:
            batch: Dict with "video", "mel", "caption", optionally "first_frame".

        Returns:
            Dict of loss values.
        """
        video = batch["video"].to(self.device)
        # Audio may not be present in Stage 1 (video-only) training
        if "mel" in batch:
            audio = batch["mel"].to(self.device)
        else:
            # Dummy audio latent — (B, 8, 4, 16) works with patch (1,4)
            B = video.shape[0]
            audio = torch.randn(B, 8, 4, 16, device=self.device, dtype=video.dtype)
        captions = batch["caption"]

        # Encode text
        if self.text_encoder is not None:
            text_emb = self.text_encoder(captions).to(self.device)
        else:
            text_emb = torch.zeros(
                len(captions), self.model.dim, device=self.device
            )

        # Run video through VideoVAE to get clean latent
        # (In practice, this is done offline or via a pre-loaded VAE)
        # Here we assume video is already in latent space, or we use a VAE
        # For simplicity, treat video as needing VAE encoding
        # Actual implementation would call self.video_vae.encode(video)

        # First frame conditioning
        first_frame_mask = None
        if "first_frame" in batch and batch["first_frame"] is not None:
            first_frame_mask = torch.ones(
                1, 1, video.shape[2], 1, 1, device=self.device
            )
            first_frame_mask[:, :, 0:1, :, :] = 0.0

        # Preprocess: raw pixel video -> "latent-like" for DB-DiT
        # In production: VideoVAE.encode(video). For now: downscale + pad channels
        B, C, T, H, W = video.shape
        v_latent = torch.nn.functional.interpolate(
            video.reshape(B*T, C, H, W), size=(H//8, W//8), mode='bilinear', antialias=True
        ).reshape(B, C, T, H//8, W//8)
        # Pad 3 channels -> 16 channels (VAE latent dim)
        v_latent = torch.cat([
            v_latent, torch.zeros(B, 13, T, H//8, W//8, device=self.device, dtype=video.dtype)
        ], dim=1)
        a_latent = audio  # AudioVAE.encode(audio) in practice

        # Compute flow matching loss
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32,
        ):
            losses = self.flow_matching.get_training_loss(
                self.model, v_latent, a_latent, text_emb,
                first_frame_mask=first_frame_mask,
            )
            loss = losses["loss"] / self.grad_accum_steps

        # Backward
        self.scaler.scale(loss).backward()

        return {k: v.item() for k, v in losses.items()}

    def train(self, resume_from: str | None = None):
        """Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from.
        """
        # Resume if specified
        if resume_from is not None:
            self.state = ckpt_utils.load_checkpoint(
                self.model, self.optimizer, self.scheduler,
                resume_from, self.device,
            )

        self.model.train()
        self.model.to(self.device)

        step = self.state.step
        data_iter = iter(self.train_loader)
        running_losses = defaultdict(float)

        print(f"[Trainer] Starting training from step {step}, max_steps={self.max_steps}")
        t_start = time.time()

        while step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.state.epoch += 1

            # Training step
            losses = self.train_step(batch)

            for k, v in losses.items():
                running_losses[k] += v

            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # EMA update
                if self.ema is not None:
                    self.ema.update(self.model)

                # Update model step counter for CBGA warmup
                if hasattr(self.model, "set_step"):
                    self.model.set_step(step)

            step += 1
            self.state.step = step

            # Logging
            if step % self.log_every == 0 and is_main_process():
                elapsed = time.time() - t_start
                steps_per_sec = self.log_every / max(elapsed, 1e-8)
                avg_losses = {
                    k: v / self.log_every for k, v in running_losses.items()
                }

                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"[Step {step}/{self.max_steps}] "
                    + " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
                    + f" | lr={lr:.2e} | {steps_per_sec:.1f} steps/s"
                )

                if self.wandb_run is not None:
                    self.wandb_run.log({
                        **{f"train/{k}": v for k, v in avg_losses.items()},
                        "train/lr": lr,
                        "train/step": step,
                    })

                running_losses.clear()
                t_start = time.time()

            # Checkpoint
            if step % self.checkpoint_every == 0 and is_main_process():
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"step_{step:07d}.pt"
                )
                ckpt_utils.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.state, ckpt_path,
                )
                print(f"[Checkpoint] Saved to {ckpt_path}")

        print(f"[Trainer] Training complete at step {step}")
