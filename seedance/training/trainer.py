"""Main training loop for Seedance 2.0 multi-stage training."""

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.text_encoder = text_encoder
        self.world_size = world_size
        self.local_rank = local_rank

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
        self.use_tensorboard = config.get("logging", {}).get("tensorboard", False)
        self.tb_log_dir = config.get("logging", {}).get("tensorboard_log_dir", "runs")

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

        # TensorBoard
        self.tb_writer = None
        if self.use_tensorboard and is_main_process():
            self.tb_writer = SummaryWriter(log_dir=self.tb_log_dir)
            print(f"[Trainer] TensorBoard logging to: {self.tb_log_dir}")

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
        # video: (B, C, T, H, W)  e.g. (4, 3, 32, 256, 256)
        B, C, T, H, W = video.shape
        # Merge B and T for 2D interpolation: (B*T, C, H, W)
        v_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        v_flat = torch.nn.functional.interpolate(
            v_flat, size=(H // 8, W // 8), mode='bilinear', antialias=True,
        )
        # Restore to (B, C, T, H', W')
        v_latent = v_flat.reshape(B, T, C, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        # Pad 3 channels -> 16 channels (VAE latent dim)
        v_latent = torch.cat([
            v_latent,
            torch.zeros(B, 13, T, H // 8, W // 8, device=self.device, dtype=video.dtype),
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

        if is_main_process():
            print(f"[Trainer] Starting training from step {step}, max_steps={self.max_steps}")
        t_start = time.time()

        while step < self.max_steps:
            # Get batch (with epoch-aware DistributedSampler)
            try:
                batch = next(data_iter)
            except StopIteration:
                self.state.epoch += 1
                # Notify DistributedSampler of new epoch for different shuffle
                if (hasattr(self.train_loader, "sampler")
                        and hasattr(self.train_loader.sampler, "set_epoch")):
                    self.train_loader.sampler.set_epoch(self.state.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

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

            # Logging (all-reduce losses across GPUs for accurate reporting)
            if step % self.log_every == 0:
                avg_losses = {
                    k: v / self.log_every for k, v in running_losses.items()
                }
                # Average losses across all ranks
                from seedance.training.distributed import all_reduce_losses
                avg_losses = all_reduce_losses(avg_losses)

                if is_main_process():
                    elapsed = time.time() - t_start
                    steps_per_sec = self.log_every / max(elapsed, 1e-8)
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

                    if self.tb_writer is not None:
                        for k, v in avg_losses.items():
                            self.tb_writer.add_scalar(f"train/{k}", v, step)
                        self.tb_writer.add_scalar("train/lr", lr, step)
                        self.tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, step)

                running_losses.clear()
                t_start = time.time()

            # Checkpoint (only on main process, FSDP-aware)
            if step % self.checkpoint_every == 0 and is_main_process():
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"step_{step:07d}.pt"
                )
                ckpt_utils.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.state, ckpt_path,
                )
                print(f"[Checkpoint] Saved to {ckpt_path}")

        # Cleanup
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

        print(f"[Trainer] Training complete at step {step}")


if __name__ == "__main__":
    # Delegate to __main__.py when run as python -m seedance.training.trainer
    from seedance.training.__main__ import main
    main()
