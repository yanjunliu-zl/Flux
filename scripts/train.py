#!/usr/bin/env python3
"""Seedance 2.0 Training Entry Point.

Usage:
    # Stage 1: Video pretraining
    python scripts/train.py --config configs/train/stage1_video_pretrain.yaml

    # Stage 3: AV joint training (resume)
    python scripts/train.py --config configs/train/stage3_av_joint.yaml \
        --resume checkpoints/stage3_av_joint/step_100000.pt

    # Override config values from CLI
    python scripts/train.py --config configs/train/stage1_video_pretrain.yaml \
        training.batch_size=8 training.max_steps=100000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from flux.utils.config import load_config
from flux.utils.logging import setup_logging
from flux.models import DBDiT, T5Encoder
from flux.data import VideoDataset, AudioDataset, AVDataset
from flux.data.collate import collate_video_batch, collate_audio_batch, collate_av_batch
from flux.training import Trainer, setup_distributed, wrap_model


def build_dataset(config, stage: int):
    """Build dataset and collate function based on training stage."""
    data_cfg = config.training.data

    if stage == 1:
        dataset = VideoDataset(
            manifest_path=data_cfg.manifest_path,
            resolution=data_cfg.get("resolution", 256),
            num_frames=data_cfg.get("num_frames", 32),
            frame_stride=data_cfg.get("frame_stride", 1),
            fps_condition=data_cfg.get("fps_condition", True),
            caption_dropout_prob=data_cfg.get("caption_dropout_prob", 0.1),
        )
        collate_fn = collate_video_batch
    elif stage == 2:
        dataset = AudioDataset(
            manifest_path=data_cfg.manifest_path,
            sample_rate=data_cfg.get("sample_rate", 16000),
            n_mels=data_cfg.get("n_mels", 80),
            hop_length=data_cfg.get("hop_length", 256),
            caption_dropout_prob=data_cfg.get("caption_dropout_prob", 0.1),
        )
        collate_fn = collate_audio_batch
    elif stage in (3, 4):
        dataset = AVDataset(
            manifest_path=data_cfg.manifest_path,
            video_resolution=data_cfg.video.get("resolution", 256),
            video_num_frames=data_cfg.video.get("num_frames", 32),
            video_frame_stride=data_cfg.video.get("frame_stride", 1),
            audio_sample_rate=data_cfg.audio.get("sample_rate", 16000),
            caption_dropout_prob=data_cfg.get("caption_dropout_prob", 0.1),
            first_frame_condition_prob=data_cfg.get("first_frame_condition_prob", 0.3),
        )
        collate_fn = collate_av_batch
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    return dataset, collate_fn


def build_model(config, stage: int):
    """Build DB-DiT model based on training stage."""
    model_cfg = config.training.get("model_init", {})

    # Load model config (small or base)
    model_config_path = model_cfg.get(
        "config", "configs/model/db_dit_small.yaml"
    )
    model_config = load_config(model_config_path)

    model = DBDiT(
        dim=model_config.model.dim,
        num_layers=model_config.model.num_layers,
        num_heads=model_config.model.num_heads,
        ffn_ratio=model_config.model.ffn_ratio,
        qk_norm=model_config.model.qk_norm,
        dropout=model_config.model.get("dropout", 0.0),
        cbga_layers=model_config.model.cbga_layers,
        cbga_gate_warmup_steps=model_config.model.cbga_gate_warmup_steps,
        video_patch_size=tuple(model_config.model.video.patch_size),
        video_latent_channels=model_config.model.video.latent_channels,
        video_rope_theta=model_config.model.video.rope_theta,
        video_rope_dim_t=model_config.model.video.get("rope_dim_t"),
        video_rope_dim_h=model_config.model.video.get("rope_dim_h"),
        video_rope_dim_w=model_config.model.video.get("rope_dim_w"),
        audio_patch_size=tuple(model_config.model.audio.patch_size),
        audio_latent_channels=model_config.model.audio.latent_channels,
        audio_rope_theta=model_config.model.audio.rope_theta,
    )

    # Load pretrained weights if specified
    if "checkpoint" in model_cfg:
        print(f"Loading pretrained weights from {model_cfg.checkpoint}")
        state_dict = torch.load(model_cfg.checkpoint, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)

    return model


def main():
    parser = argparse.ArgumentParser(description="Seedance 2.0 Training")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, overrides=args.overrides)
    stage = config.training.stage
    print(f"[Train] Stage {stage}: {config.training.description}")

    # Setup distributed
    local_rank, world_size, device = setup_distributed()
    logger = setup_logging()

    # Build dataset
    dataset, collate_fn = build_dataset(config, stage)
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model = build_model(config, stage)
    model = wrap_model(
        model,
        mixed_precision=config.training.get("mixed_precision", "bf16"),
        activation_checkpointing=config.training.get("gradient_checkpointing", True),
    )

    # Build text encoder
    model_cfg = config.training.get("model_init", {})
    model_config_path = model_cfg.get("config", "configs/model/db_dit_small.yaml")
    try:
        dit_cfg = load_config(model_config_path)
        text_model_name = dit_cfg.model.text.encoder
    except Exception:
        text_model_name = "google/t5-v1_1-base"
    try:
        text_encoder = T5Encoder(model_name=text_model_name, device=device)
    except Exception as e:
        print(f"[Warning] T5 encoder not available ({e}), using zero embeddings")
        text_encoder = None

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=dict(config.training),
        device=device,
        text_encoder=text_encoder,
    )

    # Train
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
