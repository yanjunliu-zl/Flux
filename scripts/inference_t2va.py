#!/usr/bin/env python3
"""Seedance 2.0 Text-to-Video-Audio Inference.

Usage:
    python scripts/inference_t2va.py --checkpoint checkpoints/step_0000200.pt \
        --prompt "A cat playing piano" --output outputs/cat_piano.mp4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from flux.utils.config import load_config
from flux.models import VideoVAE, AudioVAE, DBDiT, T5Encoder
from flux.pipelines import T2VAPipeline


def build_model_from_config(model_config_path: str, device: torch.device, dtype: torch.dtype) -> DBDiT:
    """Build DB-DiT model matching the training configuration exactly."""
    cfg = load_config(model_config_path).model
    model = DBDiT(
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ffn_ratio=cfg.get("ffn_ratio", 4.0),
        qk_norm=cfg.get("qk_norm", True),
        dropout=cfg.get("dropout", 0.0),
        cbga_layers=list(cfg.cbga_layers),
        cbga_gate_warmup_steps=cfg.cbga_gate_warmup_steps,
        video_patch_size=tuple(cfg.video.patch_size),
        video_latent_channels=cfg.video.latent_channels,
        video_rope_theta=cfg.video.rope_theta,
        video_rope_dim_t=cfg.video.get("rope_dim_t"),
        video_rope_dim_h=cfg.video.get("rope_dim_h"),
        video_rope_dim_w=cfg.video.get("rope_dim_w"),
        audio_patch_size=tuple(cfg.audio.patch_size),
        audio_latent_channels=cfg.audio.latent_channels,
        audio_rope_theta=cfg.audio.rope_theta,
    )
    return model.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Seedance 2.0 T2VA Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Training checkpoint .pt")
    parser.add_argument("--model_config", type=str, default="configs/model/db_dit_small.yaml",
                        help="Model config used during training")
    parser.add_argument("--t5_model", type=str, default="google/t5-v1_1-base",
                        help="T5 model matching training config")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted")
    parser.add_argument("--output", type=str, default="outputs/output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=16, help="Video frames")
    parser.add_argument("--width", type=int, default=256, help="Video width")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--steps", type=int, default=10, help="Sampling steps")
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--cfg_video", type=float, default=3.0, help="Video CFG scale")
    parser.add_argument("--cfg_audio", type=float, default=1.0, help="Audio CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 (avoid bf16 issues)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.bfloat16

    print(f"[T2VA] Loading model config: {args.model_config}")
    model = build_model_from_config(args.model_config, device, dtype)

    print(f"[T2VA] Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # Filter to match model params (checkpoint may have extra keys from FSDP)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    skipped = len(state_dict) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[T2VA] Skipped {skipped} incompatible keys (FSDP prefix mismatch)")
    model.eval()

    print(f"[T2VA] Loading T5: {args.t5_model}")
    text_encoder = T5Encoder(model_name=args.t5_model, device=device)

    print(f"[T2VA] Prompt: {args.prompt}")
    print(f"[T2VA] Device: {device}, dtype: {dtype}, steps: {args.steps}, sampler: {args.sampler}")

    pipeline = T2VAPipeline(
        vae_video=VideoVAE(),
        vae_audio=AudioVAE(),
        db_dit=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    pipeline.generate_to_file(
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_steps=args.steps,
        sampler=args.sampler,
        cfg_video=args.cfg_video,
        cfg_audio=args.cfg_audio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
