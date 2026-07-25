#!/usr/bin/env python3
"""Seedance 2.0 Image-to-Video-Audio Inference.

Usage:
    python scripts/inference_i2va.py --config configs/inference/i2va.yaml
    python scripts/inference_i2va.py --config configs/inference/i2va.yaml \
        --image inputs/photo.jpg --prompt "Person turns and smiles"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from flux.utils.config import load_config
from flux.models import VideoVAE, AudioVAE, DBDiT, T5Encoder
from flux.pipelines import I2VAPipeline


def main():
    parser = argparse.ArgumentParser(description="Seedance 2.0 I2VA Inference")
    parser.add_argument("--config", type=str, default="configs/inference/i2va.yaml")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg_video", type=float, default=None)
    parser.add_argument("--cfg_audio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    inf_cfg = config.inference

    prompt = args.prompt or inf_cfg.prompt
    negative_prompt = args.negative_prompt or inf_cfg.get("negative_prompt", "")
    checkpoint_path = args.checkpoint or inf_cfg.checkpoint_path
    output_path = args.output or inf_cfg.get("output_path", "output_i2va.mp4")
    num_steps = args.steps or inf_cfg.num_steps
    cfg_video = args.cfg_video or inf_cfg.cfg_video
    cfg_audio = args.cfg_audio or inf_cfg.cfg_audio
    seed = args.seed or inf_cfg.seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[I2VA] Image: {args.image}, Prompt: {prompt}")

    model_cfg = inf_cfg.model
    model = DBDiT(
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.get("num_heads", model_cfg.dim // 64),
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    vae_video = VideoVAE()
    vae_audio = AudioVAE()
    text_encoder = T5Encoder(device=device)

    pipeline = I2VAPipeline(
        vae_video=vae_video,
        vae_audio=vae_audio,
        db_dit=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    pipeline.generate_to_file(
        image=args.image,
        prompt=prompt,
        output_path=output_path,
        negative_prompt=negative_prompt,
        num_frames=inf_cfg.get("num_frames", 32),
        width=inf_cfg.get("width", 256),
        height=inf_cfg.get("height", 256),
        fps=inf_cfg.get("fps", 16),
        num_steps=num_steps,
        sampler=inf_cfg.get("sampler", "heun"),
        cfg_video=cfg_video,
        cfg_audio=cfg_audio,
        seed=seed,
    )


if __name__ == "__main__":
    main()
