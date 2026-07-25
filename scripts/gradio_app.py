#!/usr/bin/env python3
"""Gradio web demo for Seedance 2.0.

Usage:
    python scripts/gradio_app.py --checkpoint checkpoints/model.pt --port 7860
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gradio as gr

from flux.models import VideoVAE, AudioVAE, DBDiT, T5Encoder
from flux.pipelines import T2VAPipeline, I2VAPipeline


def create_demo(
    checkpoint_path: str,
    device: str = "cuda",
    share: bool = False,
):
    """Create Gradio web interface."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Build model
    model = DBDiT(dim=768, num_layers=12, num_heads=12)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    vae_video = VideoVAE()
    vae_audio = AudioVAE()
    text_encoder = T5Encoder(device=device, dtype=dtype)

    t2va = T2VAPipeline(
        vae_video=vae_video,
        vae_audio=vae_audio,
        db_dit=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    i2va = I2VAPipeline(
        vae_video=vae_video,
        vae_audio=vae_audio,
        db_dit=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    def generate_t2va(prompt, negative_prompt, duration, steps, cfg_video, cfg_audio, seed):
        num_frames = int(duration * 16)
        video, audio = t2va.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_steps=int(steps),
            cfg_video=cfg_video,
            cfg_audio=cfg_audio,
            seed=int(seed),
        )
        video = video[0].permute(1, 2, 3, 0)  # (C,T,H,W) -> (T,H,W,C)
        video = (video + 1) / 2 * 255
        video = video.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return video

    def generate_i2va(image, prompt, duration, steps, cfg_video, cfg_audio, seed):
        if image is None:
            return None
        num_frames = int(duration * 16)
        video, audio = i2va.generate(
            image=image,
            prompt=prompt,
            num_frames=num_frames,
            num_steps=int(steps),
            cfg_video=cfg_video,
            cfg_audio=cfg_audio,
            seed=int(seed),
        )
        video = video[0].permute(1, 2, 3, 0)
        video = (video + 1) / 2 * 255
        video = video.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return video

    with gr.Blocks(title="Seedance 2.0") as demo:
        gr.Markdown("# 🎬 Seedance 2.0 — Audio-Video Joint Generation")

        with gr.Tab("Text-to-Video-Audio"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="A dog running on grass")
                    neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry")
                    duration = gr.Slider(2, 10, value=2, step=1, label="Duration (seconds)")
                    steps = gr.Slider(10, 50, value=30, step=5, label="Sampling Steps")
                    cfg_v = gr.Slider(1, 10, value=5, step=0.5, label="CFG Video")
                    cfg_a = gr.Slider(1, 10, value=4, step=0.5, label="CFG Audio")
                    seed = gr.Number(value=42, label="Seed")
                    btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    output = gr.Video(label="Generated Video")

            btn.click(
                generate_t2va,
                inputs=[prompt, neg_prompt, duration, steps, cfg_v, cfg_a, seed],
                outputs=output,
            )

        with gr.Tab("Image-to-Video-Audio"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Input Image", type="pil")
                    prompt_i = gr.Textbox(label="Prompt", value="The person turns and smiles")
                    dur_i = gr.Slider(2, 10, value=2, step=1, label="Duration")
                    steps_i = gr.Slider(10, 50, value=30, step=5, label="Steps")
                    cfg_vi = gr.Slider(1, 10, value=5, step=0.5, label="CFG Video")
                    cfg_ai = gr.Slider(1, 10, value=4, step=0.5, label="CFG Audio")
                    seed_i = gr.Number(value=42, label="Seed")
                    btn_i = gr.Button("Generate", variant="primary")
                with gr.Column():
                    output_i = gr.Video(label="Generated Video")

            btn_i.click(
                generate_i2va,
                inputs=[image, prompt_i, dur_i, steps_i, cfg_vi, cfg_ai, seed_i],
                outputs=output_i,
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Seedance 2.0 Gradio Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Share publicly")
    args = parser.parse_args()

    demo = create_demo(args.checkpoint, args.device, args.share)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
