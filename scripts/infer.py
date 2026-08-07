#!/usr/bin/env python3
"""Quick inference script to test the current checkpoint visually.

Usage:
    python scripts/infer.py --checkpoint checkpoints/step_0060000.pt --prompt "..."
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from flux.models.db_dit import DBDiT
from flux.pipelines.pipeline_t2va import T2VAPipeline
from flux.utils.vae_utils import build_video_vae


class FakeVAE(nn.Module):
    """Stub VAE for bilinear fallback when no real VideoVAE is available.

    decode_latent() in vae_utils detects this via _is_fake_vae() and uses
    bilinear upscale instead of real VAE decode.
    """
    def __init__(self, latent_channels: int, sample_rate: int = 16000):
        super().__init__()
        self.latent_channels = latent_channels
        self.sample_rate = sample_rate
        self.mel_transform = FakeMelTransform()

    def decode(self, x):
        raise NotImplementedError("Decode is handled inline in pipeline_t2va")

    def encode(self, x):
        raise NotImplementedError("Encode is not used in inference")


class FakeMelTransform:
    def get_output_length(self, num_samples: int) -> int:
        """Approximate mel frame count for given audio samples.

        Standard: hop_length=160, so ~100 frames per second at 16kHz.
        """
        return num_samples // 160


class FakeT5(nn.Module):
    """Stub T5 that returns random embeddings — for testing visual quality only.

    Real training uses t5-base from HuggingFace.
    """
    def __init__(self, dim: int = 768, max_length: int = 77):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        # Register a dummy buffer so nn.Module.to(device) works correctly
        self.register_buffer('_device_tracker', torch.zeros(1))

    def forward(self, texts: list[str]) -> torch.Tensor:
        B = len(texts)
        dev = self._device_tracker.device
        return torch.randn(B, self.max_length, self.dim, device=dev)

    def __call__(self, texts: list[str]):
        return self.forward(texts)


def load_real_t5(device, model_name: str = "t5-base"):
    """Load real T5 from HuggingFace (already cached from training)."""
    from flux.models.text_encoder.t5_encoder import T5Encoder
    t5 = T5Encoder(model_name=model_name, max_length=77, device=str(device))
    print(f"  [T5] Loaded: {model_name}")
    return t5


def load_model(checkpoint_path: str, device: torch.device):
    """Load DB-DiT model from training checkpoint.

    Architecture matches stage1_video_pretrain.yaml: dim=1024, layers=24, heads=16.
    """
    print(f"\n[Model] Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract model state
    if "model" in state:
        model_state = state["model"]

        # Print training info
        train_state = state.get("state", {})
        step = train_state.get("step", "?")
        loss = train_state.get("loss", "?")
        print(f"  Step: {step}, Loss: {loss}")
    else:
        model_state = state

    # Build model matching stage1 config
    model = DBDiT(
        dim=1024,
        num_layers=24,
        num_heads=16,
        context_dim=768,          # t5-base
        ffn_ratio=4.0,
        qk_norm=True,
        dropout=0.0,
        cbga_layers=list(range(12, 24)),  # layers 12-23 (half of 24 layers)
        video_patch_size=(1, 2, 2),
        video_latent_channels=16,
        audio_patch_size=(1, 4),
        audio_latent_channels=8,
    )

    # Load weights
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)}")
        # Only show first few
        for k in missing[:3]:
            print(f"    {k}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:3]:
            print(f"    {k}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.1f}M")
    return model


def main():
    parser = argparse.ArgumentParser(description="Quick inference with Seedance checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/step_0060000.pt")
    parser.add_argument("--prompt", default="A person smiling at the camera, natural lighting")
    parser.add_argument("--negative_prompt", default="low quality, blurry, distorted, watermark, text, deformed")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--cfg_video", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/inference_test.mp4")
    parser.add_argument("--t5_model", default="t5-base")
    parser.add_argument("--vae_checkpoint", default=None,
                        help="Path to VideoVAE checkpoint (enables real VAE decode)")
    parser.add_argument("--vae_temporal_strides", type=int, nargs="+", default=[1, 1, 1, 1],
                        help="VAE temporal strides (default: 1 1 1 1 = no temporal compression)")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[Device] {device}, dtype={dtype}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load T5
    print(f"\n[T5] Loading text encoder...")
    text_encoder = load_real_t5(device, args.t5_model)

    # Build VideoVAE (real) or FakeVAE (bilinear fallback)
    if args.vae_checkpoint:
        print(f"\n[VAE] Loading VideoVAE from {args.vae_checkpoint}")
        vae_video = build_video_vae(
            latent_channels=16,
            temporal_strides=tuple(args.vae_temporal_strides),
            pretrained_path=args.vae_checkpoint,
            device=device,
            dtype=dtype,
        )
    else:
        print("\n[VAE] No checkpoint provided, using bilinear fallback (FakeVAE)")
        vae_video = FakeVAE(latent_channels=16)

    vae_audio = FakeVAE(latent_channels=8)

    # Create pipeline
    pipeline = T2VAPipeline(
        vae_video=vae_video,
        vae_audio=vae_audio,
        db_dit=model,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    # ── Run inference ──
    prompts = [
        args.prompt,
        "A close-up portrait of a woman with natural makeup, soft lighting",
        "A person walking in a city street, sunny day",
    ]

    for i, prompt in enumerate(prompts):
        seed = args.seed + i
        output_path = args.output.replace(".mp4", f"_{i}.mp4")

        print(f"\n{'='*60}")
        print(f"[Generate {i+1}/{len(prompts)}] seed={seed}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: {output_path}")

        try:
            pipeline.generate_to_file(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_frames=args.num_frames,
                width=args.width,
                height=args.height,
                fps=args.fps,
                num_steps=args.num_steps,
                cfg_video=args.cfg_video,
                seed=seed,
                output_path=output_path,
            )
            sz = os.path.getsize(output_path) / 1024
            print(f"  SUCCESS: {sz:.0f} KB")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Outputs in outputs/")
    print(f"  View: ffplay {args.output.replace('.mp4', '_0.mp4')}")


if __name__ == "__main__":
    main()
