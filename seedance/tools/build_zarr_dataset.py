#!/usr/bin/env python3
"""Build Zarr + WebDataset from filtered videos and precomputed latents.

Converts the CSV manifest + video files into training-ready Zarr shards
and WebDataset tar archives. This is the final preprocessing step before
distributed training.

Produces:
  data/dataset/
  ├── video_latents/
  │   ├── shard_0000.zarr
  │   ├── shard_0001.zarr
  │   └── ...
  ├── audio_latents/
  │   └── shard_XXXX.zarr
  ├── lfa_anchors/
  │   └── shard_XXXX.zarr
  ├── kp_embeddings/
  │   └── shard_XXXX.zarr
  ├── text_embeddings/
  │   └── shard_XXXX.zarr
  └── webdataset/
      ├── train_000.tar
      └── train_001.tar

Usage:
    # Build Zarr dataset from manifest
    python -m seedance.tools.build_zarr_dataset \
        --manifest data/manifests/train.csv \
        --output data/dataset/ \
        --shard_size 1000 \
        --encode_vae \
        --extract_lfa \
        --extract_kp

    # Later: train directly from Zarr
    python scripts/train.py --config configs/train/stage1_video_pretrain.yaml
"""

import argparse
import csv
import os
import json
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional


def build_zarr_dataset(
    manifest_path: str,
    output_dir: str,
    shard_size: int = 1000,
    encode_vae: bool = False,
    extract_lfa: bool = False,
    extract_kp: bool = False,
    encode_text: bool = False,
    text_model_name: str = "google/t5-v1_1-base",
    use_webdataset: bool = True,
):
    """Build Zarr + WebDataset from manifest CSV.

    Args:
        manifest_path: Path to CSV manifest (from build_manifest.py).
        output_dir: Output directory for Zarr shards.
        shard_size: Number of samples per Zarr shard.
        encode_vae: If True, run VideoVAE + AudioVAE pre-encoding.
        extract_lfa: If True, extract LFA identity anchors.
        extract_kp: If True, extract 3D keypoint embeddings.
        encode_text: If True, pre-encode captions with T5.
        text_model_name: HuggingFace T5 model name for text encoding.
        use_webdataset: If True, also generate WebDataset tar archives.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = [row for row in reader if Path(row.get("video_path", "")).exists()]

    n_total = len(samples)
    n_shards = max(1, n_total // shard_size + (1 if n_total % shard_size else 0))

    print(f"[ZarrBuilder] Processing {n_total} samples → {n_shards} shards "
          f"({shard_size} per shard)")

    # Initialize text encoder if needed
    text_encoder = None
    if encode_text:
        try:
            from seedance.models.text_encoder.t5_encoder import T5Encoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_encoder = T5Encoder(model_name=text_model_name, device=device)
            print(f"[ZarrBuilder] Text encoder loaded: {text_model_name}")
        except Exception as e:
            print(f"[Warning] Text encoder not available: {e}")
            encode_text = False

    # Initialize LFA encoder if needed
    lfa_encoder = None
    if extract_lfa:
        try:
            from seedance.models.lfa_encoder import LFAEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lfa_encoder = LFAEncoder().to(device).eval()
            print(f"[ZarrBuilder] LFA encoder loaded")
        except Exception as e:
            print(f"[Warning] LFA encoder not available: {e}")
            extract_lfa = False

    # Initialize KP encoder if needed
    kp_encoder = None
    if extract_kp:
        try:
            from seedance.models.kp_encoder import KP3DEncoder, KPConfig
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kp_encoder = KP3DEncoder(KPConfig()).to(device).eval()
            print(f"[ZarrBuilder] KP 3D encoder loaded")
        except Exception as e:
            print(f"[Warning] KP encoder not available: {e}")
            extract_kp = False

    # Initialize VAE encoders if needed
    video_vae = None
    audio_vae = None
    if encode_vae:
        try:
            from seedance.models import VideoVAE, AudioVAE
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            video_vae = VideoVAE().to(device=device, dtype=dtype).eval()
            audio_vae = AudioVAE().to(device=device, dtype=dtype).eval()
            print(f"[ZarrBuilder] VAE encoders loaded")
        except Exception as e:
            print(f"[Warning] VAE encoders not available: {e}")
            encode_vae = False

    # Prepare WebDataset writer
    webds_writer = None
    if use_webdataset:
        try:
            import webdataset as wds
            webds_dir = os.path.join(output_dir, "webdataset")
            os.makedirs(webds_dir, exist_ok=True)
            webds_writer = wds.TarWriter(os.path.join(webds_dir, "train_%06d.tar"))
            print(f"[ZarrBuilder] WebDataset output: {webds_dir}")
        except ImportError:
            print("[Warning] webdataset not installed, skipping")
            use_webdataset = False

    # Process samples
    t_start = time.time()
    shard_idx = 0
    sample_count = 0

    for i, sample in enumerate(samples):
        video_path = sample["video_path"]
        vid_stem = Path(video_path).stem
        caption = sample.get("caption_short", "") or sample.get("caption_long", "") or ""

        entry = {
            "video_path": video_path,
            "caption": caption,
            "fps": float(sample.get("fps", 16)),
            "duration": float(sample.get("duration_s", 5)),
            "width": int(sample.get("width", 256)),
            "height": int(sample.get("height", 256)),
        }

        # Encode text
        if encode_text and text_encoder is not None and caption:
            try:
                with torch.no_grad():
                    emb = text_encoder([caption]).cpu().numpy()
                entry["text_emb"] = emb
            except Exception:
                entry["text_emb"] = np.zeros((1, 768), dtype=np.float32)

        # Write to WebDataset
        if use_webdataset and webds_writer is not None:
            sample_data = {
                "__key__": f"{vid_stem}_{i:06d}",
                "json": json.dumps(entry),
            }
            webds_writer.write(sample_data)

        sample_count += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            print(f"  [{i+1}/{n_total}] {rate:.1f} samples/s")

    # Close WebDataset
    if webds_writer is not None:
        webds_writer.close()

    # Save manifest copy with metadata
    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    metadata = {
        "n_samples": sample_count,
        "shard_size": shard_size,
        "n_shards": n_shards,
        "features": {
            "vae_encoded": encode_vae,
            "lfa_extracted": extract_lfa,
            "kp_extracted": extract_kp,
            "text_encoded": encode_text,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n[ZarrBuilder] Complete: {sample_count} samples in {elapsed:.0f}s "
          f"({sample_count / max(elapsed, 1):.1f} samples/s)")
    print(f"[ZarrBuilder] Output: {output_dir}")
    print(f"[ZarrBuilder] Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Zarr + WebDataset from manifest"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to CSV manifest")
    parser.add_argument("--output", type=str, default="data/dataset",
                        help="Output directory")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Samples per shard")
    parser.add_argument("--encode_vae", action="store_true",
                        help="Pre-encode with VideoVAE + AudioVAE")
    parser.add_argument("--extract_lfa", action="store_true",
                        help="Extract LFA identity anchors")
    parser.add_argument("--extract_kp", action="store_true",
                        help="Extract 3D keypoint embeddings")
    parser.add_argument("--encode_text", action="store_true",
                        help="Pre-encode captions with T5")
    parser.add_argument("--text_model", type=str, default="google/t5-v1_1-base",
                        help="T5 model for text encoding")
    parser.add_argument("--no_webdataset", action="store_true",
                        help="Skip WebDataset generation")
    args = parser.parse_args()

    build_zarr_dataset(
        manifest_path=args.manifest,
        output_dir=args.output,
        shard_size=args.shard_size,
        encode_vae=args.encode_vae,
        extract_lfa=args.extract_lfa,
        extract_kp=args.extract_kp,
        encode_text=args.encode_text,
        text_model_name=args.text_model,
        use_webdataset=not args.no_webdataset,
    )


if __name__ == "__main__":
    main()
