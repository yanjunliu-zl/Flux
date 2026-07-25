#!/usr/bin/env python3
"""Extract CelebA-HQ parquet to training-ready format.

Creates pseudo-videos from static images by repeating frames with subtle
augmentation (random crop jitter + slight zoom), so the model learns
both face structure AND some motion tolerance.

Usage:
    python scripts/extract_celeba_hq.py [--output data/celeba_hq_videos]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Number of frames per pseudo-video (matched to training config: num_frames=32)
NUM_FRAMES = 32
FPS = 16
OUTPUT_RES = 256  # Match training resolution


def extract_image(row: dict, idx: int, output_dir: Path) -> dict | None:
    """Decode JPEG bytes, create pseudo-video, save as mp4.

    Applies subtle augmentation across frames:
    - Slight random crop jitter (up to 2% shift)
    - Slight zoom in/out (98%-102%)
    - Subtle brightness variation

    This teaches the model that faces have natural micro-movements
    while keeping the core identity intact.
    """
    # Decode JPEG
    img_bytes = row["image"]
    if isinstance(img_bytes, dict):
        img_bytes = img_bytes["bytes"]

    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]

    # Resize to slightly larger than target so we can crop-jitter
    base_size = int(OUTPUT_RES * 1.1)  # 10% padding for jitter
    img_resized = cv2.resize(img, (base_size, base_size), interpolation=cv2.INTER_LANCZOS4)

    frames = []
    rng = np.random.RandomState(idx)

    for f in range(NUM_FRAMES):
        # Subtle smooth zoom: 0.98 → 1.02 over the duration
        t = f / max(NUM_FRAMES - 1, 1)
        zoom = 1.0 + 0.02 * np.sin(t * np.pi)  # 0.98 to 1.02 sine wave

        # Crop jitter (up to 2% shift)
        crop_size = int(base_size * (0.98 + 0.02 * zoom))  # subtle crop size change
        max_shift = max(1, base_size - crop_size)
        dx = rng.randint(0, max_shift + 1)
        dy = rng.randint(0, max_shift + 1)

        cropped = img_resized[dy:dy + crop_size, dx:dx + crop_size]

        # Resize to target
        frame = cv2.resize(cropped, (OUTPUT_RES, OUTPUT_RES), interpolation=cv2.INTER_LANCZOS4)

        # Subtle brightness (±1%)
        brightness = 1.0 + 0.01 * np.sin(t * 2 * np.pi + rng.rand() * np.pi)
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)

        frames.append(frame)

    # Save as mp4
    video_path = output_dir / f"celeba_{idx:05d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (OUTPUT_RES, OUTPUT_RES))

    for frame in frames:
        writer.write(frame)
    writer.release()

    if not video_path.exists() or video_path.stat().st_size < 1024:
        return None

    return {
        "video_path": str(video_path.relative_to(PROJECT_DIR)),
        "num_frames": NUM_FRAMES,
        "height": OUTPUT_RES,
        "width": OUTPUT_RES,
        "fps": round(FPS, 2),
        "duration_s": round(NUM_FRAMES / FPS, 2),
        "audio_path": "",
        "caption_short": row["text"],
        "caption_long": row["text"],
        "caption_audio": "",
        "speaker_id": "",
        "dataset": "celeba_hq",
    }


def main():
    parser = argparse.ArgumentParser(description="Extract CelebA-HQ from parquet to videos")
    parser.add_argument("--input", default="data/celeba_hq/data")
    parser.add_argument("--output", default="data/celeba_hq_videos")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max images to process (0=all, for quick test)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all parquet files
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        sys.exit(1)

    print(f"Loading {len(parquet_files)} parquet files...")
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        dfs.append(df)
        print(f"  {pf.name}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df)} images")

    if args.max_images > 0:
        df = df.head(args.max_images)
        print(f"Limited to {args.max_images} (--max_images)")

    # Process
    print(f"\nExtracting {len(df)} images → {NUM_FRAMES}f pseudo-videos ({args.workers} workers)...")
    rows = []

    # Use ProcessPoolExecutor for parallel video encoding
    chunk_size = max(1, len(df) // args.workers)
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        for j, (_, row) in enumerate(chunk.iterrows()):
            idx = i + j
            result = extract_image(row, idx, output_dir)
            if result:
                rows.append(result)
            if (idx + 1) % 500 == 0:
                print(f"  [{idx + 1}/{len(df)}] {len(rows)} ok")

    print(f"\nExtracted: {len(rows)}/{len(df)} videos")

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * (1 - args.val_split))
    train_rows = rows[:split]
    val_rows = rows[split:]

    # Write manifest
    fieldnames = [
        "video_path", "num_frames", "height", "width", "fps", "duration_s",
        "audio_path", "caption_short", "caption_long", "caption_audio",
        "speaker_id", "dataset",
    ]

    for split_name, data in [("train", train_rows), ("val", val_rows)]:
        manifest_path = PROJECT_DIR / "data" / "manifests" / f"celeba_hq_{split_name}.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"  {split_name}: {len(data)} → {manifest_path}")

    # Summary
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.mp4")) / 1e9
    print(f"\nDone: {len(rows)} pseudo-videos, {total_size:.1f} GB")

    # Caption diversity
    unique_captions = len(set(r["caption_short"] for r in rows))
    print(f"Caption diversity: {unique_captions} unique / {len(rows)} total")


if __name__ == "__main__":
    main()
