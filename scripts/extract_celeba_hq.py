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
WORK_SIZE = 340    # Larger working area for realistic crop (33% bigger)


def _smooth_path(rng: np.random.RandomState, n: int, amplitude: float) -> np.ndarray:
    """Generate a smooth random path using summed sine waves (like camera shake).

    Returns values in [-amplitude, +amplitude] that vary smoothly over n steps.
    """
    t = np.linspace(0, 2 * np.pi, n)
    path = np.zeros(n)
    # Sum 3-5 sine waves with random freq and phase for natural-looking motion
    for _ in range(rng.randint(3, 6)):
        freq = rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        path += amp * np.sin(freq * t + phase)
    # Normalize to [-amplitude, +amplitude]
    path -= path.min()
    path /= max(path.max(), 1e-8)
    path = (path - 0.5) * 2 * amplitude
    return path


def extract_image(row: dict, idx: int, output_dir: Path) -> dict | None:
    """Decode JPEG, create pseudo-video with realistic camera motion.

    Simulates natural camera movement:
    - Breathing zoom: slow 0.92-1.08x sinusoidal zoom (like handheld)
    - Camera sway: smooth random pan up to 12% shift in both axes
    - Micro-rotation: ±2° with natural drift
    - Exposure flutter: subtle brightness variation
    - Random per-video motion parameters for diversity

    This teaches the model that real videos have continuous frame-to-frame
    changes, preventing the "static image" collapse.
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

    # Resize to larger working area so we have room to pan/zoom
    img_resized = cv2.resize(img, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_LANCZOS4)

    rng = np.random.RandomState(idx)

    # ── Per-video motion parameters (randomize for diversity) ──
    # Zoom range: mimics breathing or slight lean-in/out
    zoom_min = rng.uniform(0.90, 0.95)
    zoom_max = rng.uniform(1.05, 1.12)
    zoom_speed = rng.uniform(0.5, 1.5)  # oscillation speed

    # Pan amplitude: percentage of WORK_SIZE (e.g., 0.10 = up to 10% shift)
    pan_amplitude = rng.uniform(0.06, 0.12) * WORK_SIZE

    # Brightness variation amplitude
    brightness_amp = rng.uniform(0.02, 0.06)

    # Rotation amplitude in degrees
    rotation_amp = rng.uniform(1.0, 3.0)

    # Generate smooth camera paths
    pan_x = _smooth_path(rng, NUM_FRAMES, pan_amplitude)
    pan_y = _smooth_path(rng, NUM_FRAMES, pan_amplitude)
    rot_path = _smooth_path(rng, NUM_FRAMES, rotation_amp)
    bright_path = _smooth_path(rng, NUM_FRAMES, brightness_amp)

    crop_base = int(WORK_SIZE * 0.75)  # base crop = 75% of work area
    max_crop = int(WORK_SIZE * 0.92)   # max when zoomed out
    min_crop = int(WORK_SIZE * 0.58)   # min when zoomed in

    frames = []

    for f in range(NUM_FRAMES):
        t = f / max(NUM_FRAMES - 1, 1)

        # Zoom: sinusoidal (breathe in/out)
        zoom = zoom_min + (zoom_max - zoom_min) * (0.5 + 0.5 * np.sin(t * np.pi * 2 * zoom_speed))

        # Crop size changes with zoom
        crop_size = int(crop_base / zoom)
        crop_size = np.clip(crop_size, min_crop, max_crop)

        # Center of crop: WORK_SIZE/2 + smooth pan
        cx = int(WORK_SIZE / 2 + pan_x[f])
        cy = int(WORK_SIZE / 2 + pan_y[f])

        # Build rotation matrix
        angle = rot_path[f]
        M = cv2.getRotationMatrix2D((crop_size / 2, crop_size / 2), angle, 1.0)

        # Extract crop (with rotation margin)
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(WORK_SIZE, x1 + crop_size)
        y2 = min(WORK_SIZE, y1 + crop_size)
        cropped = img_resized[y1:y2, x1:x2]

        # Pad if crop goes out of bounds
        if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
            cropped = padded

        # Apply rotation then resize to target
        rotated = cv2.warpAffine(cropped, M, (crop_size, crop_size),
                                 borderMode=cv2.BORDER_REFLECT)
        frame = cv2.resize(rotated, (OUTPUT_RES, OUTPUT_RES),
                          interpolation=cv2.INTER_LANCZOS4)

        # Brightness flutter
        brightness = 1.0 + bright_path[f]
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
