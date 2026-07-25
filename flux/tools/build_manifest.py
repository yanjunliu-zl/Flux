#!/usr/bin/env python3
"""Build training CSV manifest from processed data.

Usage:
    python -m flux.tools.build_manifest --video_dir data/filtered/ \
        --audio_dir data/audio/ --captions data/captioned/captions.json \
        --output data/manifests/train.csv
"""

import argparse
import os
import csv
import json
import cv2
from pathlib import Path
import random


def build_manifest(
    video_dir: str,
    audio_dir: str | None,
    captions_file: str | None,
    output_path: str,
    train_split: float = 0.9,
    seed: int = 42,
):
    """Build CSV manifest for training.

    Args:
        video_dir: Directory with filtered videos.
        audio_dir: Directory with audio files (optional).
        captions_file: JSON file with captions (optional).
        output_path: Output CSV path.
        train_split: Fraction for training set.
        seed: Random seed.
    """
    random.seed(seed)

    # Load captions
    captions = {}
    if captions_file and os.path.exists(captions_file):
        with open(captions_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

    video_path = Path(video_dir)
    video_files = list(video_path.glob("*.mp4")) + list(video_path.glob("*/*.mp4"))

    # Gather video metadata
    rows = []
    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dur = frames / max(fps, 1)
        cap.release()

        # Audio path
        audio_path = ""
        if audio_dir:
            af = Path(audio_dir) / f"{vf.stem}.wav"
            if af.exists():
                audio_path = str(af)

        # Captions
        cap_data = captions.get(vf.name, {})
        caption_short = cap_data.get("caption_short", "")
        caption_long = cap_data.get("caption_long", "")
        caption_audio = cap_data.get("caption_audio", "")

        rows.append({
            "video_path": str(vf),
            "num_frames": frames,
            "height": h,
            "width": w,
            "fps": round(fps, 2),
            "duration_s": round(dur, 2),
            "audio_path": audio_path,
            "caption_short": caption_short,
            "caption_long": caption_long,
            "caption_audio": caption_audio,
        })

    # Shuffle and split
    random.shuffle(rows)
    split_idx = int(len(rows) * train_split)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    # Write CSV files
    fieldnames = [
        "video_path", "num_frames", "height", "width", "fps", "duration_s",
        "audio_path", "caption_short", "caption_long", "caption_audio",
    ]

    for split, data in [("train", train_rows), ("val", val_rows)]:
        split_path = output_path.replace(".csv", f"_{split}.csv") if "train" not in output_path else (
            output_path.replace("train", split) if split == "val" else output_path
        )
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"  {split}: {len(data)} samples → {split_path}")

    print(f"Manifest created: {len(train_rows)} train + {len(val_rows)} val = {len(rows)} total")


def main():
    parser = argparse.ArgumentParser(description="Build training manifest")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--audio_dir", type=str, default=None, help="Audio directory")
    parser.add_argument("--captions", type=str, default=None, help="Captions JSON file")
    parser.add_argument("--output", type=str, default="data/manifests/train.csv")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_manifest(
        args.video_dir, args.audio_dir, args.captions,
        args.output, args.train_split, args.seed,
    )


if __name__ == "__main__":
    main()
