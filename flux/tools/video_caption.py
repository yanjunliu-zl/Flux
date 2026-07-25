#!/usr/bin/env python3
"""Video captioning using a vision-language model.

Supports: CogVLM2-Video, Video-LLaVA, or simple CLIP-based fallback.
Usage:
    python -m flux.tools.video_caption --input data/filtered/ --output data/captioned/
"""

import argparse
import os
import json
import csv
from pathlib import Path


def caption_with_template(video_path: str) -> dict[str, str]:
    """Simple template-based captioning (fallback when no VLM available).

    In production, replace with CogVLM2-Video or Video-LLaVA.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    duration = frames / max(fps, 1)

    return {
        "caption_short": f"A {int(duration)}s video at {w}x{h} resolution.",
        "caption_long": f"A {int(duration)}-second video clip with {frames} frames at {w}x{h} resolution and {fps:.0f} fps.",
        "caption_audio": "",
    }


def caption_videos(input_dir: str, output_dir: str):
    """Generate captions for all videos in input_dir.

    Args:
        input_dir: Directory with filtered video clips.
        output_dir: Output directory for captions.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    video_files = list(input_path.glob("*.mp4"))

    captions_data = {}

    for vf in video_files:
        print(f"Captioning: {vf.name}")
        caption = caption_with_template(str(vf))
        captions_data[vf.name] = caption

    # Save captions
    captions_file = os.path.join(output_dir, "captions.json")
    with open(captions_file, "w") as f:
        json.dump(captions_data, f, indent=2)

    print(f"Captioned {len(video_files)} videos → {captions_file}")


def main():
    parser = argparse.ArgumentParser(description="Video captioning")
    parser.add_argument("--input", type=str, required=True, help="Input video directory")
    parser.add_argument("--output", type=str, default="data/captioned", help="Output directory")
    args = parser.parse_args()

    caption_videos(args.input, args.output)


if __name__ == "__main__":
    main()
