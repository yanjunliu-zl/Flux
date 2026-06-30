#!/usr/bin/env python3
"""End-to-end data preparation pipeline for Seedance 2.0.

Orchestrates the full data preprocessing workflow:
  1. Download videos (WebVid, Pexels, or custom URLs)
  2. Scene detection → clip segmentation
  3. Quality filtering → motion/aesthetic/resolution
  4. Audio extraction → WAV files
  5. AV sync scoring
  6. Build training manifest → CSV

Usage:
    # Full pipeline from WebVid
    python -m seedance.tools.pipeline --source webvid --max_videos 5000

    # Full pipeline from Pexels
    export PEXELS_API_KEY="your-key"
    python -m seedance.tools.pipeline --source pexels --query "nature city" --max_videos 1000

    # Process existing video directory
    python -m seedance.tools.pipeline --source local --video_dir /path/to/videos
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def run_step(description: str, cmd: list[str], cwd: str | None = None):
    """Run a pipeline step, printing progress."""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, "-m"] + cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[Pipeline] ❌ Step failed: {description}")
        return False
    print(f"[Pipeline] ✓ {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Seedance 2.0 full data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # WebVid
  python -m seedance.tools.pipeline --source webvid --max_videos 5000

  # Pexels
  export PEXELS_API_KEY=your-key
  python -m seedance.tools.pipeline --source pexels --max_videos 2000

  # Local videos (skip download)
  python -m seedance.tools.pipeline --source local --video_dir /data/videos/
""",
    )

    # Source selection
    parser.add_argument("--source", type=str, default="webvid",
                        choices=["webvid", "pexels", "pixabay", "local"],
                        help="Data source")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Existing video directory (for --source local)")

    # Download options
    parser.add_argument("--max_videos", type=int, default=5000,
                        help="Maximum videos to download/process")
    parser.add_argument("--query", type=str, default="nature landscape ocean city people animals",
                        help="Search query for stock sites")
    parser.add_argument("--workers", type=int, default=8,
                        help="Download workers")

    # Quality filter options
    parser.add_argument("--min_height", type=int, default=360)
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--min_flow", type=float, default=0.05)

    # Output
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_filter", action="store_true")
    parser.add_argument("--skip_audio", action="store_true")
    parser.add_argument("--skip_caption", action="store_true")
    parser.add_argument("--skip_sync", action="store_true")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True)

    # Define paths
    raw_dir = data_root / "raw_videos"
    clips_dir = data_root / "clips"
    filtered_dir = data_root / "filtered"
    audio_dir = data_root / "audio"
    captions_dir = data_root / "captions"
    manifest_dir = data_root / "manifests"

    # Step 1: Download
    if not args.skip_download:
        if args.source == "webvid":
            run_step(
                "Download WebVid-10M metadata + videos",
                [
                    "seedance.tools.download_webvid",
                    f"--max_videos={args.max_videos}",
                    f"--workers={args.workers}",
                    f"--output={raw_dir}",
                ],
            )
        elif args.source in ("pexels", "pixabay"):
            run_step(
                f"Search + download {args.source} videos",
                [
                    "seedance.tools.download_pexels",
                    f"--source={args.source}",
                    f"--query={args.query}",
                    f"--max_videos={args.max_videos}",
                    f"--workers={args.workers}",
                    f"--output={raw_dir}",
                ],
            )
        elif args.source == "local":
            if not args.video_dir or not Path(args.video_dir).exists():
                print(f"[Pipeline] ❌ --video_dir required for --source local")
                return
            raw_dir = Path(args.video_dir)
            print(f"\n[Pipeline] Using existing videos from {raw_dir}")
    else:
        print(f"\n[Pipeline] Skipping download, expecting videos in {raw_dir}")

    # Find the actual video directory (may be a subdir like "videos/")
    if (raw_dir / "videos").exists():
        video_input_dir = str(raw_dir / "videos")
    else:
        video_input_dir = str(raw_dir)

    # Count videos
    n_videos = len(list(Path(video_input_dir).glob("*.mp4"))) if Path(video_input_dir).exists() else 0
    if n_videos == 0:
        print(f"[Pipeline] ❌ No videos found in {video_input_dir}")
        return
    print(f"\n[Pipeline] Found {n_videos} videos to process")

    # Step 2: Scene detection
    run_step(
        "Scene detection → clip segmentation",
        [
            "seedance.tools.scene_detection",
            f"--input={video_input_dir}",
            f"--output={clips_dir}",
            f"--min_duration={args.min_duration}",
            f"--max_duration={args.max_duration}",
        ],
    )

    # Step 3: Quality filtering
    if not args.skip_filter:
        run_step(
            "Quality filtering",
            [
                "seedance.tools.quality_filter",
                f"--input={clips_dir}",
                f"--output={filtered_dir}",
                f"--min_height={args.min_height}",
                f"--min_duration={args.min_duration}",
                f"--max_duration={args.max_duration}",
                f"--min_flow={args.min_flow}",
            ],
        )
    else:
        filtered_dir = clips_dir

    # Step 4: Audio extraction
    if not args.skip_audio:
        run_step(
            "Audio extraction (FFmpeg)",
            [
                "seedance.tools.audio_extract",
                f"--input={filtered_dir}",
                f"--output={audio_dir}",
            ],
        )

    # Step 5: Caption generation
    if not args.skip_caption:
        run_step(
            "Video captioning",
            [
                "seedance.tools.video_caption",
                f"--input={filtered_dir}",
                f"--output={captions_dir}",
            ],
        )

    # Step 6: AV sync filtering
    if not args.skip_sync and (audio_dir / "..").exists():
        run_step(
            "AV sync scoring",
            [
                "seedance.tools.av_sync_filter",
                f"--video_dir={filtered_dir}",
                f"--audio_dir={audio_dir}",
                f"--output={manifest_dir}",
            ],
        )

    # Step 7: Build manifest
    manifest_dir.mkdir(exist_ok=True)
    captions_file = captions_dir / "captions.json"
    run_step(
        "Build training manifest",
        [
            "seedance.tools.build_manifest",
            f"--video_dir={filtered_dir}",
            f"--audio_dir={audio_dir}",
            f"--captions={captions_file}" if captions_file.exists() else "",
            f"--output={manifest_dir / 'train.csv'}",
        ],
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"{'='*60}")
    print(f"  Raw:      {raw_dir}")
    print(f"  Clips:    {clips_dir}")
    print(f"  Filtered: {filtered_dir}")
    print(f"  Audio:    {audio_dir}")
    print(f"  Manifest: {manifest_dir}")
    print(f"\nNext: Train!")
    print(f"  python scripts/train.py --config configs/train/stage1_video_pretrain.yaml")


if __name__ == "__main__":
    main()
