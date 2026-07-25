#!/usr/bin/env python3
"""WebVid-10M dataset downloader.

Downloads videos and captions from the WebVid-10M dataset.
Source: huggingface.co/datasets/TempoFunk/webvid-10m

The dataset provides:
  - 10M video URL + caption pairs
  - ~2.5M subset with downloadable videos (rest are dead links)
  - Both short (~20 word) and long (~100 word) captions

Usage:
    # Download metadata only (fast, ~200MB CSV)
    python -m flux.tools.download_webvid --metadata_only \
        --output data/webvid/

    # Download videos with captions (slow, needs network)
    python -m flux.tools.download_webvid --max_videos 5000 \
        --output data/webvid/ --workers 8

    # Stream metadata only (no download) — for checking stats
    python -m flux.tools.download_webvid --stream
"""

import argparse
import csv
import os
import shutil
import sys
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def _find_yt_dlp() -> str:
    """Find the yt-dlp executable in PATH or venv."""
    yt = shutil.which("yt-dlp")
    if yt:
        return yt
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    for p in Path(sys.executable).parent.parent.glob("**/yt-dlp"):
        return str(p)
    return "yt-dlp"


def download_metadata(output_dir: str, max_rows: Optional[int] = None) -> Path:
    """Download WebVid-10M metadata (video URLs + captions) from HuggingFace.

    Downloads the parquet files, converts to CSV, and saves locally.

    Args:
        output_dir: Output directory.
        max_rows: Max rows to save (None = all).

    Returns:
        Path to the metadata CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "webvid_metadata.csv"

    try:
        from datasets import load_dataset
        print("[WebVid] Loading metadata from HuggingFace...")
        ds = load_dataset("TempoFunk/webvid-10m", split="train", streaming=True)

        rows = []
        for i, row in enumerate(ds):
            if max_rows and i >= max_rows:
                break
            rows.append({
                "videoid": row["videoid"],
                "name": row.get("name", ""),
                "page_url": row.get("page_url", ""),
                "contentUrl": row["contentUrl"],
                "duration": row.get("duration", ""),
                "caption": row.get("caption", row.get("name", "")),
            })
            if i % 1000 == 0 and i > 0:
                print(f"  Loaded {i} rows...")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"[WebVid] Saved {len(rows)} rows → {csv_path}")
        return csv_path

    except ImportError:
        print("[WebVid] datasets package not installed. Install: pip install datasets pyarrow")
        print("[WebVid] Falling back: download CSV from HuggingFace directly...")
        return _download_metadata_http(output_dir, max_rows)


def _download_metadata_http(output_dir: str, max_rows: Optional[int] = None) -> Path:
    """Fallback: download CSV release from HuggingFace."""
    import urllib.request
    csv_path = Path(output_dir) / "webvid_metadata.csv"

    # Try HuggingFace direct URL
    urls = [
        "https://huggingface.co/datasets/TempoFunk/webvid-10m/resolve/main/data/train-00000-of-00001.parquet",
    ]

    for url in urls:
        try:
            print(f"[WebVid] Downloading {url}")
            urllib.request.urlretrieve(url, str(csv_path.with_suffix(".parquet")))
            # Convert parquet to CSV
            try:
                import pandas as pd
                df = pd.read_parquet(csv_path.with_suffix(".parquet"))
                if max_rows:
                    df = df.head(max_rows)
                df.to_csv(csv_path, index=False)
                print(f"[WebVid] Converted {len(df)} rows → {csv_path}")
                return csv_path
            except ImportError:
                print("[WebVid] Need pandas for parquet→CSV: pip install pandas pyarrow")
                return csv_path.with_suffix(".parquet")
        except Exception as e:
            print(f"  Failed: {e}")

    return csv_path


def download_video(info: dict, output_dir: str, timeout: int = 60) -> Optional[dict]:
    """Download a single video using yt-dlp.

    Args:
        info: Dict with 'videoid', 'contentUrl', 'caption'.
        output_dir: Output directory.
        timeout: Download timeout in seconds.

    Returns:
        Dict with video metadata if successful, None if failed.
    """
    videoid = info["videoid"]
    url = info["contentUrl"]
    caption = info.get("caption", "")
    output_template = os.path.join(output_dir, f"{videoid}.%(ext)s")

    yt_dlp = _find_yt_dlp()
    cmd = [
        yt_dlp,
        "-f", "best",  # WebVid videos are low-res direct links
        "-o", output_template,
        "--no-playlist",
        "--socket-timeout", str(timeout),
        "--retries", "2",
        "--quiet",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            return None

        # Find the downloaded file
        for ext in ["mp4", "webm", "mkv"]:
            video_path = Path(output_dir) / f"{videoid}.{ext}"
            if video_path.exists() and video_path.stat().st_size > 1024:
                return {
                    "videoid": videoid,
                    "video_path": str(video_path),
                    "caption": caption,
                    "url": url,
                }
        return None
    except subprocess.TimeoutExpired:
        return None


def download_dataset(
    metadata_csv: str,
    output_dir: str,
    max_videos: int = 5000,
    workers: int = 8,
    resume: bool = True,
):
    """Download WebVid videos from metadata CSV.

    Args:
        metadata_csv: Path to metadata CSV.
        output_dir: Output directory for videos.
        max_videos: Maximum number of videos to download.
        workers: Number of parallel download workers.
        resume: If True, skip already-downloaded videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter and shuffle
    import random
    random.shuffle(rows)

    if resume:
        existing = set(f.stem for f in Path(output_dir).glob("*"))
        rows = [r for r in rows if r["videoid"] not in existing]
        print(f"[WebVid] Resuming: {len(existing)} already downloaded")

    rows = rows[:max_videos]
    print(f"[WebVid] Downloading {len(rows)} videos with {workers} workers...")

    # Track progress
    succeeded = []
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_video, row, output_dir): row
            for row in rows
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                succeeded.append(result)
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1)
                print(f"  [{i+1}/{len(rows)}] {len(succeeded)} ok, {failed} failed | {rate:.1f} vids/s")

    # Save manifest
    manifest_path = os.path.join(output_dir, "downloaded.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        if succeeded:
            writer = csv.DictWriter(f, fieldnames=["videoid", "video_path", "caption", "url"])
            writer.writeheader()
            writer.writerows(succeeded)

    elapsed = time.time() - start_time
    print(f"\n[WebVid] Done in {elapsed:.0f}s: {len(succeeded)} downloaded, {failed} failed")
    print(f"[WebVid] Manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Download WebVid-10M dataset")
    parser.add_argument("--output", type=str, default="data/webvid", help="Output directory")
    parser.add_argument("--max_videos", type=int, default=5000, help="Max videos to download")
    parser.add_argument("--max_metadata", type=int, default=None, help="Max metadata rows")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    parser.add_argument("--metadata_only", action="store_true", help="Only download metadata CSV")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Use existing metadata CSV")
    parser.add_argument("--stream", action="store_true", help="Stream metadata (read-only)")
    parser.add_argument("--no_resume", action="store_true", help="Don't skip existing files")
    args = parser.parse_args()

    if args.stream:
        ds = None
        try:
            from datasets import load_dataset
            ds = load_dataset("TempoFunk/webvid-10m", split="train", streaming=True)
        except ImportError:
            print("Install: pip install datasets pyarrow")
            return
        for i, row in enumerate(ds):
            if i >= 5:
                break
            print(f"  [{row['videoid']}] {row.get('caption', 'N/A')[:120]}")
        print(f"  ... (use --max_metadata N to load more)")
        return

    # Step 1: Get metadata
    if args.metadata_csv and os.path.exists(args.metadata_csv):
        metadata_csv = args.metadata_csv
        print(f"[WebVid] Using existing metadata: {metadata_csv}")
    else:
        meta_dir = os.path.join(args.output, "metadata")
        metadata_csv = str(download_metadata(meta_dir, args.max_metadata))

    if args.metadata_only:
        print(f"[WebVid] Metadata saved to: {metadata_csv}")
        return

    # Step 2: Download videos
    video_dir = os.path.join(args.output, "videos")
    manifest = download_dataset(
        metadata_csv=str(metadata_csv),
        output_dir=video_dir,
        max_videos=args.max_videos,
        workers=args.workers,
        resume=not args.no_resume,
    )

    print(f"\nNext step: quality filter the downloaded videos")
    print(f"  python -m flux.tools.quality_filter --input {video_dir} --output data/filtered/")


if __name__ == "__main__":
    main()
