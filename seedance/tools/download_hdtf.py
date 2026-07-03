#!/usr/bin/env python3
"""HDTF dataset downloader — YouTube → talking face clips.

Downloads the HDTF (High-resolution Talking Face) dataset from YouTube
using the official metadata, then splits videos into talking-face clips
based on provided timestamps.

Dataset: CVPR 2021, ~370 YouTube videos → ~369 talking face clips
         Each clip is 720p/1080p with native audio, 512x512 face crop

Usage:
    # Step 1: Download metadata (from GitHub)
    python -m seedance.tools.download_hdtf --download_metadata --output data/hdtf/

    # Step 2: Download videos + extract clips
    python -m seedance.tools.download_hdtf --max_videos 50 --output data/hdtf/ --workers 4

    # Step 3: Crop faces + generate manifest
    python -m seedance.tools.ingest_talking_data --input_dir data/hdtf/clips/ --dataset hdtf --output data/manifests/hdtf_manifest.csv
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


def _find_yt_dlp() -> str:
    yt = shutil.which("yt-dlp")
    if yt:
        return yt
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    return "yt-dlp"


def download_metadata(output_dir: str):
    """Download HDTF metadata from official GitHub repo."""
    import urllib.request

    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset"
    subsets = ["RD", "WDA", "WRA"]
    file_types = ["video_url", "annotion_time", "crop_wh", "crop_ratio", "resolution"]

    for subset in subsets:
        for ftype in file_types:
            fname = f"{subset}_{ftype}.txt"
            url = f"{base_url}/{fname}"
            local = os.path.join(meta_dir, fname)
            try:
                urllib.request.urlretrieve(url, local)
                print(f"  {fname}")
            except Exception as e:
                print(f"  FAILED {fname}: {e}")

    print(f"\nMetadata saved to {meta_dir}")


def parse_metadata(meta_dir: str) -> list[dict]:
    """Parse HDTF metadata into a list of clip descriptors.

    Returns:
        List of dicts with keys:
          video_name, youtube_url, subset, clip_index,
          start_time, end_time, crop_x, crop_y, crop_w, crop_h, crop_ratio.
    """
    clips = []

    for subset in ["RD", "WDA", "WRA"]:
        # Parse video URLs
        url_file = os.path.join(meta_dir, f"{subset}_video_url.txt")
        if not os.path.exists(url_file):
            continue

        video_urls = {}
        with open(url_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lstrip("﻿")
                if not line:
                    continue
                # Format: video_name youtube_url (space-separated)
                parts = line.split()
                if len(parts) >= 2 and "youtube" in parts[1].lower():
                    video_urls[parts[0]] = parts[1]

        # Parse annotation times
        time_file = os.path.join(meta_dir, f"{subset}_annotion_time.txt")
        if not os.path.exists(time_file):
            continue

        with open(time_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lstrip("﻿")
                if not line:
                    continue
                # Format: video_name.mp4 HH:MM:SS-HH:MM:SS HH:MM:SS-HH:MM:SS ...
                # (space-separated, first token is video filename)
                parts = line.split()
                video_name = parts[0].replace(".mp4", "")  # Strip .mp4
                time_ranges = parts[1:]

                for i, tr in enumerate(time_ranges):
                    tr = tr.strip()
                    # Support both MM:SS-MM:SS and HH:MM:SS-HH:MM:SS formats
                    match = re.match(
                        r"(\d+):(\d+)(?::(\d+))?\s*-\s*(\d+):(\d+)(?::(\d+))?", tr
                    )
                    if match:
                        g = match.groups()
                        if g[2] is not None:  # HH:MM:SS format
                            start_sec = int(g[0])*3600 + int(g[1])*60 + int(g[2])
                        else:  # MM:SS format
                            start_sec = int(g[0])*60 + int(g[1])
                        if g[5] is not None:  # HH:MM:SS format
                            end_sec = int(g[3])*3600 + int(g[4])*60 + int(g[5])
                        else:  # MM:SS format
                            end_sec = int(g[3])*60 + int(g[4])

                        clip = {
                            "video_name": video_name,
                            "youtube_url": video_urls.get(video_name, ""),
                            "subset": subset,
                            "clip_index": i,
                            "start_time": f"{start_sec//3600:02d}:{(start_sec%3600)//60:02d}:{start_sec%60:02d}",
                            "end_time": f"{end_sec//3600:02d}:{(end_sec%3600)//60:02d}:{end_sec%60:02d}",
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                        }
                        clips.append(clip)

    return clips


def download_clip(clip: dict, output_dir: str, timeout: int = 120) -> Optional[dict]:
    """Download a single HDTF clip from YouTube.

    Strategy: download full video, then cut segment with ffmpeg.
    Full videos are cached and shared across clips from the same source.

    Args:
        clip: Clip descriptor from parse_metadata().
        output_dir: Output directory.
        timeout: Download timeout.

    Returns:
        Updated clip dict with 'video_path' if successful, None if failed.
    """
    url = clip["youtube_url"]
    if not url:
        return None

    video_name = clip["video_name"]
    clip_idx = clip["clip_index"]
    output_name = f"{video_name}_{clip_idx}"
    output_path = Path(output_dir) / f"{output_name}.mp4"

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 4096:
        clip["video_path"] = str(output_path)
        return clip

    yt_dlp = _find_yt_dlp()
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    # Download full video (cached by video_name)
    cache_dir = Path(output_dir) / ".cache"
    cache_dir.mkdir(exist_ok=True)
    full_video = cache_dir / f"{video_name}.mp4"

    if not full_video.exists() or full_video.stat().st_size < 4096:
        cache_template = str(cache_dir / f"{video_name}.%(ext)s")
        cmd = [
            yt_dlp,
            "-f", "best[height<=720]",
            "-o", cache_template,
            "--no-playlist",
            "--socket-timeout", str(timeout),
            "--retries", "2",
            "--quiet",
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout + 60)
            if result.returncode != 0:
                return None
            # Rename to .mp4 if yt-dlp used a different extension
            for ext in ["mp4", "webm", "mkv"]:
                cached = cache_dir / f"{video_name}.{ext}"
                if cached.exists() and cached.stat().st_size > 4096:
                    if ext != "mp4":
                        cached.rename(full_video)
                    break
        except subprocess.TimeoutExpired:
            return None

    if not full_video.exists() or full_video.stat().st_size < 4096:
        return None

    # Cut segment with ffmpeg
    start = clip["start_sec"]
    duration = clip["end_sec"] - clip["start_sec"]
    cut_cmd = [
        ffmpeg, "-y", "-ss", str(start), "-i", str(full_video),
        "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    try:
        result = subprocess.run(cut_cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            # Retry with re-encode (some videos need it)
            cut_cmd2 = [
                ffmpeg, "-y", "-ss", str(start), "-i", str(full_video),
                "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
                str(output_path),
            ]
            result = subprocess.run(cut_cmd2, capture_output=True, timeout=60)
            if result.returncode != 0:
                return None
    except subprocess.TimeoutExpired:
        return None

    if output_path.exists() and output_path.stat().st_size > 4096:
        clip["video_path"] = str(output_path)
        return clip
    return None


def main():
    parser = argparse.ArgumentParser(description="Download HDTF talking-face dataset")
    parser.add_argument("--output", type=str, default="data/hdtf",
                        help="Output directory")
    parser.add_argument("--max_videos", type=int, default=50,
                        help="Max YouTube videos to download")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--download_metadata", action="store_true",
                        help="Only download metadata from GitHub")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Download metadata
    meta_dir = os.path.join(args.output, "metadata")
    if not os.path.exists(meta_dir) or not os.listdir(meta_dir):
        print("[HDTF] Downloading metadata...")
        download_metadata(args.output)

    if args.download_metadata:
        return

    # Parse metadata
    clips = parse_metadata(meta_dir)
    print(f"[HDTF] Parsed {len(clips)} clip annotations from metadata")

    # Download clips
    clips = clips[:args.max_videos]
    clips_dir = os.path.join(args.output, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    print(f"[HDTF] Downloading {len(clips)} clips with {args.workers} workers...")
    print(f"[HDTF] This may take a while — YouTube rate limits apply.")
    print()

    succeeded = []
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_clip, c, clips_dir): c
            for c in clips
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                succeeded.append(result)
            else:
                failed += 1
            if (i + 1) % 10 == 0 or i == len(clips) - 1:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(clips)}] {len(succeeded)} ok, {failed} failed "
                      f"({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\n[HDTF] Done in {elapsed:.0f}s: {len(succeeded)}/{len(clips)} downloaded")

    if succeeded:
        print(f"\nNext steps:")
        print(f"  1. Ingest into manifest:")
        print(f"     python -m seedance.tools.ingest_talking_data \\")
        print(f"       --input_dir {clips_dir} --dataset hdtf \\")
        print(f"       --output data/manifests/hdtf_manifest.csv")
        print(f"  2. Run quality filter:")
        print(f"     python -m seedance.tools.quality_filter \\")
        print(f"       --input {clips_dir} --output data/filtered/hdtf/")


if __name__ == "__main__":
    main()
