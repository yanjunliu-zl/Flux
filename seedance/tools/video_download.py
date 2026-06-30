#!/usr/bin/env python3
"""Video downloader using yt-dlp.

Usage:
    python -m seedance.tools.video_download --input urls.txt --output data/raw_videos/
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path


def download_videos(
    url_file: str,
    output_dir: str,
    max_videos: int | None = None,
    format_str: str = "best[height<=720]",
    concurrent: int = 4,
):
    """Download videos from a list of URLs.

    Args:
        url_file: File with one URL per line.
        output_dir: Output directory for videos.
        max_videos: Maximum number of videos to download.
        format_str: yt-dlp format selector.
        concurrent: Number of concurrent downloads.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if max_videos:
        urls = urls[:max_videos]

    print(f"Downloading {len(urls)} videos to {output_dir}")

    for i, url in enumerate(urls):
        output_template = os.path.join(output_dir, f"%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f", format_str,
            "-o", output_template,
            "--no-playlist",
            "--socket-timeout", "30",
            "--retries", "3",
            url,
        ]

        try:
            print(f"[{i+1}/{len(urls)}] {url}")
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  ⚠ Timeout: {url}")
        except subprocess.CalledProcessError:
            print(f"  ⚠ Failed: {url}")

    print(f"Done. Downloaded to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download videos for Seedance training")
    parser.add_argument("--input", type=str, required=True, help="File with video URLs")
    parser.add_argument("--output", type=str, default="data/raw_videos", help="Output directory")
    parser.add_argument("--max", type=int, default=None, help="Max videos to download")
    parser.add_argument("--format", type=str, default="best[height<=720]", help="yt-dlp format")
    args = parser.parse_args()

    download_videos(args.input, args.output, args.max, args.format)


if __name__ == "__main__":
    main()
