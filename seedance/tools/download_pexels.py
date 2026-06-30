#!/usr/bin/env python3
"""Pexels/Pixabay free stock video downloader.

Downloads high-quality royalty-free videos for training.
Pexels videos are 1080p-4K, most with ambient audio.

Usage:
    # Search and download
    python -m seedance.tools.download_pexels --query "nature landscape" \
        --max_videos 1000 --output data/pexels/

    # Download from a list of URLs
    python -m seedance.tools.download_pexels --url_file urls.txt --output data/pexels/

    # Set API key (free: pexels.com/api)
    export PEXELS_API_KEY="your-key-here"
"""

import argparse
import csv
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import urlencode


def _find_yt_dlp() -> str:
    """Find the yt-dlp executable in PATH or venv."""
    yt = shutil.which("yt-dlp")
    if yt:
        return yt
    # Try venv bin
    venv_bin = Path(sys.executable).parent / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    # Try pip location
    for p in Path(sys.executable).parent.parent.glob("**/yt-dlp"):
        return str(p)
    return "yt-dlp"  # fallback


def search_pexels(
    query: str,
    api_key: str,
    max_results: int = 100,
    min_duration: int = 5,
    max_duration: int = 30,
    min_width: int = 1280,
) -> list[dict]:
    """Search Pexels API for videos.

    Args:
        query: Search query.
        api_key: Pexels API key (free from pexels.com/api).
        max_results: Maximum results to return.
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.
        min_width: Minimum video width.

    Returns:
        List of video dicts with 'url', 'duration', 'width', 'height', 'id'.
    """
    videos = []
    page = 1
    per_page = min(80, max_results)

    while len(videos) < max_results:
        params = {
            "query": query,
            "per_page": per_page,
            "page": page,
            "size": "medium",  # ~720p
        }
        url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
        req = Request(url, headers={"Authorization": api_key})
        req.add_header("User-Agent", "Seedance/2.0")

        try:
            with urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except Exception as e:
            print(f"  API error: {e}")
            break

        if not data.get("videos"):
            break

        for v in data["videos"]:
            dur = v.get("duration", 0)
            w = v.get("width", 0)
            if dur < min_duration or dur > max_duration:
                continue
            if w < min_width:
                continue

            # Pick the best quality video file
            video_files = sorted(
                v.get("video_files", []),
                key=lambda f: (f.get("width", 0), f.get("height", 0)),
                reverse=True,
            )
            if video_files:
                videos.append({
                    "id": v["id"],
                    "url": video_files[0]["link"],
                    "duration": dur,
                    "width": w,
                    "height": v.get("height", 0),
                    "user": v.get("user", {}).get("name", ""),
                    "tags": ", ".join(v.get("tags", [])),
                })

            if len(videos) >= max_results:
                break

        page += 1
        if page > 10:  # Safety limit
            break

    return videos


def search_pixabay(
    query: str,
    api_key: str,
    max_results: int = 100,
) -> list[dict]:
    """Search Pixabay API for videos.

    Args:
        query: Search query.
        api_key: Pixabay API key (free from pixabay.com/api/docs).
        max_results: Maximum results.

    Returns:
        List of video dicts.
    """
    videos = []
    page = 1
    per_page = min(200, max_results)

    while len(videos) < max_results:
        params = {
            "key": api_key,
            "q": query,
            "video_type": "film",
            "per_page": per_page,
            "page": page,
        }
        url = f"https://pixabay.com/api/videos/?{urlencode(params)}"

        try:
            with urlopen(url, timeout=30) as resp:
                data = json.load(resp)
        except Exception:
            break

        for v in data.get("hits", []):
            dur = v.get("duration", 0)
            if dur < 3 or dur > 30:
                continue

            videos.append({
                "id": v["id"],
                "url": v.get("videos", {}).get("medium", {}).get("url", ""),
                "duration": dur,
                "width": v.get("video_width", 0),
                "height": v.get("video_height", 0),
                "user": v.get("user", ""),
                "tags": v.get("tags", ""),
            })

            if len(videos) >= max_results:
                break

        page += 1

    return videos


def download_video(info: dict, output_dir: str) -> dict | None:
    """Download a single video via HTTP (fast, for time-sensitive CDN URLs).

    Args:
        info: Dict with 'id', 'url', 'duration'.
        output_dir: Output directory.

    Returns:
        Updated info with 'video_path' if successful.
    """
    vid = info["id"]
    url = info["url"]
    output_path = Path(output_dir) / f"{vid}.mp4"

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 2048:
        info["video_path"] = str(output_path)
        return info

    # Direct HTTP download (Pexels URLs are temporary, need speed)
    try:
        from urllib.request import urlopen
        req = Request(url, headers={"User-Agent": "Seedance/2.0"})
        req.add_header("Accept", "video/mp4")

        with urlopen(req, timeout=30) as resp:
            size = int(resp.headers.get("Content-Length", 0))
            if size < 10_000:  # Skip tiny files (thumbnails)
                return None

            data = resp.read()
            if len(data) < 10_000:
                return None

            output_path.write_bytes(data)

        if output_path.stat().st_size > 2048:
            info["video_path"] = str(output_path)
            return info
        output_path.unlink(missing_ok=True)
        return None

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Download Pexels/Pixabay stock videos")
    parser.add_argument("--query", type=str, default="nature landscape ocean city people",
                        help="Search query (space-separated keywords)")
    parser.add_argument("--source", type=str, default="pexels", choices=["pexels", "pixabay"])
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set env var)")
    parser.add_argument("--url_file", type=str, default=None, help="File with video URLs")
    parser.add_argument("--output", type=str, default="data/stock_videos", help="Output directory")
    parser.add_argument("--max_videos", type=int, default=500, help="Max videos")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Get videos from URL file or API search
    if args.url_file:
        with open(args.url_file) as f:
            urls = [line.strip() for line in f if line.strip() and "http" in line]
        videos = [{"id": f"url_{i:04d}", "url": u, "duration": 5} for i, u in enumerate(urls)]
        print(f"[Stock] Loaded {len(videos)} URLs from file")
    else:
        api_key = args.api_key or os.environ.get(
            "PEXELS_API_KEY" if args.source == "pexels" else "PIXABAY_API_KEY", ""
        )
        if not api_key:
            print("[Stock] No API key found. Set:")
            print(f"  export {'PEXELS_API_KEY' if args.source == 'pexels' else 'PIXABAY_API_KEY'}=your-key")
            print(f"  Get free key: {'pexels.com/api' if args.source == 'pexels' else 'pixabay.com/api/docs'}")
            return

        print(f"[Stock] Searching {args.source} for: {args.query}")
        if args.source == "pexels":
            videos = search_pexels(args.query, api_key, args.max_videos)
        else:
            videos = search_pixabay(args.query, api_key, args.max_videos)
        print(f"[Stock] Found {len(videos)} videos")

    if not videos:
        print("[Stock] No videos found")
        return

    # Download
    print(f"[Stock] Downloading {min(len(videos), args.max_videos)} videos...")
    videos = videos[:args.max_videos]
    succeeded = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_video, v, args.output): v for v in videos}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                succeeded.append(result)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(videos)}] {len(succeeded)} downloaded")

    # Save manifest
    manifest_path = os.path.join(args.output, "stock_manifest.csv")
    if succeeded:
        # Flatten tags/user fields
        for v in succeeded:
            v.pop("url", None)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=succeeded[0].keys())
            writer.writeheader()
            writer.writerows(succeeded)

    print(f"[Stock] Done: {len(succeeded)}/{len(videos)} downloaded → {args.output}")
    print(f"Next: python -m seedance.tools.quality_filter --input {args.output} --output data/filtered/")


if __name__ == "__main__":
    main()
