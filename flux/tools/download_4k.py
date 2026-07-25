#!/usr/bin/env python3
"""4K training data downloader — UltraVideo, Pexels 4K, Nature-1k.

Usage:
    # UltraVideo (primary 4K dataset, 42K videos)
    python -m flux.tools.download_4k --source ultravideo --max_videos 10000

    # Pexels 4K-only
    export PEXELS_API_KEY="your-key"
    python -m flux.tools.download_4k --source pexels_4k --max_videos 5000

    # All sources combined
    python -m flux.tools.download_4k --source all --max_videos 20000

    # Dry run: check how many videos available without downloading
    python -m flux.tools.download_4k --source ultravideo --dry_run
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import urlencode


# ═══════════════════════════════════════════════════════════════════════
# UltraVideo — NeurIPS 2025, 42K 4K videos with structured captions
# ═══════════════════════════════════════════════════════════════════════

def download_ultravideo(
    output_dir: str,
    max_videos: int = 10000,
    workers: int = 8,
    dry_run: bool = False,
) -> int:
    """Download UltraVideo dataset from HuggingFace.

    UltraVideo: 42K short videos (3-10s) + 17K long videos (10s+)
    4K (2160p) resolution, 100+ categories, avg 824-word captions.

    Args:
        output_dir: Output directory.
        max_videos: Max videos to download.
        workers: Parallel workers.
        dry_run: If True, only check availability.

    Returns:
        Number of downloaded videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[UltraVideo] NeurIPS 2025 — 42K 4K videos with structured captions")

    try:
        from huggingface_hub import list_repo_files, hf_hub_download

        # List available files
        print("[UltraVideo] Listing repo files...")
        files = list_repo_files("APRIL-AIGC/UltraVideo", repo_type="dataset")
        video_files = [f for f in files if f.endswith(('.mp4', '.webm', '.mkv'))]
        caption_files = [f for f in files if f.endswith(('.json', '.jsonl', '.csv'))]

        print(f"[UltraVideo] Found {len(video_files)} videos, {len(caption_files)} caption files")

        if dry_run:
            print(f"[UltraVideo] Dry run — {min(len(video_files), max_videos)} videos available")
            return 0

        # Download captions first (small, fast)
        for cf in caption_files[:5]:  # First 5 caption files
            try:
                hf_hub_download(
                    "APRIL-AIGC/UltraVideo", cf,
                    repo_type="dataset", local_dir=output_dir,
                )
            except Exception as e:
                print(f"  Caption {cf}: {e}")

        # Download videos
        to_download = video_files[:max_videos]
        print(f"[UltraVideo] Downloading {len(to_download)} videos...")

        succeeded = 0
        def _dl_one(fn):
            try:
                hf_hub_download(
                    "APRIL-AIGC/UltraVideo", fn,
                    repo_type="dataset", local_dir=output_dir,
                )
                return fn
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_dl_one, f): f for f in to_download}
            for i, f in enumerate(as_completed(futs)):
                if f.result():
                    succeeded += 1
                if (i + 1) % 200 == 0:
                    print(f"  [{i+1}/{len(to_download)}] {succeeded} downloaded")

        print(f"[UltraVideo] Done: {succeeded}/{len(to_download)} videos → {output_dir}")
        return succeeded

    except ImportError:
        print("[UltraVideo] huggingface_hub not available. Install: pip install huggingface_hub")
        return 0
    except Exception as e:
        print(f"[UltraVideo] Error: {e}")
        print("[UltraVideo] Trying HTTP fallback...")
        return _download_ultravideo_http(output_dir, max_videos, workers, dry_run)


def _download_ultravideo_http(
    output_dir: str, max_videos: int, workers: int, dry_run: bool,
) -> int:
    """HTTP fallback for UltraVideo."""
    # UltraVideo has a HF mirror accessible page
    base = "https://huggingface.co/datasets/APRIL-AIGC/UltraVideo"
    print(f"[UltraVideo] Visit {base} to download manually")
    print("[UltraVideo] Use: huggingface-cli download APRIL-AIGC/UltraVideo --repo-type dataset --local-dir data/ultravideo")
    return 0


# ═══════════════════════════════════════════════════════════════════════
# Pexels 4K — stock footage, API-driven
# ═══════════════════════════════════════════════════════════════════════

def search_pexels_4k(
    query: str, api_key: str, max_results: int = 100,
    min_duration: int = 5, max_duration: int = 30,
) -> list[dict]:
    """Search Pexels API for 4K-only videos."""
    videos = []
    page = 1
    per_page = min(80, max_results)

    while len(videos) < max_results:
        params = {
            "query": query, "per_page": per_page, "page": page,
            "size": "large",  # 4K only
        }
        url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
        req = Request(url, headers={"Authorization": api_key})
        req.add_header("User-Agent", "Seedance/2.0")

        try:
            with urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except Exception:
            break

        if not data.get("videos"):
            break

        for v in data["videos"]:
            dur = v.get("duration", 0)
            w = v.get("width", 0)
            if dur < min_duration or dur > max_duration:
                continue
            if w < 3840:  # Must be 4K
                continue

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
                    "tags": ", ".join(v.get("url", "")),
                })
            if len(videos) >= max_results:
                break
        page += 1
        if page > 15:
            break

    return videos


def download_pexels_4k(
    output_dir: str,
    api_key: str,
    max_videos: int = 5000,
    workers: int = 16,
    dry_run: bool = False,
) -> int:
    """Download 4K videos from Pexels.

    Args:
        output_dir: Output directory.
        api_key: Pexels API key.
        max_videos: Max videos.
        workers: Parallel workers.
        dry_run: If True, only search, don't download.

    Returns:
        Number downloaded.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Pexels 4K] Searching for 4K stock footage...")

    queries = [
        "nature landscape 4k", "city aerial drone", "ocean waves beach",
        "mountains hiking", "forest river waterfall", "urban street night",
        "wildlife animals", "sports action slow motion", "food cooking closeup",
        "travel adventure scenic", "technology office modern",
        "sunset sunrise timelapse", "flowers garden macro",
        "people portrait lifestyle", "abstract particles lights",
    ]

    seen = set()
    all_vids = []
    for q in queries:
        vids = search_pexels_4k(q, api_key, max_results=200, min_duration=3, max_duration=25)
        new = 0
        for v in vids:
            if v["id"] not in seen:
                seen.add(v["id"])
                all_vids.append(v)
                new += 1
        print(f"  {q}: {new} new 4K videos (total: {len(all_vids)})")

    print(f"\n[Pexels 4K] Found {len(all_vids)} unique 4K videos")

    if dry_run:
        return len(all_vids)

    # Download via HTTP (fast — Pexels URLs expire quickly)
    to_dl = all_vids[:max_videos]
    succeeded = []

    def _dl(vid_info):
        vid = vid_info["id"]
        url = vid_info["url"]
        path = Path(output_dir) / f"pexels_{vid}.mp4"
        if path.exists() and path.stat().st_size > 10000:
            vid_info["video_path"] = str(path)
            return vid_info
        try:
            req = Request(url, headers={"User-Agent": "Seedance/2.0"})
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) > 50000:
                    path.write_bytes(data)
                    vid_info["video_path"] = str(path)
                    return vid_info
        except Exception:
            pass
        return None

    print(f"[Pexels 4K] Downloading {len(to_dl)} videos...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_dl, v): v for v in to_dl}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                succeeded.append(r)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(to_dl)}] {len(succeeded)} ok")

    # Save manifest
    if succeeded:
        manifest_path = os.path.join(output_dir, "pexels_4k_manifest.csv")
        for v in succeeded:
            v.pop("url", None)
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(succeeded[0].keys()))
            w.writeheader()
            w.writerows(succeeded)

    n = len(succeeded)
    sz = sum(Path(str(v.get("video_path", ""))).stat().st_size
             for v in succeeded if v.get("video_path") and Path(v["video_path"]).exists()) / 1e9
    print(f"[Pexels 4K] Done: {n} videos, {sz:.1f}GB → {output_dir}")
    return n


# ═══════════════════════════════════════════════════════════════════════
# Main — orchestrate all sources
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download 4K training data for Seedance 2.5-scale models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # UltraVideo only
  python -m flux.tools.download_4k --source ultravideo --max_videos 10000

  # Pexels 4K only
  export PEXELS_API_KEY=your-key
  python -m flux.tools.download_4k --source pexels_4k --max_videos 5000

  # All sources
  python -m flux.tools.download_4k --source all --max_videos 20000

  # Check availability without downloading
  python -m flux.tools.download_4k --source all --dry_run
""",
    )
    parser.add_argument("--source", type=str, default="all",
                        choices=["ultravideo", "pexels_4k", "all"],
                        help="Data source")
    parser.add_argument("--output", type=str, default="data/4k",
                        help="Output directory")
    parser.add_argument("--max_videos", type=int, default=10000,
                        help="Max videos total (across all sources)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check availability without downloading")
    parser.add_argument("--pexels_api_key", type=str, default=None,
                        help="Pexels API key (or set PEXELS_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.pexels_api_key or os.environ.get("PEXELS_API_KEY", "")

    total = 0
    sources = ["ultravideo", "pexels_4k"] if args.source == "all" else [args.source]

    for src in sources:
        out = os.path.join(args.output, src)
        remaining = args.max_videos - total

        if src == "ultravideo":
            n = download_ultravideo(out, max_videos=remaining,
                                    workers=args.workers, dry_run=args.dry_run)
            total += n

        elif src == "pexels_4k":
            if not api_key:
                print("[Pexels 4K] No API key — skipping. Set PEXELS_API_KEY env var.")
                continue
            n = download_pexels_4k(out, api_key, max_videos=remaining,
                                   workers=args.workers, dry_run=args.dry_run)
            total += n

    print(f"\n{'='*60}")
    print(f"4K data download complete: {total} videos total → {args.output}")
    if total > 0:
        print(f"\nNext: quality filter → build manifest → Stage 5 fine-tune")
        print(f"  python -m flux.tools.quality_filter --input {args.output} --output data/4k_filtered --min_height 2160")
        print(f"  python -m flux.tools.build_manifest --video_dir data/4k_filtered --output data/manifests/train_4k.csv")
        print(f"  bash scripts/train.sh 200b  # or: python scripts/train.py --config configs/train/stage1_200b_moe.yaml")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
