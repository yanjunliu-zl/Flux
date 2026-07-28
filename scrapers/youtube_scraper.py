#!/usr/bin/env python3
"""YouTube animation / meme video scraper.

Searches YouTube for meme-style and 2D animation content:
- Stick figure animation ("stickman animation")
- Meme / goofy animation ("silly animation meme")
- Simple cartoon shorts ("simple animation short")

Usage:
    python scrapers/youtube_scraper.py --query "stickman animation meme" --max_videos 500
    python scrapers/youtube_scraper.py --all_queries --max_videos 2000
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════════
# YouTube search via yt-dlp (no API key needed)
# ═══════════════════════════════════════════════════════════════════════

def search_youtube(
    query: str,
    max_results: int = 500,
    min_duration: int = 10,
    max_duration: int = 300,
) -> list[dict]:
    """Search YouTube using yt-dlp's built-in search.

    Returns video metadata without downloading.

    Args:
        query: Search query.
        max_results: Max results.
        min_duration, max_duration: Duration range in seconds.

    Returns:
        List of dicts with video_id, title, duration, url, etc.
    """
    yt_dlp = _find_yt_dlp()
    # yt-dlp search: ytsearchN:query
    search_term = f"ytsearch{min(max_results, 200)}:{query}"

    cmd = [
        yt_dlp,
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
        "--match-filter", f"duration >= {min_duration} & duration <= {max_duration}",
        search_term,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                videos.append({
                    "video_id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "duration": data.get("duration", 0) or 0,
                    "view_count": data.get("view_count", 0) or 0,
                    "url": data.get("webpage_url", ""),
                    "channel": data.get("channel", ""),
                    "description": (data.get("description", "") or "")[:200],
                    "query": query,
                })
            except json.JSONDecodeError:
                continue

        return videos
    except subprocess.TimeoutExpired:
        return []


def _find_yt_dlp() -> str:
    import shutil
    yt = shutil.which("yt-dlp")
    if yt:
        return yt
    venv = Path(sys.executable).parent / "yt-dlp"
    if venv.exists():
        return str(venv)
    return "yt-dlp"


# ═══════════════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════════════

def download_video(info: dict, output_dir: str, timeout: int = 120) -> dict | None:
    """Download a single YouTube video."""
    vid = info["video_id"]
    output_template = os.path.join(output_dir, f"{vid}.%(ext)s")
    yt_dlp = _find_yt_dlp()

    cmd = [
        yt_dlp,
        "-f", "best[height<=1080]",
        "-o", output_template,
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "2",
        "--quiet",
        "--no-warnings",
        info["url"],
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        for ext in ["mp4", "webm", "mkv"]:
            path = Path(output_dir) / f"{vid}.{ext}"
            if path.exists() and path.stat().st_size > 10240:
                info["video_path"] = str(path)
                return info
        return None
    except subprocess.TimeoutExpired:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

# Pre-built search queries for portrait/people content
PEOPLE_QUERIES = [
    "portrait photography behind the scenes natural light",
    "daily vlog casual day in my life aesthetic",
    "street fashion lookbook outfit ideas casual",
    "close up portrait photoshoot outdoor",
    "makeup tutorial no talking close up face",
    "travel vlog walking around city streets",
    "dance practice freestyle solo studio",
    "fitness workout home gym single person",
    "cooking vlog home kitchen hands only",
    "street interview asking people questions",
    "selfie video talking to camera indoors",
    "unboxing haul clothing accessories hands",
    "hair tutorial styling braiding close up",
    "drawing sketching art process hands close",
    "playing musical instrument piano guitar solo",
    "yoga stretching routine at home morning",
    "study with me library desk pov",
    "room tour apartment house aesthetic vlog",
    "photoshoot model posing outdoor nature",
    "skincare routine morning night close up face",
]

# Pre-built search queries for "沙雕动画"-style content
DEFAULT_QUERIES = [
    "stickman animation meme funny",
    "silly animation short",
    "simple 2D animation comedy",
    "hand drawn cartoon meme",
    "animation meme compilation",
    "poorly drawn animation funny",
    "low budget animation comedy sketch",
    "stick figure fight animation",
    "meme animation green screen",
    "animated shitpost",
    "gartic phone animation funny",
    "flipaclip animation meme",
    "pivot animator stick figure funny",
    "source filmmaker meme animation",
    "roblox animation funny meme",
]


def main():
    parser = argparse.ArgumentParser(
        description="YouTube animation meme video scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=None,
                        help="Single search query")
    parser.add_argument("--all_queries", action="store_true",
                        help="Use all pre-built animation meme queries")
    parser.add_argument("--people", action="store_true",
                        help="Use portrait/vlog/people queries (replaces --all_queries)")
    parser.add_argument("--output", type=str, default="data/animation_yt",
                        help="Output directory")
    parser.add_argument("--max_videos", type=int, default=2000,
                        help="Max videos to download total")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--search_only", action="store_true",
                        help="Only search, don't download")
    parser.add_argument("--min_duration", type=int, default=5)
    parser.add_argument("--max_duration", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Search ─────────────────────────────────────────────────────
    if args.people:
        queries = PEOPLE_QUERIES
        args.output = args.output or "data/people_youtube"
    elif args.all_queries:
        queries = DEFAULT_QUERIES
    else:
        queries = [args.query or DEFAULT_QUERIES[0]]
    per_query = max(1, args.max_videos // len(queries))

    all_videos = []
    for q in queries:
        print(f"[YouTube] Searching: {q}...")
        vids = search_youtube(q, max_results=per_query,
                              min_duration=args.min_duration,
                              max_duration=args.max_duration)
        all_videos.extend(vids)
        print(f"  {q[:50]}: {len(vids)} videos")

    # Deduplicate
    seen = set()
    unique = []
    for v in all_videos:
        if v["video_id"] not in seen:
            seen.add(v["video_id"])
            unique.append(v)
    all_videos = unique[:args.max_videos]
    print(f"\n[YouTube] Total unique: {len(all_videos)} videos")

    # Save search results
    search_csv = os.path.join(args.output, "search_results.csv")
    with open(search_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "title", "duration", "view_count", "url", "channel", "query"])
        w.writeheader()
        for v in all_videos:
            w.writerow({k: v.get(k, "") for k in w.fieldnames})

    if args.search_only:
        print(f"[YouTube] Search results: {search_csv}")
        return

    # ── Download ───────────────────────────────────────────────────
    print(f"\n[YouTube] Downloading {len(all_videos)} videos...")
    succeeded = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_video, v, args.output): v for v in all_videos}
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                succeeded.append(r)
            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(time.time() - start, 1)
                print(f"  [{i+1}/{len(all_videos)}] {len(succeeded)} ok | {rate:.1f} vids/s")

    # ── Manifest ───────────────────────────────────────────────────
    manifest_path = os.path.join(args.output, "manifest.csv")
    if succeeded:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            fields = ["video_id", "title", "duration", "video_path", "url", "channel", "query"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(succeeded)

    elapsed = time.time() - start
    files = list(Path(args.output).glob("*.mp4")) + list(Path(args.output).glob("*.webm"))
    sz = sum(f.stat().st_size for f in files) / 1e9

    print(f"\n[YouTube] Done in {elapsed:.0f}s:")
    print(f"  Downloaded: {len(files)} videos, {sz:.1f}GB")
    print(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
