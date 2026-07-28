#!/usr/bin/env python3
"""
Build training manifest: use all available data on disk.

Prioritizes VLM captions > dataset captions > template captions.
Controls VoxCeleb2 proportion to prevent domination.

Usage:
    python scripts/build_balanced_manifest.py
    python scripts/build_balanced_manifest.py --vox_count 10000 --output data/manifests/train_full.csv
"""

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ── Helpers ──────────────────────────────────────────────────────────
import cv2


def scan_video_dir(directory: Path, recurse: bool = False) -> list[dict]:
    """Scan a directory for mp4/mkv/webm files and return video metadata."""
    videos = []
    glob_fn = directory.rglob if recurse else directory.glob
    for ext in ["*.mp4", "*.mkv", "*.webm"]:
        for vf in glob_fn(ext):
            if vf.is_dir():
                continue
            meta = _get_meta(vf)
            if meta:
                videos.append(meta)
    return videos


def _get_meta(video_path: Path) -> dict | None:
    """Get metadata for a video file."""
    video_path = video_path.resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = frames / max(fps, 1.0)
    if frames < 8 or duration > 300:  # skip extremely short or very long
        return None
    return {
        "video_path": str(video_path.relative_to(PROJECT_DIR)),
        "num_frames": frames, "height": h, "width": w,
        "fps": round(fps, 2), "duration_s": round(duration, 2),
        "file_name": video_path.name,
    }


def load_captions_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_webvid_meta() -> dict[str, str]:
    meta_dir = DATA_DIR / "webvid" / "metadata" / "data" / "train" / "partitions"
    caps = {}
    if meta_dir.exists():
        for cf in sorted(meta_dir.glob("*.csv")):
            with open(cf, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vid = row.get("videoid", "")
                    name = row.get("name", "")
                    if vid and name:
                        caps[vid + ".mp4"] = name
    return caps


def _caption_for(name: str, vlm: dict, captions_db: dict, webvid: dict,
                 fallback_fn, fb_seed: int) -> tuple[str, str]:
    """Resolve best caption: VLM > captions.json > webvid meta > fallback template."""
    # 1. VLM
    v = vlm.get(name, {})
    if v.get("caption_short"):
        return v["caption_short"], v.get("caption_long", v["caption_short"])
    # 2. captions.json
    c = captions_db.get(name, {})
    s = c.get("caption_short", "")
    if s and not re.match(r"A \d+s video at \d+x\d+", s):
        return s, c.get("caption_long", s)
    # 3. WebVid metadata
    w = webvid.get(name, "")
    if w:
        return w, w
    # 4. fallback
    fb = fallback_fn(fb_seed)
    return fb, fb


# ── Template captions (diversity via random variation) ───────────────
VOX_TEMPLATES = [
    "A person speaking directly to the camera, indoor setting",
    "Close-up of a person talking, plain background",
    "Talking head interview clip, studio lighting",
    "Someone giving a speech facing the camera",
    "Portrait shot of a person speaking to the lens",
    "Head and shoulders of someone talking, natural light",
    "A vlogger addressing the camera, casual setting",
    "A speaker narrating on screen, medium close-up",
    "Interview-style video, person answering questions",
    "A person explaining while looking at the camera",
]

GENERAL_TEMPLATES = [
    "A video clip showing {scene}", "{scene} footage",
    "A shot of {scene}", "Video of {scene}",
    "Footage of {scene}, natural colors",
]
SCENES = ["outdoor scenery", "urban street", "nature landscape",
          "city traffic", "animals moving", "food preparation",
          "technology equipment", "travel destination"]

ANIM_TEMPLATES = [
    "An animated scene with characters {action}",
    "Animation clip: characters {action}",
    "2D animation of {action}",
    "Cartoon sequence showing {action}",
    "Anime-style clip: {action}",
]
ANIM_ACTIONS = ["moving dynamically", "interacting with objects",
                "in a conversation", "in an action sequence",
                "with colorful backgrounds", "expressing emotions"]


def _vox_cap(seed: int) -> str:
    return VOX_TEMPLATES[seed % len(VOX_TEMPLATES)]


def _anim_cap(seed: int) -> str:
    return random.Random(seed).choice(ANIM_TEMPLATES).format(
        action=random.Random(seed + 1).choice(ANIM_ACTIONS))


def _general_cap(seed: int) -> str:
    return random.Random(seed).choice(GENERAL_TEMPLATES).format(
        scene=random.Random(seed + 1).choice(SCENES))


# ═══════════════════════════════════════════════════════════════════════
def build_manifest(
    vox_count: int = 50000,
    output: str = "data/manifests/train_full.csv",
    val_split: float = 0.05,
    seed: int = 42,
):
    random.seed(seed)
    rng = random.Random(seed)

    # ── Load caption sources ─────────────────────────────────────────
    vlm_captions = load_captions_json(DATA_DIR / "manifests" / "vlm_captions.json")
    print(f"[vlm] {len(vlm_captions)} captions")

    captions_db = load_captions_json(DATA_DIR / "manifests" / "captions.json")
    print(f"[captions.json] {len(captions_db)} entries")

    webvid_meta = load_webvid_meta()
    print(f"[webvid_meta] {len(webvid_meta)} entries")

    all_rows: list[dict] = []
    already = set()
    stats: Counter = Counter()

    FN = ["video_path", "num_frames", "height", "width", "fps", "duration_s",
          "audio_path", "caption_short", "caption_long", "caption_audio",
          "speaker_id", "dataset"]

    def _add(videos, dataset, caption_fn, cap_seed_start=0):
        """Add videos to manifest with dedup. Returns count added."""
        added = 0
        for i, v in enumerate(videos):
            name = v["file_name"]
            if name in already:
                continue
            cs, cl = _caption_for(name, vlm_captions, captions_db, webvid_meta,
                                  lambda s=cap_seed_start + i: caption_fn(s),
                                  cap_seed_start + i)
            all_rows.append({**v, "audio_path": "", "speaker_id": "",
                             "caption_short": cs, "caption_long": cl,
                             "caption_audio": "", "dataset": dataset})
            already.add(name)
            added += 1
        return added

    # ═════════════════════════════════════════════════════════════════
    # 1. PEXELS_PEOPLE (high-res people)
    # ═════════════════════════════════════════════════════════════════
    pex = scan_video_dir(DATA_DIR / "pexels_people")
    n = _add(pex, "pexels_people", lambda s: _general_cap(s), 10000)
    stats["pexels_people"] = n
    print(f"[pexels_people] {n}")

    # ═════════════════════════════════════════════════════════════════
    # 2. CELEBA_HQ
    # ═════════════════════════════════════════════════════════════════
    chq = scan_video_dir(DATA_DIR / "celeba_hq_videos")
    # CelebA-HQ manifest has proper captions — load them
    celeba_manifest = DATA_DIR / "manifests" / "celeba_hq_train.csv"
    celeba_caps = {}
    if celeba_manifest.exists():
        with open(celeba_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                celeba_caps[Path(row["video_path"]).name] = row["caption_short"]
    n = 0
    for v in chq:
        name = v["file_name"]
        if name in already:
            continue
        cs = celeba_caps.get(name, "")
        if not cs:
            cs = f"A portrait photograph"
        all_rows.append({**v, "audio_path": "", "speaker_id": "",
                         "caption_short": cs, "caption_long": cs,
                         "caption_audio": "", "dataset": "celeba_hq"})
        already.add(name)
        n += 1
    stats["celeba_hq"] = n
    print(f"[celeba_hq] {n}")

    # ═════════════════════════════════════════════════════════════════
    # 3. FILTERED/PEOPLE + FILTERED subcategories
    # ═════════════════════════════════════════════════════════════════
    for sub in ["people", "animals", "city", "food", "nature", "tech", "travel"]:
        d = DATA_DIR / "filtered" / sub
        if not d.exists():
            continue
        vids = scan_video_dir(d)
        n = _add(vids, f"filtered_{sub}", lambda s: _general_cap(s), 20000)
        stats[f"filtered_{sub}"] = n
        print(f"[filtered/{sub}] {n}")

    # Top-level filtered too
    ftop = scan_video_dir(DATA_DIR / "filtered")
    # only files directly in filtered/ (not subdirs)
    ftop = [v for v in ftop if str(Path(v["video_path"]).parent) == "data/filtered"]
    n = _add(ftop, "filtered", lambda s: _general_cap(s), 21000)
    stats["filtered"] = n
    print(f"[filtered] {n}")

    # ═════════════════════════════════════════════════════════════════
    # 4. HDTF (high-res talking faces)
    # ═════════════════════════════════════════════════════════════════
    hdtf = scan_video_dir(DATA_DIR / "hdtf" / "clips")
    # HDTF manifest has captions
    hdtf_manifest = DATA_DIR / "manifests" / "hdtf_manifest.csv"
    hdtf_caps = {}
    if hdtf_manifest.exists():
        with open(hdtf_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hdtf_caps[Path(row["video_path"]).name] = row.get("caption_short", "")
    n = 0
    for v in hdtf:
        name = v["file_name"]
        if name in already:
            continue
        cs = hdtf_caps.get(name, "A person talking to the camera")
        all_rows.append({**v, "audio_path": "", "speaker_id": "",
                         "caption_short": cs, "caption_long": cs,
                         "caption_audio": "", "dataset": "hdtf"})
        already.add(name)
        n += 1
    stats["hdtf"] = n
    print(f"[hdtf] {n}")

    # ═════════════════════════════════════════════════════════════════
    # 5. ANIMATION (all sources)
    # ═════════════════════════════════════════════════════════════════
    anim_dirs = [
        DATA_DIR / "animation_bilibili_filtered",
        DATA_DIR / "animation_filtered",
    ]
    anim_added = 0
    for ad in anim_dirs:
        if ad.exists():
            vids = scan_video_dir(ad)
            n = _add(vids, "animation", lambda s: _anim_cap(s), 30000 + anim_added)
            anim_added += n
            print(f"[animation] {ad.name}: {n}")
    stats["animation"] = anim_added
    print(f"[animation] total: {anim_added}")

    # ═════════════════════════════════════════════════════════════════
    # 6. BILIBILI PEOPLE
    # ═════════════════════════════════════════════════════════════════
    bili_dirs = [DATA_DIR / "people_bilibili_filtered", DATA_DIR / "people_bilibili"]
    for bd in bili_dirs:
        if bd.exists():
            vids = scan_video_dir(bd)
            n = _add(vids, "bilibili_people", lambda s: _general_cap(s), 40000)
            stats["bilibili_people"] += n
    print(f"[bilibili_people] {stats['bilibili_people']}")

    # ═════════════════════════════════════════════════════════════════
    # 6b. YOUTUBE PEOPLE (downloaded + VLM captioned)
    # ═════════════════════════════════════════════════════════════════
    yt_dirs = [DATA_DIR / "people_youtube_filtered", DATA_DIR / "people_youtube"]
    yt_added = 0
    for yd in yt_dirs:
        if yd.exists():
            vids = scan_video_dir(yd)
            n = _add(vids, "youtube_people", lambda s: _general_cap(s), 45000 + yt_added)
            yt_added += n
            print(f"[youtube_people] {yd.name}: {n}")
    stats["youtube_people"] = yt_added
    print(f"[youtube_people] total: {yt_added}")

    # ═════════════════════════════════════════════════════════════════
    # 7. WEBVID (all filtered + sample raw)
    # ═════════════════════════════════════════════════════════════════
    wv_filtered = scan_video_dir(DATA_DIR / "webvid_filtered")
    n = _add(wv_filtered, "webvid", lambda s: _general_cap(s), 50000)
    stats["webvid"] = n
    print(f"[webvid] filtered: {n}")

    # ═════════════════════════════════════════════════════════════════
    # 8. PEXELS categories
    # ═════════════════════════════════════════════════════════════════
    for cat in ["animals", "city", "food", "nature", "tech", "travel"]:
        d = DATA_DIR / f"pexels_{cat}"
        if d.exists():
            vids = scan_video_dir(d)
            n = _add(vids, f"pexels_{cat}", lambda s: _general_cap(s), 60000)
            stats[f"pexels_{cat}"] = n
    print(f"[pexels_categories] {sum(stats.get(f'pexels_{c}', 0) for c in ['animals','city','food','nature','tech','travel'])}")

    # ═════════════════════════════════════════════════════════════════
    # 9. VOXCELEB2 (downsample from 1.1M — use manifest paths directly)
    # ═════════════════════════════════════════════════════════════════
    vox_manifest = DATA_DIR / "manifests" / "train_stage1.csv"
    vox_added = 0
    if vox_manifest.exists():
        vox_rows = []
        with open(vox_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("dataset", "") == "voxceleb2":
                    # Fix path from Windows to local
                    vp = row["video_path"]
                    if vp.startswith("D:\\") or vp.startswith("data/"):
                        pass
                    vox_rows.append(row)
        rng.shuffle(vox_rows)
        take = min(len(vox_rows), vox_count)
        print(f"[voxceleb2] {len(vox_rows)} candidates, taking {take}...")
        for i, row in enumerate(vox_rows[:take]):
            vp = row["video_path"]
            # Normalize path: strip Windows prefix, ensure local
            if "voxceleb2" in vp:
                vp = vp[vp.index("voxceleb2"):]
            local = PROJECT_DIR / "data" / vp
            if not local.exists():
                continue
            meta = _get_meta(local)
            if not meta:
                continue
            cs = _vox_cap(i)
            all_rows.append({**meta, "audio_path": row.get("audio_path", ""),
                             "speaker_id": row.get("speaker_id", ""),
                             "caption_short": cs, "caption_long": cs,
                             "caption_audio": "", "dataset": "voxceleb2"})
            already.add(meta["file_name"])
            vox_added += 1
            if (vox_added + 1) % 10000 == 0:
                print(f"  [{vox_added}/{take}]")
        stats["voxceleb2"] = vox_added
    print(f"[voxceleb2] {vox_added} / {vox_count} target")

    # ═════════════════════════════════════════════════════════════════
    # SHUFFLE, SPLIT, WRITE
    # ═════════════════════════════════════════════════════════════════
    for row in all_rows:
        row.pop("file_name", None)
    rng.shuffle(all_rows)

    split_idx = int(len(all_rows) * (1 - val_split))
    train, val = all_rows[:split_idx], all_rows[split_idx:]

    base = str(output).replace(".csv", "")
    for split_name, data in [("train", train), ("val", val)]:
        out = f"{base}_{split_name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FN)
            w.writeheader()
            w.writerows(data)
        print(f"\n  [{split_name}] {len(data)} → {out}")

    # ── Report ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MANIFEST SUMMARY: {len(all_rows)} videos")
    print(f"{'='*70}")
    ds_dist = Counter(r["dataset"] for r in train)
    for ds, n in ds_dist.most_common():
        pct = n / len(train) * 100
        bar = "█" * int(pct) + "░" * (50 - int(pct))
        print(f"  {ds:25s} {n:>6d} ({pct:5.1f}%) {bar[:50]}")

    res_dist = Counter()
    for r in train:
        res_dist[f"{r['width']}x{r['height']}"] += 1
    print(f"\nTop resolutions:")
    for res, n in res_dist.most_common(8):
        print(f"  {res:15s} {n:>6d} ({n/len(train)*100:.1f}%)")

    uniq = len(set(r["caption_short"] for r in train))
    total_dur = sum(float(r["duration_s"]) for r in train) / 3600
    print(f"\nUnique captions: {uniq}")
    print(f"Total duration:  {total_dur:.1f} hours")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Build training manifest from all disk data")
    parser.add_argument("--vox_count", type=int, default=50000,
                        help="Max VoxCeleb2 samples (default: 50000)")
    parser.add_argument("--output", default="data/manifests/train_full.csv")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifest(args.vox_count, args.output, args.val_split, args.seed)


if __name__ == "__main__":
    main()
