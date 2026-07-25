#!/usr/bin/env python3
"""
Build a balanced training manifest with diverse human/people data.

Fixes the 99.75% VoxCeleb2 imbalance by:
1. Downsampling VoxCeleb2 and enriching its captions
2. Adding pexels_people + filtered/people + people-related videos from captions.json
3. Adding diverse general videos (WebVid, Pexels categories) for visual diversity
4. Outputting a new manifest with a reasonable data mix

Usage:
    python scripts/build_balanced_manifest.py [--output data/manifests/train_balanced.csv]
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

# ── Caption templates for VoxCeleb2 (diverse talking-head descriptions) ──
VOX_TEMPLATES = [
    "A person speaking directly to the camera, indoor setting",
    "Close-up of a person talking, plain background",
    "A speaker addressing the camera in a studio interview",
    "Someone giving a speech or presentation, facing forward",
    "Talking head shot, person looking at the lens",
    "A person having a conversation on screen, medium close-up",
    "Interview-style video, person speaking to the audience",
    "A person narrating or explaining something on camera",
    "Head and shoulders shot of someone talking",
    "A speaker presenting, front-facing camera angle",
    "Portrait-style video of a person speaking",
    "Someone recording a video message, direct eye contact",
    "A vlogger talking to their camera, casual setting",
    "Professional interview clip, person answering questions",
]

# ── Caption templates for pexels_people without VLM captions ──
PEOPLE_TEMPLATES = [
    "A person {action} {setting}",
    "{angle} shot of {subject} {action} {setting}",
    "{subject} {action} in {setting}, {quality}",
    "{quality} footage of {subject} {action}",
    "{angle} view: {subject} {action} {setting}",
]

ACTIONS = [
    "walking", "standing", "sitting", "smiling at the camera", "looking away thoughtfully",
    "posing for a portrait", "moving gracefully", "gesturing with hands", "turning their head",
    "showing an expression", "interacting naturally", "facing the camera", "in motion",
    "relaxing", "posing", "looking into the distance", "laughing", "engaged in activity",
]

SETTINGS = [
    "indoors with natural light", "outdoors in daylight", "against a plain wall",
    "in a studio setting", "in an urban environment", "with a blurred background",
    "in a natural outdoor scene", "in warm ambient light", "in a bright room",
    "in soft window light", "under studio lighting", "in a casual environment",
    "with dramatic lighting", "in golden hour light", "in a contemporary setting",
]

SUBJECTS = [
    "a young woman", "a young man", "a woman", "a man", "a person",
    "a model", "an adult", "a girl", "a boy", "a teenage girl",
    "a teenage boy", "an elderly person", "a middle-aged woman", "a middle-aged man",
    "someone", "an elegant woman", "a stylish person",
]

ANGLES = [
    "Medium close-up", "Close-up portrait", "Wide", "Medium", "Tight", "Full-body",
    "Half-body", "Shoulders-up portrait", "Three-quarter",
]

QUALITIES = [
    "high quality", "cinematic", "professional", "beautifully lit", "sharp 4K",
    "vivid", "crisp", "well-composed", "stunning", "detailed",
]


def generate_people_caption(seed: int) -> str:
    """Generate a descriptive pseudo-caption for a person video."""
    rng = random.Random(seed)
    template = rng.choice(PEOPLE_TEMPLATES)
    caption = template.format(
        action=rng.choice(ACTIONS),
        setting=rng.choice(SETTINGS),
        subject=rng.choice(SUBJECTS),
        angle=rng.choice(ANGLES),
        quality=rng.choice(QUALITIES),
    )
    # Sometimes add a second sentence
    if rng.random() < 0.3:
        extras = [
            " Natural expression throughout.",
            " The lighting is flattering.",
            " Smooth camera movement.",
            " Clean, sharp focus.",
            " Professional color grading.",
            " Minimal background distractions.",
            " Great composition and framing.",
        ]
        caption += rng.choice(extras)
    return caption


def load_captions() -> dict:
    """Load captions.json if it exists."""
    captions_file = DATA_DIR / "manifests" / "captions.json"
    if captions_file.exists():
        with open(captions_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def scan_video_dir(directory: Path) -> list[dict]:
    """Scan a directory for mp4 files and return video metadata."""
    videos = []
    for vf in sorted(directory.glob("*.mp4")):
        import cv2
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frames / max(fps, 1.0)
        cap.release()

        # Skip very short (<1s) or very long (>60s) videos
        if frames < 16 or duration > 60:
            continue

        videos.append({
            "video_path": str(vf.relative_to(PROJECT_DIR)),
            "num_frames": frames,
            "height": h,
            "width": w,
            "fps": round(fps, 2),
            "duration_s": round(duration, 2),
            "file_name": vf.name,
        })
    return videos


def get_video_metadata(video_path_str: str) -> dict | None:
    """Get metadata for a video by its path (relative from project root)."""
    full_path = PROJECT_DIR / video_path_str
    if not full_path.exists():
        return None
    import cv2
    cap = cv2.VideoCapture(str(full_path))
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / max(fps, 1.0)
    cap.release()

    if frames < 16 or duration > 60:
        return None

    return {
        "video_path": video_path_str,
        "num_frames": frames,
        "height": h,
        "width": w,
        "fps": round(fps, 2),
        "duration_s": round(duration, 2),
    }


def load_webvid_metadata() -> dict[str, str]:
    """Load WebVid metadata: videoid → caption (name field)."""
    meta_dir = DATA_DIR / "webvid" / "metadata" / "data" / "train" / "partitions"
    captions: dict[str, str] = {}
    if meta_dir.exists():
        for csv_file in sorted(meta_dir.glob("*.csv")):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("videoid", "")
                    name = row.get("name", "")
                    if vid and name:
                        captions[vid + ".mp4"] = name
    return captions


PEOPLE_KW = [
    'person', 'people', 'man ', 'woman', 'women', 'face', 'talking', 'speaking',
    'smiling', 'walking', 'running', 'dancing', 'hands', 'crowd', 'portrait',
    'businessman', 'businesswoman', 'worker', 'child', 'girl', 'boy', 'lady',
    'model', 'actor', 'human', 'body', 'hair', 'makeup', 'fashion',
    'dress', 'suit', 'posing', 'looking', 'wearing', 'beautiful',
    'young man', 'young woman', 'guy', 'couple', 'family', 'baby',
    'student', 'doctor', 'nurse', 'athlete', 'dancer', 'singer',
    'teenager', 'elderly', 'grandma', 'grandpa', 'mother', 'father',
    'daughter', 'son', 'friend', 'colleague', 'customer', 'patient',
]


def _is_people_caption(text: str, min_kw: int = 2) -> bool:
    """Check if a caption describes people/humans."""
    s = text.lower()
    return sum(1 for kw in PEOPLE_KW if kw in s) >= min_kw


def build_balanced_manifest(
    vox_count: int = 15000,
    general_count: int = 3000,
    output: str = "data/manifests/train_balanced.csv",
    val_split: float = 0.05,
    seed: int = 42,
):
    """Build a balanced training manifest."""
    random.seed(seed)
    rng = random.Random(seed)
    captions_db = load_captions()
    webvid_meta = load_webvid_metadata()
    # Load VLM captions if available (takes priority over template captions)
    vlm_captions_file = DATA_DIR / "manifests" / "vlm_captions.json"
    vlm_captions = {}
    if vlm_captions_file.exists():
        with open(vlm_captions_file, "r", encoding="utf-8") as f:
            vlm_captions = json.load(f)
        print(f"[vlm_captions] Loaded {len(vlm_captions)} VLM captions")
    print(f"[webvid_meta] Loaded {len(webvid_meta)} captions from WebVid metadata")

    all_rows: list[dict] = []
    already_added: set[str] = set()
    stats: dict[str, int] = Counter()

    fieldnames = [
        "video_path", "num_frames", "height", "width", "fps", "duration_s",
        "audio_path", "caption_short", "caption_long", "caption_audio",
        "speaker_id", "dataset",
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # 1. PEXELS_PEOPLE — 168 high-res people videos (highest quality)
    # ═══════════════════════════════════════════════════════════════════════
    pexels_people_dir = DATA_DIR / "pexels_people"
    pexels_videos = scan_video_dir(pexels_people_dir)
    print(f"[pexels_people] Found {len(pexels_videos)} valid videos")

    for i, v in enumerate(pexels_videos):
        name = v["file_name"]
        # Check existing caption
        cap = captions_db.get(name, {})
        cap_short = cap.get("caption_short", "")
        cap_long = cap.get("caption_long", "")

        # Use VLM caption if available, else check existing, else template
        vlm = vlm_captions.get(name, {})
        if vlm.get("caption_short"):
            cap_short = vlm["caption_short"]
            cap_long = vlm.get("caption_long", cap_short)
        elif not cap_short or re.match(r"A \d+s video at \d+x\d+", cap_short):
            cap_short = generate_people_caption(seed * 1000 + i)
            cap_long = cap_short

        all_rows.append({
            **v,
            "audio_path": "",
            "caption_short": cap_short,
            "caption_long": cap_long,
            "caption_audio": "",
            "speaker_id": "",
            "dataset": "pexels_people",
        })
        already_added.add(name)
        stats["pexels_people"] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # 2. FILTERED/PEOPLE — 37 quality-filtered people videos
    # ═══════════════════════════════════════════════════════════════════════
    filtered_people_dir = DATA_DIR / "filtered" / "people"
    fp_videos = scan_video_dir(filtered_people_dir)
    print(f"[filtered/people] Found {len(fp_videos)} valid videos")

    for i, v in enumerate(fp_videos):
        name = v["file_name"]
        cap = captions_db.get(name, {})
        cap_short = cap.get("caption_short", "")
        cap_long = cap.get("caption_long", "")

        vlm = vlm_captions.get(name, {})
        if vlm.get("caption_short"):
            cap_short = vlm["caption_short"]
            cap_long = vlm.get("caption_long", cap_short)
        elif not cap_short or re.match(r"A \d+s video at \d+x\d+", cap_short):
            cap_short = generate_people_caption(seed * 2000 + i)
            cap_long = cap_short

        all_rows.append({
            **v,
            "audio_path": "",
            "caption_short": cap_short,
            "caption_long": cap_long,
            "caption_audio": "",
            "speaker_id": "",
            "dataset": "filtered_people",
        })
        already_added.add(name)
        stats["filtered_people"] += 1

    # ═══════════════════════════════════════════════════════════════════════
    # 2c. CELEBA_HQ — 30,000 high-res face pseudo-videos with real captions
    # ═══════════════════════════════════════════════════════════════════════
    celeba_manifest = DATA_DIR / "manifests" / "celeba_hq_train.csv"
    if celeba_manifest.exists():
        celeba_added = 0
        with open(celeba_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = Path(row["video_path"]).name
                if name in already_added:
                    continue
                # Verify file exists
                video_path = PROJECT_DIR / row["video_path"]
                if not video_path.exists():
                    continue
                all_rows.append({
                    "video_path": row["video_path"],
                    "num_frames": int(row["num_frames"]),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                    "fps": float(row["fps"]),
                    "duration_s": float(row["duration_s"]),
                    "audio_path": row.get("audio_path", ""),
                    "caption_short": row["caption_short"],
                    "caption_long": row["caption_long"],
                    "caption_audio": row.get("caption_audio", ""),
                    "speaker_id": "",
                    "dataset": "celeba_hq",
                })
                already_added.add(name)
                celeba_added += 1
        print(f"[celeba_hq] Added {celeba_added} videos (from {celeba_manifest})")
        stats["celeba_hq"] = celeba_added
    else:
        print(f"[celeba_hq] Manifest not found, run: python scripts/extract_celeba_hq.py")

    # ═══════════════════════════════════════════════════════════════════════
    # 2d. BILIBILI_PEOPLE — downloaded portrait/human videos from Bilibili
    # ═══════════════════════════════════════════════════════════════════════
    bili_people_dirs = [
        DATA_DIR / "people_bilibili_filtered",
        DATA_DIR / "people_bilibili",
    ]
    bili_videos = []
    for bd in bili_people_dirs:
        if bd.exists():
            bili_videos.extend(scan_video_dir(bd))
    bili_videos = list({v["file_name"]: v for v in bili_videos}.values())
    print(f"[bilibili_people] Found {len(bili_videos)} valid videos")

    bili_added = 0
    for i, v in enumerate(bili_videos):
        name = v["file_name"]
        if name in already_added:
            continue
        vlm = vlm_captions.get(name, {})
        if vlm.get("caption_short"):
            cap_short = vlm["caption_short"]
            cap_long = vlm.get("caption_long", cap_short)
        else:
            cap_short = generate_people_caption(seed * 5000 + i)
            cap_long = cap_short
        all_rows.append({
            **v,
            "audio_path": "",
            "caption_short": cap_short,
            "caption_long": cap_long,
            "caption_audio": "",
            "speaker_id": "",
            "dataset": "bilibili_people",
        })
        already_added.add(name)
        bili_added += 1
    print(f"[bilibili_people] Added {bili_added} videos")
    stats["bilibili_people"] = bili_added

    # ═══════════════════════════════════════════════════════════════════════
    # 3. PEOPLE-RELATED CAPTIONED VIDEOS — from captions.json
    # ═══════════════════════════════════════════════════════════════════════
    people_captioned = []
    for fname, cap in captions_db.items():
        s = (cap.get("caption_short", "") + " " + cap.get("caption_long", "")).lower()
        if _is_people_caption(s, min_kw=2):
            people_captioned.append((fname, cap))

    # Deduplicate against already added videos
    added_people_captioned = 0
    for fname, cap in people_captioned:
        if fname in already_added:
            continue
        # Find the actual video file
        found = False
        search_dirs = [
            DATA_DIR / "filtered",
            DATA_DIR / "webvid_filtered",
            DATA_DIR / "pexels_raw",
        ]
        for search_dir in search_dirs:
            video_path = search_dir / fname
            if not video_path.exists():
                # Try recursive search
                for candidate in search_dir.rglob(fname):
                    video_path = candidate
                    break
            if video_path.exists():
                meta = get_video_metadata(str(video_path.relative_to(PROJECT_DIR)))
                if meta:
                    all_rows.append({
                        **meta,
                        "audio_path": "",
                        "caption_short": cap.get("caption_short", ""),
                        "caption_long": cap.get("caption_long", ""),
                        "caption_audio": cap.get("caption_audio", ""),
                        "speaker_id": "",
                        "dataset": "people_captioned",
                    })
                    already_added.add(fname)
                    added_people_captioned += 1
                    found = True
                break
        if not found:
            # Try filtered subdirectories one level down
            for subdir in DATA_DIR.glob("filtered/*/"):
                video_path = subdir / fname
                if video_path.exists():
                    meta = get_video_metadata(str(video_path.relative_to(PROJECT_DIR)))
                    if meta:
                        all_rows.append({
                            **meta,
                            "audio_path": "",
                            "caption_short": cap.get("caption_short", ""),
                            "caption_long": cap.get("caption_long", ""),
                            "caption_audio": cap.get("caption_audio", ""),
                            "speaker_id": "",
                            "dataset": "people_captioned",
                        })
                        already_added.add(fname)
                        added_people_captioned += 1
                        found = True
                    break

    print(f"[people_captioned] Added {added_people_captioned} videos (from {len(people_captioned)} candidates)")
    stats["people_captioned"] = added_people_captioned

    # ═══════════════════════════════════════════════════════════════════════
    # 4. DIVERSE GENERAL VIDEOS — non-people for visual world knowledge
    # ═══════════════════════════════════════════════════════════════════════
    general_captioned = []
    for fname, cap in captions_db.items():
        s = (cap.get("caption_short", "") + " " + cap.get("caption_long", "")).lower()
        if not _is_people_caption(s, min_kw=1):  # Non-people videos
            general_captioned.append((fname, cap))

    rng.shuffle(general_captioned)
    added_general = 0
    for fname, cap in general_captioned:
        if added_general >= general_count:
            break
        if fname in already_added:
            continue
        for search_dir in [
            DATA_DIR / "filtered",
            DATA_DIR / "webvid_filtered",
            DATA_DIR / "pexels_raw",
        ]:
            video_path = search_dir / fname
            if not video_path.exists():
                for candidate in search_dir.rglob(fname):
                    video_path = candidate
                    break
            if video_path.exists():
                meta = get_video_metadata(str(video_path.relative_to(PROJECT_DIR)))
                if meta:
                    all_rows.append({
                        **meta,
                        "audio_path": "",
                        "caption_short": cap.get("caption_short", ""),
                        "caption_long": cap.get("caption_long", ""),
                        "caption_audio": cap.get("caption_audio", ""),
                        "speaker_id": "",
                        "dataset": "general_captioned",
                    })
                    already_added.add(fname)
                    added_general += 1
                break

    print(f"[general_captioned] Added {added_general} general videos")
    stats["general_captioned"] = added_general

    # Also add from pexels category directories (animals, city, food, nature, tech, travel)
    pexels_categories = ["animals", "city", "food", "nature", "tech", "travel"]
    pexels_total = 0
    for cat in pexels_categories:
        cat_dir = DATA_DIR / f"pexels_{cat}"
        if not cat_dir.exists():
            continue
        cat_videos = scan_video_dir(cat_dir)
        rng.shuffle(cat_videos)
        take = min(len(cat_videos), 80)  # Take up to 80 per category
        for v in cat_videos[:take]:
            name = v["file_name"]
            if name in already_added:
                continue
            cap = captions_db.get(name, {})
            cap_short = cap.get("caption_short", "")
            if not cap_short:
                cap_short = f"A video of {cat.replace('_', ' ')}."
            all_rows.append({
                **v,
                "audio_path": "",
                "caption_short": cap_short,
                "caption_long": cap.get("caption_long", cap_short),
                "caption_audio": cap.get("caption_audio", ""),
                "speaker_id": "",
                "dataset": f"pexels_{cat}",
            })
            already_added.add(name)
            pexels_total += 1
    print(f"[pexels_categories] Added {pexels_total} videos from 6 categories")
    stats["pexels_categories"] = pexels_total

    # ═══════════════════════════════════════════════════════════════════════
    # 5. WEBVID_FILTERED — 7,994 videos with Shutterstock captions
    # ═══════════════════════════════════════════════════════════════════════
    webvid_dir = DATA_DIR / "webvid_filtered"
    wv_videos = scan_video_dir(webvid_dir)
    print(f"[webvid_filtered] Found {len(wv_videos)} valid videos")

    # Classify into people vs non-people
    wv_people = []
    wv_non_people = []
    for v in wv_videos:
        name = v["file_name"]
        caption = webvid_meta.get(name, "")
        if _is_people_caption(caption, min_kw=1):
            wv_people.append((v, caption))
        else:
            wv_non_people.append((v, caption))

    print(f"[webvid_filtered] People-related: {len(wv_people)}, Non-people: {len(wv_non_people)}")

    # Add ALL people-related WebVid videos
    wv_people_added = 0
    for v, caption in wv_people:
        name = v["file_name"]
        if name in already_added:
            continue
        cap_short = caption if caption else f"A video of people."
        all_rows.append({
            **v,
            "audio_path": "",
            "caption_short": cap_short,
            "caption_long": cap_short,
            "caption_audio": "",
            "speaker_id": "",
            "dataset": "webvid_people",
        })
        already_added.add(name)
        wv_people_added += 1
    print(f"[webvid_people] Added {wv_people_added} videos")
    stats["webvid_people"] = wv_people_added

    # Add sampled non-people WebVid videos for visual diversity
    rng.shuffle(wv_non_people)
    wv_general_added = 0
    wv_general_target = general_count - stats.get("general_captioned", 0) - stats.get("pexels_categories", 0)
    wv_general_target = max(wv_general_target, 1000)  # At least 1000 more general videos
    for v, caption in wv_non_people:
        if wv_general_added >= wv_general_target:
            break
        name = v["file_name"]
        if name in already_added:
            continue
        cap_short = caption if caption else f"A video."
        all_rows.append({
            **v,
            "audio_path": "",
            "caption_short": cap_short,
            "caption_long": cap_short,
            "caption_audio": "",
            "speaker_id": "",
            "dataset": "webvid_general",
        })
        already_added.add(name)
        wv_general_added += 1
    print(f"[webvid_general] Added {wv_general_added} videos (target: {wv_general_target})")
    stats["webvid_general"] = wv_general_added

    # ═══════════════════════════════════════════════════════════════════════
    # 6. VOXCELEB2 — downsampled + enriched captions
    # ═══════════════════════════════════════════════════════════════════════
    vox_manifest = DATA_DIR / "manifests" / "train_stage1.csv"
    if vox_manifest.exists():
        vox_samples = []
        with open(vox_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("dataset", "") == "voxceleb2":
                    vox_samples.append(row)

        print(f"[voxceleb2] Total available: {len(vox_samples)}")
        rng.shuffle(vox_samples)
        take = min(len(vox_samples), vox_count)
        added_vox = 0
        for i, row in enumerate(vox_samples[:take]):
            video_path = row.get("video_path", "")
            # Try to find the actual file
            local_path = PROJECT_DIR / video_path
            if not local_path.exists():
                # Try fixing path (some manifests have Windows paths)
                fname = Path(video_path).name
                spk = row.get("speaker_id", "")
                if spk:
                    candidates = list((DATA_DIR / "voxceleb2").rglob(f"*{spk}*/{fname}"))
                else:
                    candidates = list((DATA_DIR / "voxceleb2").rglob(fname))
                if candidates:
                    local_path = candidates[0]

            if not local_path.exists():
                continue

            meta = get_video_metadata(str(local_path.relative_to(PROJECT_DIR)))
            if not meta:
                continue

            # Enrich caption with one of the diverse templates
            caption = rng.choice(VOX_TEMPLATES)
            speaker = row.get("speaker_id", "")

            all_rows.append({
                **meta,
                "audio_path": row.get("audio_path", ""),
                "caption_short": caption,
                "caption_long": caption,
                "caption_audio": row.get("caption_audio", ""),
                "speaker_id": speaker,
                "dataset": "voxceleb2",
            })
            added_vox += 1

        print(f"[voxceleb2] Added {added_vox} videos (downsampled from {len(vox_samples)})")
        stats["voxceleb2"] = added_vox
    else:
        print("[voxceleb2] Manifest not found, skipping")

    # ═══════════════════════════════════════════════════════════════════════
    # 7. ANIMATION — keep a small amount for motion diversity
    # ═══════════════════════════════════════════════════════════════════════
    anim_manifest = DATA_DIR / "manifests" / "train_animation.csv"
    if anim_manifest.exists():
        anim_samples = []
        with open(anim_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                anim_samples.append(row)

        rng.shuffle(anim_samples)
        take = min(len(anim_samples), 500)
        added_anim = 0
        for row in anim_samples[:take]:
            video_path = row.get("video_path", "")
            local_path = PROJECT_DIR / video_path
            if not local_path.exists():
                continue
            meta = get_video_metadata(str(local_path.relative_to(PROJECT_DIR)))
            if not meta:
                continue
            all_rows.append({
                **meta,
                "audio_path": row.get("audio_path", ""),
                "caption_short": row.get("caption_short", "An animated video"),
                "caption_long": row.get("caption_long", "An animated video"),
                "caption_audio": row.get("caption_audio", ""),
                "speaker_id": "",
                "dataset": "animation",
            })
            added_anim += 1
        print(f"[animation] Added {added_anim} videos")
        stats["animation"] = added_anim

    # ═══════════════════════════════════════════════════════════════════════
    # SHUFFLE, SPLIT, WRITE
    # ═══════════════════════════════════════════════════════════════════════
    # Remove internal-use-only 'file_name' key before writing
    for row in all_rows:
        row.pop("file_name", None)

    rng.shuffle(all_rows)

    split_idx = int(len(all_rows) * (1 - val_split))
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    output_base = str(output).replace(".csv", "")
    for split, data in [("train", train_rows), ("val", val_rows)]:
        out_path = f"{output_base}_{split}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"\n  [{split}] {len(data)} samples → {out_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"BALANCED MANIFEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(all_rows)} videos ({len(train_rows)} train / {len(val_rows)} val)")

    # Dataset distribution in training set
    train_dataset_dist = Counter(r["dataset"] for r in train_rows)
    print(f"\nTraining set composition:")
    for ds, count in train_dataset_dist.most_common():
        pct = count / len(train_rows) * 100
        print(f"  {ds:25s}: {count:>6d}  ({pct:5.1f}%)")

    # Resolution analysis
    res_dist = Counter()
    for r in train_rows:
        res = f"{r['width']}x{r['height']}"
        res_dist[res] += 1
    print(f"\nTop resolutions in training set:")
    for res, count in res_dist.most_common(10):
        pct = count / len(train_rows) * 100
        print(f"  {res:20s}: {count:>6d}  ({pct:5.1f}%)")

    # Caption diversity
    unique_captions = len(set(r["caption_short"] for r in train_rows))
    print(f"\nCaption diversity: {unique_captions} unique captions / {len(train_rows)} samples")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Build balanced training manifest")
    parser.add_argument("--vox_count", type=int, default=15000,
                        help="Number of VoxCeleb2 samples to keep (default: 15000)")
    parser.add_argument("--general_count", type=int, default=3000,
                        help="Number of general/diverse videos to include (default: 3000)")
    parser.add_argument("--output", type=str,
                        default="data/manifests/train_balanced.csv")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building balanced manifest...")
    print(f"  VoxCeleb2 target: {args.vox_count}")
    print(f"  General target:   {args.general_count}")
    print()

    build_balanced_manifest(
        vox_count=args.vox_count,
        general_count=args.general_count,
        output=args.output,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
