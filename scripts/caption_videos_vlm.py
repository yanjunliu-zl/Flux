#!/usr/bin/env python3
"""Caption videos using Qwen2-VL for training data quality.

Replaces template captions on pexels_people, bilibili_people, and
filtered_people videos with real VLM-generated descriptions.

Usage:
    python scripts/caption_videos_vlm.py --model Qwen/Qwen2-VL-2B-Instruct
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CAPTION_PROMPT = """Describe this video in one English sentence. Focus on:
- The person/people visible (age, gender, appearance, expression)
- What they are doing (action, pose, movement)
- The setting (indoor/outdoor, background, lighting)
- Camera framing (close-up, medium shot, wide shot)

Be concise but specific. Do NOT start with "The video shows..." or "This is...". Just describe directly.

Example: "A young woman with long brown hair smiling at the camera in soft natural window light, medium close-up portrait."
"""


def sample_frames(video_path: Path, num_frames: int = 6) -> list:
    """Sample evenly-spaced frames from a video.

    Returns list of PIL Images.
    """
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        indices = list(range(total))
    else:
        step = total / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Resize to max 448px to avoid exceeding VLM token limit
            w, h = pil_img.size
            if max(w, h) > 448:
                ratio = 448 / max(w, h)
                pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            frames.append(pil_img)
    cap.release()
    return frames


def caption_video(
    video_path: Path,
    model,
    processor,
    num_frames: int = 6,
    max_new_tokens: int = 80,
) -> str:
    """Generate a caption for a single video using Qwen2-VL."""
    frames = sample_frames(video_path, num_frames)
    if not frames:
        return ""

    # Build message with images
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": CAPTION_PROMPT})

    messages = [{"role": "user", "content": content}]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = generated[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return caption


def main():
    parser = argparse.ArgumentParser(description="Caption videos with Qwen2-VL")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Qwen2-VL model ID (2B is fast, 7B is better)")
    parser.add_argument("--input", nargs="+",
                        default=[
                            "data/pexels_people",
                            "data/people_bilibili_filtered",
                            "data/filtered/people",
                        ])
    parser.add_argument("--output", default="data/manifests/vlm_captions.json")
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to caption (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing captions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Load model
    print(f"[Model] Loading {args.model}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[Model] Loaded, device: {model.device}")

    # Load existing captions
    output_path = Path(args.output)
    captions = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            captions = json.load(f)
        print(f"[Resume] {len(captions)} existing captions loaded")

    # Collect all videos to caption
    all_videos = []
    for input_dir in args.input:
        d = Path(input_dir)
        if not d.exists():
            print(f"[Skip] Directory not found: {input_dir}")
            continue
        for vf in d.glob("*.mp4"):
            if vf.name not in captions:
                all_videos.append(vf)

    print(f"[Videos] {len(all_videos)} to caption")

    if args.max_videos > 0:
        all_videos = all_videos[:args.max_videos]
        print(f"[Videos] Limited to {args.max_videos}")

    if not all_videos:
        print("Nothing to do.")
        return

    # Process
    start = time.time()
    for i, vf in enumerate(all_videos):
        try:
            caption = caption_video(vf, model, processor, args.num_frames)
            captions[vf.name] = {
                "caption_short": caption,
                "caption_long": caption,
                "caption_audio": "",
            }
        except Exception as e:
            print(f"  [{i+1}/{len(all_videos)}] ERROR {vf.name}: {e}")
            captions[vf.name] = {
                "caption_short": "A video of a person.",
                "caption_long": "A video of a person.",
                "caption_audio": "",
            }
            continue

        # Progress
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(all_videos) - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(all_videos)}] {rate:.1f}/s, ETA {eta:.0f}s | "
                  f"{vf.name}: {caption[:80]}...")

        # Save periodically
        if (i + 1) % 50 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"\nDone: {len(captions)} captions in {elapsed:.0f}s "
          f"({len(all_videos)/elapsed:.1f} videos/sec)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
