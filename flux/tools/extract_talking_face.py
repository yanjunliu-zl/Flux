#!/usr/bin/env python3
"""Talking-face preprocessing for lip-sync training data.

Extracts face crops, mouth ROIs, and viseme labels from video clips
containing speaking faces. Produces enriched training data for the
lip-sync components (LipSyncBridge, viseme loss, mouth-audio loss).

Pipeline:
  1. Face detection (OpenCV Haar / MediaPipe fallback)
  2. Mouth ROI bounding box extraction
  3. Audio phoneme transcription via Wav2Vec2
  4. Phoneme → viseme mapping (14-class MPEG-4 set)
  5. Output: enriched manifest with face/mouth/viseme annotations

Usage:
    python -m flux.tools.extract_talking_face \
        --manifest data/manifests/train.csv \
        --output data/manifests/talking_train.csv \
        --compute_visemes \
        --face_min_size 80

Requirements for best results:
    pip install mediapipe transformers torchaudio
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Phoneme → viseme mapping (ARPABET phoneme set)
# ---------------------------------------------------------------------------
PHONEME_TO_VISEME = {
    # Silence
    "sil": 0, "sp": 0, "spn": 0, "": 0,
    # Bilabial: p, b, m
    "p": 1, "b": 1, "m": 1, "pcl": 1, "bcl": 1,
    # Labiodental: f, v
    "f": 2, "v": 2,
    # Dental: th, dh
    "th": 3, "dh": 3,
    # Alveolar: t, d, s, z, n, l
    "t": 4, "d": 4, "s": 4, "z": 4, "n": 4, "l": 4, "tcl": 4, "dcl": 4,
    # Postalveolar: sh, zh, ch, jh
    "sh": 5, "zh": 5, "ch": 5, "jh": 5,
    # Velar: k, g, ng
    "k": 6, "g": 6, "ng": 6, "kcl": 6, "gcl": 6,
    # Rounded glide: w, r
    "w": 7, "r": 7,
    # High front spread: iy, ih
    "iy": 8, "ih": 8,
    # Mid front: ey, eh
    "ey": 9, "eh": 9,
    # Low front: ae, aa
    "ae": 10, "aa": 10,
    # Rounded mid back: ao, ow
    "ao": 11, "ow": 11,
    # Rounded high back: uh, uw
    "uh": 12, "uw": 12,
    # Neutral open: ah, ax, er, ay, aw, oy, others
    "ah": 13, "ax": 13, "er": 13, "ay": 13, "aw": 13, "oy": 13,
    "hh": 13, "y": 13, "el": 13, "em": 13, "en": 13, "eng": 13,
    "dx": 13, "nx": 13, "hv": 13, "epi": 13,
}


def detect_face_and_mouth(
    frame: np.ndarray,
    min_face_size: int = 80,
) -> Optional[dict]:
    """Detect face and extract mouth bounding box using SCRFD.

    Production-grade detection via InsightFace SCRFD-10G with
    5-point landmarks for accurate mouth region estimation.

    Args:
        frame: RGB image (H, W, 3), uint8.
        min_face_size: Minimum face size in pixels (used as fallback threshold).

    Returns:
        Dict with keys: face_bbox, mouth_bbox, face_present.
        Bboxes in normalized [0, 1] coordinates (x1, y1, x2, y2).
        None if no face detected.
    """
    from flux.models.face_analysis import get_face_analyzer

    H, W = frame.shape[:2]
    analyzer = get_face_analyzer()
    face = analyzer.detect_largest(frame)

    if face is None:
        return None

    # Face bbox (already in pixel coords, normalize)
    fx1, fy1, fx2, fy2 = face.bbox
    face_h = fy2 - fy1

    if face_h < min_face_size:
        return None

    x1, y1 = fx1 / W, fy1 / H
    x2, y2 = fx2 / W, fy2 / H

    # Use 5-point landmark mouth corners for accurate mouth bbox
    if face.mouth_bbox is not None:
        mx1, my1, mx2, my2 = face.mouth_bbox
        mx1, my1 = mx1 / W, my1 / H
        mx2, my2 = mx2 / W, my2 / H
    else:
        # Fallback: heuristic based on face proportions
        face_h_norm = y2 - y1
        mx1 = x1 + 0.15
        my1 = y1 + face_h_norm * 0.55
        mx2 = x2 - 0.15
        my2 = y1 + face_h_norm * 0.90

    return {
        "face_bbox": [max(0, x1), max(0, y1), min(1, x2), min(1, y2)],
        "mouth_bbox": [max(0, mx1), max(0, my1), min(1, mx2), min(1, my2)],
        "face_present": True,
    }


def compute_viseme_labels(
    audio_path: str,
    num_frames: int,
    fps: float = 16.0,
) -> Optional[list[int]]:
    """Compute per-frame viseme labels from audio.

    Uses Wav2Vec2 for phoneme recognition, then maps phonemes to visemes.

    Args:
        audio_path: Path to audio file (WAV, 16kHz mono).
        num_frames: Number of video frames.
        fps: Video FPS.

    Returns:
        List of viseme class indices (length = num_frames), or None if failed.
    """
    try:
        import torch
        import torchaudio

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Try Wav2Vec2 phoneme recognition
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

            processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-lv-60-espeak-cv-ft"
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-lv-60-espeak-cv-ft"
            )

            inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000,
                              return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                phonemes = processor.batch_decode(predicted_ids)[0]

            # Parse phoneme string (whitespace-separated ARPABET-like tokens)
            phoneme_list = phonemes.strip().split()

            # Map to visemes
            viseme_seq = [PHONEME_TO_VISEME.get(p.lower(), 0) for p in phoneme_list]

            # Resample to match video frames
            if len(viseme_seq) > 0:
                indices = np.linspace(0, len(viseme_seq) - 1, num_frames, dtype=int)
                frame_visemes = [viseme_seq[i] for i in indices]
            else:
                frame_visemes = [0] * num_frames

            return frame_visemes

        except Exception:
            # Fallback: energy-based heuristic (speech vs silence)
            frame_len = waveform.shape[1] // num_frames
            energies = []
            for i in range(num_frames):
                start = i * frame_len
                end = min(start + frame_len, waveform.shape[1])
                energy = waveform[:, start:end].abs().mean().item()
                energies.append(energy)

            # Threshold: above median = speech (viseme 13, neutral open),
            # below = silence (viseme 0)
            median_energy = np.median(energies) if energies else 0
            frame_visemes = [13 if e > median_energy else 0 for e in energies]
            return frame_visemes

    except Exception:
        return None


def process_talking_face_manifest(
    input_manifest: str,
    output_manifest: str,
    compute_visemes: bool = True,
    face_min_size: int = 80,
    sample_every: int = 1,
    max_samples: Optional[int] = None,
):
    """Process manifest: detect faces, compute mouth ROIs and viseme labels.

    Args:
        input_manifest: Input CSV manifest.
        output_manifest: Output CSV with added columns.
        compute_visemes: Whether to run Wav2Vec2 viseme labeling.
        face_min_size: Minimum detectable face size.
        sample_every: Process every Nth row (for quick testing).
        max_samples: Max samples to process.
    """
    import cv2

    # Load input manifest
    with open(input_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples:
        rows = rows[:max_samples]
    rows = rows[::sample_every]

    fieldnames = list(rows[0].keys()) + [
        "face_present", "face_bbox_x1", "face_bbox_y1", "face_bbox_x2", "face_bbox_y2",
        "mouth_bbox_x1", "mouth_bbox_y1", "mouth_bbox_x2", "mouth_bbox_y2",
        "viseme_labels",
    ]

    output_rows = []
    face_count = 0
    viseme_count = 0

    print(f"[TalkingFace] Processing {len(rows)} entries...")

    for i, row in enumerate(rows):
        video_path = row.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            output_rows.append(row)
            continue

        # Extract middle frame for face detection
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            output_rows.append(row)
            continue

        # Read middle frame
        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            output_rows.append(row)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face + mouth detection
        face_data = detect_face_and_mouth(frame_rgb, min_size=face_min_size)

        if face_data:
            face_count += 1
            row["face_present"] = "1"
            row["face_bbox_x1"] = str(face_data["face_bbox"][0])
            row["face_bbox_y1"] = str(face_data["face_bbox"][1])
            row["face_bbox_x2"] = str(face_data["face_bbox"][2])
            row["face_bbox_y2"] = str(face_data["face_bbox"][3])
            row["mouth_bbox_x1"] = str(face_data["mouth_bbox"][0])
            row["mouth_bbox_y1"] = str(face_data["mouth_bbox"][1])
            row["mouth_bbox_x2"] = str(face_data["mouth_bbox"][2])
            row["mouth_bbox_y2"] = str(face_data["mouth_bbox"][3])
        else:
            row["face_present"] = "0"
            for key in ["face_bbox_x1", "face_bbox_y1", "face_bbox_x2", "face_bbox_y2",
                        "mouth_bbox_x1", "mouth_bbox_y1", "mouth_bbox_x2", "mouth_bbox_y2"]:
                row[key] = ""

        # Viseme labels from audio
        if compute_visemes:
            audio_path = row.get("audio_path", "")
            num_frames = int(row.get("num_frames", 32))
            if audio_path and Path(audio_path).exists():
                visemes = compute_viseme_labels(audio_path, num_frames, fps)
            elif video_path:
                visemes = compute_viseme_labels(video_path, num_frames, fps)
            else:
                visemes = None

            if visemes is not None:
                viseme_count += 1
                row["viseme_labels"] = ",".join(str(v) for v in visemes)
            else:
                row["viseme_labels"] = ""

        output_rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] faces: {face_count}, visemes: {viseme_count}")

    # Write output
    with open(output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    pct_face = 100 * face_count / len(output_rows) if output_rows else 0
    pct_viseme = 100 * viseme_count / len(output_rows) if output_rows else 0
    print(f"\n[TalkingFace] Done: {len(output_rows)} entries")
    print(f"  Faces detected: {face_count} ({pct_face:.1f}%)")
    print(f"  Visemes computed: {viseme_count} ({pct_viseme:.1f}%)")
    print(f"  Output: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description="Talking-face preprocessing for lip-sync")
    parser.add_argument("--manifest", type=str, required=True, help="Input CSV manifest")
    parser.add_argument("--output", type=str, required=True, help="Output CSV manifest")
    parser.add_argument("--compute_visemes", action="store_true",
                        help="Compute viseme labels from audio (needs Wav2Vec2)")
    parser.add_argument("--face_min_size", type=int, default=80,
                        help="Minimum face detection size")
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Process every Nth entry")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process")
    args = parser.parse_args()

    process_talking_face_manifest(
        input_manifest=args.manifest,
        output_manifest=args.output,
        compute_visemes=args.compute_visemes,
        face_min_size=args.face_min_size,
        sample_every=args.sample_every,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
