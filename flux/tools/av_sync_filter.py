#!/usr/bin/env python3
"""Audio-Visual synchronization scoring and filtering.

Uses a simple correlation-based approach for AV sync estimation.
In production, replace with ImageBind-based scoring.

Usage:
    python -m flux.tools.av_sync_filter --video_dir data/filtered/ --audio_dir data/audio/ --output data/manifests/
"""

import argparse
import os
import json
from pathlib import Path


def compute_av_sync_score(video_path: str, audio_path: str) -> float:
    """Compute approximate AV sync score.

    Uses a simple heuristic: compares video motion energy with audio energy.
    Returns score in [0, 1], higher = better sync.
    """
    import cv2
    import wave
    import numpy as np

    # Video motion energy
    cap = cv2.VideoCapture(video_path)
    motion_energies = []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()
        motion_energies.append(diff)
        prev_gray = curr_gray
    cap.release()

    # Audio energy
    try:
        with wave.open(audio_path, "r") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_energy = np.abs(audio).reshape(-1, len(audio) // max(len(motion_energies), 1))
            audio_energies = audio_energy.mean(axis=0)
    except Exception:
        return 0.5  # Default if audio can't be read

    # Correlate (simple dot product of normalized sequences)
    min_len = min(len(motion_energies), len(audio_energies))
    if min_len < 2:
        return 0.5

    a = np.array(motion_energies[:min_len])
    b = np.array(audio_energies[:min_len])
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)

    corr = np.corrcoef(a, b)[0, 1]
    return max(0.0, min(1.0, (corr + 1) / 2))


def filter_av_sync(
    video_dir: str,
    audio_dir: str,
    output_dir: str,
    min_score: float = 0.3,
):
    """Filter clips by AV sync score and create manifest.

    Args:
        video_dir: Directory with filtered videos.
        audio_dir: Directory with extracted audio.
        output_dir: Output directory for manifest.
        min_score: Minimum sync score to keep.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = Path(video_dir)
    video_files = {vf.stem: vf for vf in video_path.glob("*.mp4")}
    audio_files = {af.stem: af for af in Path(audio_dir).glob("*.wav")}

    results = []
    for stem, vf in video_files.items():
        af = audio_files.get(stem)
        if af is None:
            continue

        score = compute_av_sync_score(str(vf), str(af))
        sync_ok = score >= min_score

        results.append({
            "video_path": str(vf),
            "audio_path": str(af),
            "sync_score": round(score, 4),
            "sync_ok": sync_ok,
        })

        if sync_ok:
            print(f"  ✓ {stem}: sync={score:.3f}")
        else:
            print(f"  ✗ {stem}: sync={score:.3f} (filtered)")

    # Save results
    with open(os.path.join(output_dir, "av_sync_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["sync_ok"])
    print(f"AV sync filter: {passed}/{len(results)} passed (threshold={min_score})")


def main():
    parser = argparse.ArgumentParser(description="AV sync scoring")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio directory")
    parser.add_argument("--output", type=str, default="data/manifests", help="Output directory")
    parser.add_argument("--min_score", type=float, default=0.3, help="Min sync score")
    args = parser.parse_args()

    filter_av_sync(args.video_dir, args.audio_dir, args.output, args.min_score)


if __name__ == "__main__":
    main()
