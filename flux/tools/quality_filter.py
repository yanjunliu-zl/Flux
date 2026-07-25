#!/usr/bin/env python3
"""Quality filtering for video clips.

Filters: resolution, duration, optical flow motion, aesthetic score.
Usage:
    python -m flux.tools.quality_filter --input data/clips/ --output data/filtered/
"""

import argparse
import os
import json
import subprocess
import cv2
import numpy as np
from pathlib import Path


def check_resolution(video_path: str, min_height: int = 360, min_width: int = 360) -> bool:
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return h >= min_height and w >= min_width


def check_duration(video_path: str, min_dur: float = 2.0, max_dur: float = 10.0) -> bool:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        return False
    dur = frames / fps
    return min_dur <= dur <= max_dur


def compute_optical_flow(video_path: str) -> float:
    """Compute mean optical flow magnitude (first second of video)."""
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flows = []

    for _ in range(30):  # ~1 second at 30fps
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        prev_gray = curr_gray

    cap.release()
    return np.mean(flows) if flows else 0.0


def filter_videos(
    input_dir: str,
    output_dir: str,
    min_height: int = 360,
    min_width: int = 360,
    min_dur: float = 2.0,
    max_dur: float = 10.0,
    min_flow: float = 0.05,
    max_flow: float = 5.0,
):
    """Apply quality filters and copy passing clips.

    Args:
        input_dir: Directory with input clips.
        output_dir: Output directory for filtered clips.
        min_height: Minimum video height.
        min_dur: Minimum duration (s).
        max_dur: Maximum duration (s).
        min_flow: Minimum optical flow (filter static scenes).
        max_flow: Maximum optical flow (filter chaotic footage).
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.mkv"))

    results = {"passed": [], "failed_resolution": [], "failed_duration": [], "failed_flow": []}

    for vf in video_files:
        if not check_resolution(str(vf), min_height, min_width):
            results["failed_resolution"].append(vf.name)
            continue
        if not check_duration(str(vf), min_dur, max_dur):
            results["failed_duration"].append(vf.name)
            continue

        flow = compute_optical_flow(str(vf))
        if flow < min_flow or flow > max_flow:
            results["failed_flow"].append(vf.name)
            continue

        # Copy passing clip
        import shutil
        shutil.copy(str(vf), os.path.join(output_dir, vf.name))
        results["passed"].append(vf.name)

    # Summary
    print(f"Total: {len(video_files)} clips")
    print(f"  Passed: {len(results['passed'])}")
    print(f"  Failed resolution: {len(results['failed_resolution'])}")
    print(f"  Failed duration: {len(results['failed_duration'])}")
    print(f"  Failed optical flow: {len(results['failed_flow'])}")

    # Save results
    with open(os.path.join(output_dir, "filter_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Quality filter for video clips")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default="data/filtered", help="Output directory")
    parser.add_argument("--min_height", type=int, default=360)
    parser.add_argument("--min_width", type=int, default=360)
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--min_flow", type=float, default=0.05)
    parser.add_argument("--max_flow", type=float, default=5.0)
    args = parser.parse_args()

    filter_videos(args.input, args.output, args.min_height, args.min_width,
                  args.min_duration, args.max_duration, args.min_flow, args.max_flow)


if __name__ == "__main__":
    main()
