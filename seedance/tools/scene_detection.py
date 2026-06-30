#!/usr/bin/env python3
"""Scene detection for video segmentation using PySceneDetect.

Usage:
    python -m seedance.tools.scene_detection --input data/raw_videos/ --output data/clips/
"""

import argparse
import os
from pathlib import Path


def detect_scenes(
    input_dir: str,
    output_dir: str,
    min_duration: float = 2.0,
    max_duration: float = 10.0,
    threshold: float = 27.0,
):
    """Detect scene boundaries and split videos into clips.

    Args:
        input_dir: Directory containing input videos.
        output_dir: Output directory for clips.
        min_duration: Minimum clip duration (seconds).
        max_duration: Maximum clip duration (seconds).
        threshold: PySceneDetect ContentDetector threshold.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        print("Install scenedetect: pip install scenedetect[opencv]")
        return

    os.makedirs(output_dir, exist_ok=True)

    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    input_path = Path(input_dir)
    video_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in video_extensions
    ]

    total_clips = 0
    for video_file in video_files:
        print(f"Processing: {video_file.name}")

        try:
            video = open_video(str(video_file))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            for i, (start, end) in enumerate(scene_list):
                duration = end.get_seconds() - start.get_seconds()
                if min_duration <= duration <= max_duration:
                    clip_name = f"{video_file.stem}_s{i:03d}.mp4"
                    clip_path = os.path.join(output_dir, clip_name)

                    # Use FFmpeg to extract clip
                    import subprocess
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-ss", str(start.get_seconds()),
                        "-i", str(video_file),
                        "-t", str(duration),
                        "-c:v", "libx264", "-c:a", "aac",
                        clip_path,
                    ], capture_output=True)
                    total_clips += 1
        except Exception as e:
            print(f"  ⚠ Error: {e}")

    print(f"Extracted {total_clips} clips to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Scene detection for video clips")
    parser.add_argument("--input", type=str, required=True, help="Input video directory")
    parser.add_argument("--output", type=str, default="data/clips", help="Output directory")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Min clip duration (s)")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max clip duration (s)")
    parser.add_argument("--threshold", type=float, default=27.0, help="Scene detection threshold")
    args = parser.parse_args()

    detect_scenes(args.input, args.output, args.min_duration, args.max_duration, args.threshold)


if __name__ == "__main__":
    main()
