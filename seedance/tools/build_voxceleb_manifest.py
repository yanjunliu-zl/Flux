#!/usr/bin/env python3
"""Fast VoxCeleb2 manifest builder — avoids per-file OpenCV probing for 1M+ clips.

Usage:
    python -m seedance.tools.build_voxceleb_manifest \
        --input_dir data/voxceleb2/dev/mp4/ \
        --output data/manifests/voxceleb_manifest.csv
"""

import argparse
import csv
import os
from pathlib import Path


def build_manifest_fast(input_dir: str, output_csv: str, max_files: int = 0):
    """Walk the VoxCeleb2 directory tree and write a manifest CSV.

    Directory structure: idXXXXX/YYYYYYY/00001.mp4
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path", "num_frames", "height", "width",
                "fps", "duration_s", "audio_path",
                "caption_short", "caption_long", "caption_audio",
                "speaker_id", "dataset", "face_present",
            ],
        )
        writer.writeheader()

        for dirpath, dirnames, filenames in os.walk(input_path):
            # Only process leaf directories (those containing .mp4 files)
            mp4_files = [fn for fn in filenames if fn.endswith(".mp4")]
            if not mp4_files:
                continue

            # Parse speaker_id from path: .../idXXXXX/YYYYYYY/00001.mp4
            rel = os.path.relpath(dirpath, input_path)
            parts = rel.replace("\\", "/").split("/")
            # parts = ["idXXXXX", "YYYYYYY"]
            speaker_id = parts[0] if len(parts) >= 1 and parts[0].startswith("id") else "unknown"

            for fn in sorted(mp4_files):
                abs_path = os.path.join(dirpath, fn)
                row = {
                    "video_path": abs_path,
                    "num_frames": 0,       # read at load time by dataset
                    "height": 0,
                    "width": 0,
                    "fps": 25,             # VoxCeleb2 clips are ~25fps
                    "duration_s": 0,
                    "audio_path": abs_path,  # audio is embedded in video
                    "caption_short": "A person speaking",
                    "caption_long": "",
                    "caption_audio": "",
                    "speaker_id": speaker_id,
                    "dataset": "voxceleb2",
                    "face_present": "0",
                }
                writer.writerow(row)
                count += 1

                if max_files and count >= max_files:
                    print(f"[VoxCeleb] Reached --max_files={max_files}, stopping at {count} entries")
                    return count

                if count % 100000 == 0:
                    print(f"  [{count:,} entries written...]")

    return count


def main():
    parser = argparse.ArgumentParser(description="Fast VoxCeleb2 manifest builder")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root of VoxCeleb2 MP4 tree (e.g. data/voxceleb2/dev/mp4/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max entries to write (0 = all)")
    args = parser.parse_args()

    print(f"[VoxCeleb] Scanning {args.input_dir} ...")
    count = build_manifest_fast(args.input_dir, args.output, args.max_files)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n[VoxCeleb] Done: {count:,} entries → {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
