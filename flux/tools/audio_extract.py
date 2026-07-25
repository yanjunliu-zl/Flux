#!/usr/bin/env python3
"""Audio extraction from video files using FFmpeg.

Usage:
    python -m flux.tools.audio_extract --input data/filtered/ --output data/audio/
"""

import argparse
import os
import subprocess
from pathlib import Path


def extract_audio(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    channels: int = 1,
):
    """Extract audio tracks from video files.

    Args:
        input_dir: Directory with video files.
        output_dir: Output directory for WAV files.
        sample_rate: Target sample rate.
        channels: Number of audio channels (1 = mono).
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    video_files = list(input_path.glob("*.mp4"))

    extracted = 0
    skipped = 0

    for vf in video_files:
        output_file = os.path.join(output_dir, f"{vf.stem}.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(vf),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(output_file):
            # Check for silence
            import wave
            import numpy as np
            try:
                with wave.open(output_file, "r") as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(audio**2))
                    if rms < 100:  # Silent
                        os.remove(output_file)
                        skipped += 1
                        continue
            except Exception:
                pass
            extracted += 1
        else:
            skipped += 1

    print(f"Extracted: {extracted} audio files, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio from videos")
    parser.add_argument("--input", type=str, required=True, help="Input video directory")
    parser.add_argument("--output", type=str, default="data/audio", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    extract_audio(args.input, args.output, args.sample_rate)


if __name__ == "__main__":
    main()
