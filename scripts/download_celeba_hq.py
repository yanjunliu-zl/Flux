#!/usr/bin/env python3
"""Download CelebA-HQ (30,000 high-res face images, 1024x1024).

CelebA-HQ is the standard benchmark for face generation tasks.
Used here to teach the model what high-quality human portraits look like.

Sources tried in order:
1. HuggingFace datasets (preferred — resumable, fast)
2. Google Drive via gdown (fallback)

Usage:
    python scripts/download_celeba_hq.py [--output data/celeba_hq]
"""

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "celeba_hq"


def download_huggingface(output_dir: Path) -> bool:
    """Download from HuggingFace datasets hub."""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("  huggingface_hub not installed, run: pip install huggingface_hub")
        return False

    repo_id = "huggingface-CelebA-HQ"

    print(f"  Downloading {repo_id} from HuggingFace...")
    print(f"  This downloads ~13 GB of 1024x1024 face images.")
    print(f"  Target: {output_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            resume_download=True,
            max_workers=8,
        )
        return True
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return False


def download_gdrive(output_dir: Path) -> bool:
    """Fallback: download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("  gdown not installed, run: pip install gdown")
        return False

    # CelebA-HQ Google Drive links (official from authors)
    # The dataset is split across multiple files
    files = [
        # Main images (30,000 files, ~13 GB total)
        # Source: https://github.com/tkarras/progressive_growing_of_gans
    ]

    print("  Google Drive download not fully automated.")
    print("  Please use HuggingFace or download manually:")
    print("  https://github.com/tkarras/progressive_growing_of_gans")
    return False


def verify(output_dir: Path) -> dict:
    """Verify downloaded dataset."""
    images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png"))
    total = len(images)
    size_gb = sum(f.stat().st_size for f in images) / 1e9 if images else 0

    print(f"\n  Images found: {total}")
    print(f"  Total size: {size_gb:.1f} GB")

    if total >= 30000:
        print(f"  ✓ CelebA-HQ complete ({total} images)")
    elif total > 0:
        print(f"  ⚠ Partial download ({total}/30000)")
    else:
        print(f"  ✗ No images found")
        print(f"  Contents: {list(output_dir.iterdir())[:5]}")

    return {"count": total, "size_gb": size_gb}


def main():
    parser = argparse.ArgumentParser(description="Download CelebA-HQ")
    parser.add_argument("--output", default=str(DATA_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png"))
    if len(existing) >= 30000:
        print(f"CelebA-HQ already downloaded: {len(existing)} images in {output_dir}")
        return

    print("=" * 60)
    print("CelebA-HQ Downloader")
    print(f"  Target: {output_dir}")
    print(f"  Expected: ~30,000 images, ~13 GB")
    print("=" * 60)

    success = False
    for method in [download_huggingface, download_gdrive]:
        print(f"\n--- Trying: {method.__name__} ---")
        try:
            success = method(output_dir)
            if success:
                break
        except Exception as e:
            print(f"  Failed: {e}")

    verify(output_dir)

    if not success:
        print(f"\nAutomatic download failed.")
        print(f"Manual options:")
        print(f"  1. huggingface-cli download --repo-type dataset huggingface-CelebA-HQ {output_dir}")
        print(f"  2. kaggle datasets download -d lamsimon/celebahq -p {output_dir}")


if __name__ == "__main__":
    main()
