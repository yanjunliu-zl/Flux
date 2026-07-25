"""End-to-end annotation pipeline: tag → score → filter → curate.

Ties together physics event detection, scenario classification, and motion
quality scoring into a unified pipeline that produces enriched training manifests.

Usage:
    python -m flux.data.annotation.pipeline \
        --input data/manifests/train.csv \
        --output data/manifests/train_annotated.csv \
        --sample 100
"""

import csv
import os
import json
import argparse
import time
from pathlib import Path
from collections import Counter

from flux.data.annotation.physics_events import (
    PhysicsEventDetector,
)
from flux.data.annotation.scenario_classifier import (
    ScenarioClassifier,
)
from flux.data.annotation.motion_quality import (
    MotionQualityScorer,
)


class AnnotationPipeline:
    """End-to-end annotation pipeline for physics-aware data curation.

    Processes a CSV manifest and enriches it with:
    - physics_event:    Dominant physical event type
    - physics_richness: How much physics is present [0-1]
    - scenario:         Video scenario category
    - scenario_conf:    Classification confidence
    - motion_smoothness/naturalness/diversity/stability/complexity
    - motion_overall:   Overall motion quality [0-1]
    - training_ready:   Whether this sample is recommended for training
    - caption_prefix:   Suggested caption prefix based on scenario

    Args:
        sample_frames: Frames to analyze per video (default: 16).
        output_format: "csv" or "json".
    """

    def __init__(
        self,
        sample_frames: int = 16,
        output_format: str = "csv",
    ):
        self.physics_detector = PhysicsEventDetector(sample_frames=sample_frames)
        self.scenario_classifier = ScenarioClassifier(sample_frames=sample_frames)
        self.motion_scorer = MotionQualityScorer(sample_frames=sample_frames)
        self.output_format = output_format

    def annotate_single(self, video_path: str) -> dict:
        """Annotate a single video with all metadata.

        Args:
            video_path: Path to video file.

        Returns:
            Dict with all annotation fields.
        """
        annotation = {}

        # Physics events
        physics = self.physics_detector.detect(video_path)
        annotation["physics_event"] = physics.dominant_event
        annotation["physics_richness"] = physics.physics_richness
        annotation["physics_recommended"] = physics.recommended_for_physics_training
        for event, score in physics.events.items():
            annotation[f"physics_{event}"] = score

        # Scenario
        scenario = self.scenario_classifier.classify(video_path)
        annotation["scenario"] = scenario.scenario
        annotation["scenario_confidence"] = scenario.confidence
        annotation["caption_prefix"] = scenario.suggested_caption_prefix

        # Motion quality
        motion = self.motion_scorer.score(video_path)
        annotation["motion_smoothness"] = motion.smoothness
        annotation["motion_naturalness"] = motion.naturalness
        annotation["motion_diversity"] = motion.diversity
        annotation["motion_stability"] = motion.stability
        annotation["motion_complexity"] = motion.complexity
        annotation["motion_overall"] = motion.overall
        annotation["training_ready"] = motion.is_training_ready and physics.physics_richness > 0.15

        return annotation

    def process_manifest(
        self,
        input_csv: str,
        output_path: str,
        max_videos: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """Process a full CSV manifest and produce enriched output.

        Args:
            input_csv: Path to input CSV manifest.
            output_path: Path for enriched output (CSV or JSON).
            max_videos: Maximum videos to process (for quick sampling).
            verbose: Print progress.

        Returns:
            Dict with processing statistics.
        """
        # Load manifest
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if max_videos:
            import random
            random.seed(42)
            rows = random.sample(rows, min(max_videos, len(rows)))

        total = len(rows)
        if verbose:
            print(f"[Annotation] Processing {total} videos...")

        enriched = []
        stats = Counter()
        start = time.time()

        for i, row in enumerate(rows):
            video_path = row.get("video_path", "")
            if not os.path.exists(video_path):
                stats["missing"] += 1
                continue

            # Annotate
            ann = self.annotate_single(video_path)

            # Merge with original row
            enriched_row = {**row, **ann}
            enriched.append(enriched_row)

            # Stats
            stats[f"scenario_{ann['scenario']}"] += 1
            stats[f"physics_{ann['physics_event']}"] += 1
            if ann["training_ready"]:
                stats["training_ready"] += 1
            else:
                stats["training_filtered"] += 1

            if verbose and (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / max(elapsed, 1)
                print(f"  [{i+1}/{total}] {rate:.1f} vids/s | ready={stats['training_ready']} "
                      f"filtered={stats['training_filtered']}")

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        new_fields = list(enriched[0].keys()) if enriched else []
        if self.output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=new_fields)
                writer.writeheader()
                writer.writerows(enriched)

        elapsed = time.time() - start
        if verbose:
            print(f"\n[Annotation] Done in {elapsed:.0f}s: {len(enriched)} annotated")
            print(f"  Training-ready: {stats['training_ready']} "
                  f"({stats['training_ready']/max(len(enriched),1)*100:.0f}%)")
            print(f"  Filtered: {stats['training_filtered']}")
            print(f"\n  Top scenarios:")
            for s, c in Counter({k: v for k, v in stats.items() if k.startswith("scenario_")}).most_common(5):
                print(f"    {s.replace('scenario_','')}: {c}")
            print(f"\n  Output: {output_path}")

        return {
            "total": total,
            "annotated": len(enriched),
            "training_ready": stats["training_ready"],
            "training_filtered": stats["training_filtered"],
            "elapsed_s": elapsed,
        }


def annotate_manifest(
    input_csv: str,
    output_path: str,
    max_videos: int | None = None,
    sample_frames: int = 16,
    verbose: bool = True,
) -> dict:
    """Convenience function: annotate a manifest in one call.

    Args:
        input_csv: Input manifest CSV.
        output_path: Output enriched CSV.
        max_videos: Max videos to process.
        sample_frames: Frames per video.
        verbose: Print progress.

    Returns:
        Processing statistics dict.
    """
    pipeline = AnnotationPipeline(sample_frames=sample_frames)
    return pipeline.process_manifest(input_csv, output_path, max_videos, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate video manifest with physics, scenario, and motion metadata"
    )
    parser.add_argument("--input", type=str, required=True, help="Input manifest CSV")
    parser.add_argument("--output", type=str, default="data/manifests/train_annotated.csv",
                        help="Output enriched CSV")
    parser.add_argument("--sample", type=int, default=None, help="Max videos to process")
    parser.add_argument("--frames", type=int, default=16, help="Frames per video")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"])
    args = parser.parse_args()

    annotate_manifest(
        input_csv=args.input,
        output_path=args.output,
        max_videos=args.sample,
        sample_frames=args.frames,
    )


if __name__ == "__main__":
    main()
