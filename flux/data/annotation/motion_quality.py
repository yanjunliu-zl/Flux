"""Motion quality scoring for training data curation.

Scores videos on 5 motion quality dimensions relevant to physical realism:
1. Smoothness   — temporal consistency of motion (no jitter)
2. Naturalness  — how "human-like" the motion patterns are
3. Diversity    — variety of motion across the video
4. Stability    — background stability (no camera shake)
5. Complexity   — overall motion complexity (too simple = boring, too complex = chaotic)

All scores in [0, 1]. Higher = better quality for training data.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class MotionScore:
    """Motion quality scores for a video."""
    video_path: str
    smoothness: float = 0.0
    naturalness: float = 0.0
    diversity: float = 0.0
    stability: float = 0.0
    complexity: float = 0.0
    overall: float = 0.0

    # Training recommendations
    is_training_ready: bool = False
    issues: list[str] = field(default_factory=list)


class MotionQualityScorer:
    """Score video motion quality using optical flow analysis.

    Detects common motion issues:
    - Camera shake (high-frequency global motion)
    - Motion jitter (temporal inconsistency)
    - Static scenes (no training value)
    - Chaotic motion (training noise)

    Args:
        sample_frames: Number of frames to analyze (default: 16).
        shake_threshold: Global motion variance threshold for shake detection.
        static_threshold: Minimum motion for training value.
        chaos_threshold: Maximum motion variance before "chaotic" label.
    """

    def __init__(
        self,
        sample_frames: int = 16,
        shake_threshold: float = 15.0,
        static_threshold: float = 2.0,
        chaos_threshold: float = 80.0,
    ):
        self.sample_frames = sample_frames
        self.shake_threshold = shake_threshold
        self.static_threshold = static_threshold
        self.chaos_threshold = chaos_threshold

    def score(self, video_path: str) -> MotionScore:
        """Score a single video's motion quality.

        Args:
            video_path: Path to video file.

        Returns:
            MotionScore with per-dimension scores and training readiness.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return MotionScore(video_path=video_path, issues=["cannot_open"])

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            cap.release()
            return MotionScore(video_path=video_path, issues=["too_few_frames"])

        # Sample frames
        indices = np.linspace(0, total_frames - 2, min(self.sample_frames, total_frames - 1), dtype=int)

        prev_gray = None
        flow_magnitudes = []      # Per-frame mean flow
        flow_variances = []       # Per-frame flow variance
        global_motions = []       # Global (camera) motion estimate
        local_flows_grid = []     # Grid of local flow magnitudes (4x4 grid)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_magnitudes.append(mag.mean())
                flow_variances.append(mag.var())
                # Global motion = mean of flow vectors
                global_motions.append(np.sqrt(flow[..., 0].mean()**2 + flow[..., 1].mean()**2))

                # Local flow grid (4×4)
                h, w = mag.shape
                grid = mag.reshape(4, h//4, 4, w//4).mean(axis=(1, 3))
                local_flows_grid.append(grid.flatten())

            prev_gray = gray

        cap.release()

        if not flow_magnitudes:
            return MotionScore(video_path=video_path, issues=["no_flow"])

        mags = np.array(flow_magnitudes)
        vars_ = np.array(flow_variances)
        globs = np.array(global_motions) if global_motions else np.zeros_like(mags)

        # ── Score dimensions ──────────────────────────────────────

        # 1. Smoothness: low temporal variance in flow magnitude
        temporal_roughness = np.abs(np.diff(mags)).mean() / max(mags.mean(), 1e-8)
        smoothness = np.exp(-temporal_roughness)  # [0, 1]

        # 2. Naturalness: flow distribution similar to natural motion
        # Natural motion has moderate flow values, not uniform or extreme
        naturalness = 1.0 - abs(mags.mean() / 20.0 - 0.5)
        naturalness = np.clip(naturalness, 0, 1)

        # 3. Diversity: high spatial variation across the grid
        if local_flows_grid:
            grid_stack = np.array(local_flows_grid)  # (T, 16)
            spatial_diversity = grid_stack.std(axis=0).mean() / max(grid_stack.mean(), 1e-8)
            diversity = min(1.0, spatial_diversity)
        else:
            diversity = 0.5

        # 4. Stability: low global motion variance (no camera shake)
        shake_indicator = globs.std() / max(globs.mean(), 1e-8)
        stability = np.exp(-shake_indicator / self.shake_threshold)

        # 5. Complexity: moderate overall motion (not too simple, not too chaotic)
        complexity_raw = mags.mean() / 20.0
        # Gaussian around "good" complexity centered at 0.5
        complexity = np.exp(-((complexity_raw - 0.5) ** 2) / 0.1)

        # ── Overall ───────────────────────────────────────────────
        overall = (smoothness + naturalness + diversity + stability + complexity) / 5.0

        # ── Issues ────────────────────────────────────────────────
        issues = []
        if smoothness < 0.3:
            issues.append("motion_jitter")
        if naturalness < 0.3:
            issues.append("unnatural_motion")
        if diversity < 0.3:
            issues.append("low_motion_diversity")
        if stability < 0.3:
            issues.append("camera_shake")
        if complexity < 0.3:
            issues.append("too_simple_or_chaotic")

        is_ready = len(issues) <= 1  # Allow at most 1 minor issue

        return MotionScore(
            video_path=video_path,
            smoothness=round(float(smoothness), 3),
            naturalness=round(float(naturalness), 3),
            diversity=round(float(diversity), 3),
            stability=round(float(stability), 3),
            complexity=round(float(complexity), 3),
            overall=round(float(overall), 3),
            is_training_ready=is_ready,
            issues=issues,
        )


def score_motion_quality(
    video_paths: list[str],
    verbose: bool = False,
) -> list[MotionScore]:
    """Batch-score motion quality for a list of videos."""
    scorer = MotionQualityScorer()
    results = []
    for i, path in enumerate(video_paths):
        result = scorer.score(path)
        results.append(result)
        if verbose and (i + 1) % 100 == 0:
            ready = "✓" if result.is_training_ready else "✗"
            print(f"  [{i+1}/{len(video_paths)}] {ready} score={result.overall:.2f} — {path}")
    return results
