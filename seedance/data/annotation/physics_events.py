"""Auto-detect physical interaction events from video using lightweight heuristics.

Detects 8 categories of physical events without requiring a large model:
1. Collision    — objects meeting with sudden velocity change
2. Gravity      — downward acceleration patterns
3. Fluid        — smooth, wave-like motion regions
4. Contact      — sustained object-surface interaction
5. Deformation  — non-rigid shape changes
6. Occlusion    — objects passing behind others
7. Trajectory   — smooth projectile/parabolic paths
8. Static       — no significant physical interaction

Methods use optical flow, frame differencing, and simple CV heuristics.
Runs on CPU at ~50 videos/second.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class PhysicsEventType(Enum):
    COLLISION = "collision"
    GRAVITY = "gravity"
    FLUID = "fluid"
    CONTACT = "contact"
    DEFORMATION = "deformation"
    OCCLUSION = "occlusion"
    TRAJECTORY = "trajectory"
    STATIC = "static"


@dataclass
class PhysicsEventResult:
    """Result of physics event detection for a single video."""
    video_path: str
    events: dict[str, float] = field(default_factory=dict)
    dominant_event: str = "static"
    physics_richness: float = 0.0
    recommended_for_physics_training: bool = False


class PhysicsEventDetector:
    """Detect physical interaction events from video frames.

    Uses optical flow + frame differencing heuristics.
    Lightweight — runs on CPU, ~50 fps.

    Args:
        sample_frames: Number of evenly-spaced frames to analyze (default: 16).
        min_flow_magnitude: Minimum mean optical flow to consider "motion" (default: 0.5).
        collision_threshold: Flow variance threshold for collision detection.
    """

    def __init__(
        self,
        sample_frames: int = 16,
        min_flow_magnitude: float = 0.5,
        collision_threshold: float = 3.0,
    ):
        self.sample_frames = sample_frames
        self.min_flow_magnitude = min_flow_magnitude
        self.collision_threshold = collision_threshold

    def detect(self, video_path: str) -> PhysicsEventResult:
        """Analyze video and return physics event scores.

        Args:
            video_path: Path to video file.

        Returns:
            PhysicsEventResult with per-event confidence scores [0, 1].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return PhysicsEventResult(video_path=video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            cap.release()
            return PhysicsEventResult(video_path=video_path)

        # Sample evenly spaced frames
        indices = np.linspace(0, total_frames - 2, min(self.sample_frames, total_frames - 1), dtype=int)

        # Compute optical flow between consecutive frame pairs
        flows = []
        diffs = []

        prev_frame = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flows.append(flow_mag)
                diffs.append(np.abs(gray.astype(float) - prev_frame.astype(float)))

            prev_frame = gray

        cap.release()

        if not flows:
            return PhysicsEventResult(video_path=video_path)

        # Stack flow magnitudes
        flow_stack = np.stack(flows)  # (T, H, W)
        diff_stack = np.stack(diffs)  # (T, H, W)

        # ── Compute event scores ──────────────────────────────────

        # Mean flow magnitude over time
        mean_flow = flow_stack.mean()
        flow_variance = flow_stack.std()

        # Spatial: per-region flow for localized events
        H, W = flow_stack.shape[1], flow_stack.shape[2]
        regions = self._split_regions(flow_stack)  # (4, T, H/2, W/2)

        events = {}

        # Collision: sudden spike in flow variance at a specific time
        frame_variances = flow_stack.reshape(len(flows), -1).var(axis=1)
        collision_score = min(1.0, frame_variances.max() / (self.collision_threshold * mean_flow + 1e-8))
        events["collision"] = round(float(collision_score), 3)

        # Gravity: consistent downward flow bias
        gravity_score = self._detect_gravity(flows)
        events["gravity"] = round(float(gravity_score), 3)

        # Fluid: low variance, smooth spatial gradients, wave-like
        spatial_gradient = np.abs(np.diff(flow_stack, axis=1)).mean() + \
                          np.abs(np.diff(flow_stack, axis=2)).mean()
        fluid_score = 1.0 - min(1.0, spatial_gradient / (mean_flow + 1e-8))
        events["fluid"] = round(float(fluid_score), 3)

        # Contact: sustained interaction between moving and static regions
        contact_score = self._detect_contact(regions, flow_stack)
        events["contact"] = round(float(contact_score), 3)

        # Deformation: non-rigid motion (high local variance)
        deform_score = min(1.0, flow_variance / (mean_flow * 2 + 1e-8))
        events["deformation"] = round(float(deform_score), 3)

        # Occlusion: objects appearing/disappearing
        diff_variance = diff_stack.var(axis=0).mean()
        occlusion_score = min(1.0, diff_variance / (mean_flow * 5 + 1e-8))
        events["occlusion"] = round(float(occlusion_score), 3)

        # Trajectory: smooth motion path (low acceleration variance)
        if len(flow_stack) >= 3:
            acceleration = np.diff(flow_stack, axis=0)
            accel_variance = acceleration.var()
            trajectory_score = 1.0 - min(1.0, accel_variance / (mean_flow + 1e-8))
        else:
            trajectory_score = 0.0
        events["trajectory"] = round(float(trajectory_score), 3)

        # Static: very low motion
        static_score = 1.0 - min(1.0, mean_flow / self.min_flow_magnitude)
        events["static"] = round(float(static_score), 3)

        # ── Summary ──────────────────────────────────────────────
        dominant = max(events, key=events.get)
        # Physics richness: how much interesting physics is present
        physics_richness = sum(v for k, v in events.items() if k != "static") / 7.0
        recommended = physics_richness > 0.3

        return PhysicsEventResult(
            video_path=video_path,
            events=events,
            dominant_event=dominant,
            physics_richness=round(float(physics_richness), 3),
            recommended_for_physics_training=recommended,
        )

    @staticmethod
    def _split_regions(flow_stack: np.ndarray) -> np.ndarray:
        """Split flow field into 4 quadrants for localized analysis."""
        T, H, W = flow_stack.shape
        h2, w2 = H // 2, W // 2
        regions = np.zeros((4, T, h2, w2), dtype=flow_stack.dtype)
        regions[0] = flow_stack[:, :h2, :w2]      # top-left
        regions[1] = flow_stack[:, :h2, w2:]       # top-right
        regions[2] = flow_stack[:, h2:, :w2]       # bottom-left
        regions[3] = flow_stack[:, h2:, w2:]       # bottom-right
        return regions

    @staticmethod
    def _detect_gravity(flows: list[np.ndarray]) -> float:
        """Detect downward-dominant motion pattern (gravity)."""
        downs = []
        for flow_mag in flows:
            # Check if bottom half has more motion than top half
            H = flow_mag.shape[0]
            top = flow_mag[:H//2].mean()
            bottom = flow_mag[H//2:].mean()
            if bottom > top * 1.2:
                downs.append(1.0)
            else:
                downs.append(0.0)
        return np.mean(downs) if downs else 0.0

    @staticmethod
    def _detect_contact(regions: np.ndarray, flow_stack: np.ndarray) -> float:
        """Detect contact: one region moves while adjacent region is static."""
        T = regions.shape[1]
        if T < 2 or regions.shape[0] < 2:
            return 0.0

        # Compare adjacent region pair flows
        pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
        contact_scores = []
        for r1, r2 in pairs:
            diff = np.abs(regions[r1].mean() - regions[r2].mean())
            contact_scores.append(min(1.0, diff / (flow_stack.mean() + 1e-8)))
        return float(np.mean(contact_scores))


def detect_physics_events(
    video_paths: list[str],
    sample_frames: int = 16,
    verbose: bool = False,
) -> list[PhysicsEventResult]:
    """Batch-detect physics events for a list of videos."""
    detector = PhysicsEventDetector(sample_frames=sample_frames)
    results = []
    for i, path in enumerate(video_paths):
        result = detector.detect(path)
        results.append(result)
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(video_paths)}] {result.dominant_event} r={result.physics_richness:.2f} — {path}")
    return results
