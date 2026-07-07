"""Data annotation pipeline for physics-aware training data curation.

ByteDance Seedance 2.0's success is widely attributed to "data victory" —
scenario-specific, physics-aware data labeling at scale.

This module provides a lightweight annotation pipeline that:
1. Auto-detects physical events (collision, gravity, fluid, contact)
2. Classifies videos into scenarios (sports, nature, indoor, talking, etc.)
3. Scores motion quality using optical flow heuristics
4. Tags videos with enriched metadata for physics-aware training

Reference:
    36kr (2026-07-07): "Seedance 2.0 的成功被多位从业者归结为'数据的胜利'，
    团队针对不同场景做针对性数据训练（如运动场景、游戏画面、室内空间等）"
"""

from seedance.data.annotation.physics_events import (
    PhysicsEventDetector,
    PhysicsEventType,
    detect_physics_events,
)

from seedance.data.annotation.scenario_classifier import (
    ScenarioClassifier,
    ScenarioType,
    classify_scenario,
)

from seedance.data.annotation.motion_quality import (
    MotionQualityScorer,
    MotionScore,
    score_motion_quality,
)

from seedance.data.annotation.pipeline import (
    AnnotationPipeline,
    annotate_manifest,
)

__all__ = [
    "PhysicsEventDetector",
    "PhysicsEventType",
    "detect_physics_events",
    "ScenarioClassifier",
    "ScenarioType",
    "classify_scenario",
    "MotionQualityScorer",
    "MotionScore",
    "score_motion_quality",
    "AnnotationPipeline",
    "annotate_manifest",
]
