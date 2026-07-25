"""Integration tests for data annotation pipeline."""

import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

from flux.data.annotation import (
    PhysicsEventDetector,
    PhysicsEventType,
    detect_physics_events,
    ScenarioClassifier,
    ScenarioType,
    classify_scenario,
    MotionQualityScorer,
    MotionScore,
    score_motion_quality,
    AnnotationPipeline,
    annotate_manifest,
)


@pytest.fixture
def sample_video():
    """Create a tiny synthetic video for testing."""
    path = "/tmp/test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (128, 128))
    for i in range(60):  # 2 seconds at 30fps
        # Moving rectangle to simulate motion
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        x = int(20 + i * 0.5) % 100
        cv2.rectangle(frame, (x, 40), (x + 20, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    yield path
    os.unlink(path)


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a test manifest CSV with the synthetic video."""
    path = tmp_path / "test_manifest.csv"
    with open(path, "w") as f:
        f.write("video_path,num_frames,height,width,fps,duration_s,audio_path,caption_short,caption_long,caption_audio\n")
        f.write("/tmp/test_video.mp4,60,128,128,30,2.0,,test caption,,,\n")
    return str(path)


class TestPhysicsEvents:
    def test_detect_single(self, sample_video):
        detector = PhysicsEventDetector()
        result = detector.detect(sample_video)
        assert result.video_path == sample_video
        assert len(result.events) == 8
        assert result.dominant_event in [e.value for e in PhysicsEventType]
        assert 0 <= result.physics_richness <= 1

    def test_detect_batch(self, sample_video):
        results = detect_physics_events([sample_video, sample_video])
        assert len(results) == 2
        for r in results:
            assert isinstance(r.physics_richness, float)

    def test_missing_file(self):
        detector = PhysicsEventDetector()
        result = detector.detect("/tmp/nonexistent.mp4")
        assert result.physics_richness == 0.0


class TestScenarioClassifier:
    def test_classify_single(self, sample_video):
        classifier = ScenarioClassifier()
        result = classifier.classify(sample_video)
        assert result.scenario in [s.value for s in ScenarioType]
        assert 0 <= result.confidence <= 1
        assert len(result.scenario_scores) >= 9

    def test_classify_batch(self, sample_video):
        results = classify_scenario([sample_video, sample_video])
        assert len(results) == 2

    def test_missing_file(self):
        classifier = ScenarioClassifier()
        result = classifier.classify("/tmp/nonexistent.mp4")
        assert result.scenario == "other"
        assert result.confidence == 0.0


class TestMotionQuality:
    def test_score_single(self, sample_video):
        scorer = MotionQualityScorer()
        result = scorer.score(sample_video)
        assert 0 <= result.smoothness <= 1
        assert 0 <= result.naturalness <= 1
        assert 0 <= result.diversity <= 1
        assert 0 <= result.stability <= 1
        assert 0 <= result.complexity <= 1
        assert 0 <= result.overall <= 1
        assert isinstance(result.is_training_ready, bool)

    def test_score_batch(self, sample_video):
        results = score_motion_quality([sample_video, sample_video])
        assert len(results) == 2

    def test_missing_file(self):
        scorer = MotionQualityScorer()
        result = scorer.score("/tmp/nonexistent.mp4")
        assert "cannot_open" in result.issues


class TestAnnotationPipeline:
    def test_annotate_single(self, sample_video):
        pipeline = AnnotationPipeline()
        ann = pipeline.annotate_single(sample_video)
        assert "physics_event" in ann
        assert "scenario" in ann
        assert "motion_overall" in ann
        assert "training_ready" in ann

    def test_process_manifest(self, sample_manifest, sample_video):
        # Need the sample video to exist
        stats = annotate_manifest(
            sample_manifest,
            str(Path(sample_manifest).parent / "annotated.csv"),
            max_videos=1,
            verbose=False,
        )
        assert stats["annotated"] >= 0
