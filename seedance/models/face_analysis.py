"""Production-grade face analysis for Seedance 2.0.

Replaces all demo-level face detection (Haar Cascade, basic MediaPipe)
with industry-standard pipelines:

  Detection:    SCRFD-10G  (via InsightFace) — SOTA lightweight detector
  Landmarks:    5-point    (eyes, nose, mouth corners) from SCRFD
  Face Mesh:    468-point  (via MediaPipe Face Mesh) for 3D keypoints
  Recognition:  ArcFace    (via InsightFace) for identity embedding

All components are lazy-loaded and cached. Falls back gracefully
when optional backends are unavailable.

Usage:
    from seedance.models.face_analysis import FaceAnalyzer

    analyzer = FaceAnalyzer()   # auto-selects best available backend

    # Detection + 5-point landmarks
    faces = analyzer.detect(frame)  # -> list[FaceResult]

    # 468-point 3D mesh (for lip-sync / KP encoder)
    mesh = analyzer.extract_mesh(frame)  # -> (468, 3) ndarray

    # Identity embedding (for LFA anchor)
    embedding = analyzer.extract_identity(frame)  # -> (512,) ndarray
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class FaceResult:
    """Single face detection result.

    All coordinates are in pixel space (original image coordinates).
    Confidence in [0, 1].
    """
    bbox: np.ndarray          # (4,) [x1, y1, x2, y2] in pixels
    confidence: float          # detection confidence [0, 1]
    landmarks_5: np.ndarray | None = None  # (5, 2) eye centers, nose tip, mouth corners
    identity_embedding: np.ndarray | None = None  # (512,) ArcFace embedding
    mouth_bbox: np.ndarray | None = None  # (4,) estimated mouth region


class SCRFDBackend:
    """SCRFD face detection via InsightFace.

    SCRFD (Sample and Computation Redistribution for Face Detection)
    is the industry standard for fast, accurate face detection.
    Model: SCRFD-10G (balance of speed and accuracy).

    Reference: https://github.com/deepinsight/insightface
    """

    def __init__(self, model_name: str = "buffalo_l", device: str = "auto"):
        self._model = None
        self._model_name = model_name
        self._device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string for onnxruntime."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @property
    def model(self):
        """Lazy-load InsightFace SCRFD model."""
        if self._model is None:
            import insightface
            self._model = insightface.app.FaceAnalysis(
                name=self._model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda" else ["CPUExecutionProvider"],
            )
            # SCRFD-10G at full resolution for production accuracy
            self._model.prepare(
                ctx_id=0 if self._device == "cuda" else -1,
                det_size=(640, 640),
                det_thresh=0.5,
            )
        return self._model

    def detect(
        self,
        image: np.ndarray,
        max_num: int = 0,
        extract_identity: bool = False,
    ) -> list[FaceResult]:
        """Detect faces using SCRFD.

        Args:
            image: RGB image (H, W, 3) uint8.
            max_num: Max faces to return (0 = all).
            extract_identity: Whether to extract ArcFace embeddings.

        Returns:
            List of FaceResult, sorted by confidence descending.
        """
        if image.ndim == 3 and image.shape[0] == 3:
            # CHW → HWC
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Ensure contiguous for ONNX
        image = np.ascontiguousarray(image)

        try:
            faces = self.model.get(image, max_num=max_num)
        except Exception:
            return []

        results = []
        for face in faces:
            result = FaceResult(
                bbox=face.bbox.astype(np.float32),
                confidence=float(face.det_score),
                landmarks_5=(
                    face.landmark_2d_106[:5].astype(np.float32)
                    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None
                    else face.kps.astype(np.float32) if hasattr(face, "kps") and face.kps is not None
                    else None
                ),
            )

            # Compute mouth bbox from 5-point landmarks
            if result.landmarks_5 is not None:
                # Mouth corners are landmarks[3] and landmarks[4]
                mouth_left = result.landmarks_5[3]
                mouth_right = result.landmarks_5[4]
                mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
                mouth_width = abs(mouth_right[0] - mouth_left[0])
                mouth_height = mouth_width * 0.5
                result.mouth_bbox = np.array([
                    mouth_left[0] - mouth_width * 0.1,
                    mouth_center_y - mouth_height * 0.6,
                    mouth_right[0] + mouth_width * 0.1,
                    mouth_center_y + mouth_height * 0.6,
                ], dtype=np.float32)

            # Identity embedding
            if extract_identity and hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                result.identity_embedding = face.normed_embedding.astype(np.float32)

            results.append(result)

        return results


class MediaPipeMeshBackend:
    """468-point 3D face mesh via MediaPipe Face Mesh.

    Provides dense facial landmarks for:
      - Lip-sync mouth shape analysis
      - 3D keypoint control signals
      - Expression/pose estimation

    Output: (468, 3) array in [x, y, z] pixel/image coordinates.
    """

    def __init__(self, static_mode: bool = False, max_faces: int = 1):
        self._mesh = None
        self._static_mode = static_mode
        self._max_faces = max_faces

    @property
    def mesh(self):
        """Lazy-load MediaPipe Face Mesh."""
        if self._mesh is None:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self._static_mode,
                max_num_faces=self._max_faces,
                refine_landmarks=True,  # Enable iris + lip contour refinement
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._mesh

    def extract(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """Extract 468-point 3D face meshes.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            List of (468, 3) ndarrays, one per detected face.
            Empty list if no faces found.
        """
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        H, W = image.shape[:2]

        try:
            results = self.mesh.process(image)
        except Exception:
            return []

        meshes = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = np.array([
                    [lm.x * W, lm.y * H, lm.z * W]  # z scaled like x for consistency
                    for lm in face_landmarks.landmark
                ], dtype=np.float32)
                meshes.append(points)

        return meshes


class FaceAnalyzer:
    """Unified production-grade face analysis.

    Orchestrates SCRFD detection, MediaPipe 468-point mesh extraction,
    and ArcFace identity embedding through a single interface.

    All backends are lazy-loaded on first use.

    Args:
        use_scrfd: Enable SCRFD detection + identity (default True).
        use_mesh: Enable MediaPipe 468-point mesh (default True).
        device: "auto", "cuda", or "cpu".
    """

    def __init__(
        self,
        use_scrfd: bool = True,
        use_mesh: bool = True,
        device: str = "auto",
    ):
        self._scrfd = SCRFDBackend(device=device) if use_scrfd else None
        self._mesh = MediaPipeMeshBackend() if use_mesh else None
        self.device = device

    @property
    def has_scrfd(self) -> bool:
        return self._scrfd is not None

    @property
    def has_mesh(self) -> bool:
        return self._mesh is not None

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        extract_identity: bool = False,
    ) -> list[FaceResult]:
        """Detect faces with SCRFD.

        Args:
            image: RGB image (H, W, 3) uint8 or (C, H, W) float tensor.
            extract_identity: Whether to compute ArcFace identity embeddings.

        Returns:
            List of FaceResult sorted by confidence.
        """
        if self._scrfd is None:
            return []
        return self._scrfd.detect(image, extract_identity=extract_identity)

    def extract_mesh(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """Extract 468-point 3D face meshes.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            List of (468, 3) ndarrays.
        """
        if self._mesh is None:
            return []
        return self._mesh.extract(image)

    def detect_largest(
        self,
        image: np.ndarray,
        extract_identity: bool = False,
        extract_mesh: bool = False,
    ) -> FaceResult | None:
        """Detect the largest face and optionally extract mesh + identity.

        Convenience method for single-face videos (the common case).

        Args:
            image: RGB image.
            extract_identity: Compute ArcFace embedding.
            extract_mesh: Extract 468-point mesh.

        Returns:
            FaceResult or None.
        """
        faces = self.detect(image, extract_identity=extract_identity)
        if not faces:
            return None

        # Largest face by bbox area
        result = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))

        if extract_mesh and self.has_mesh:
            meshes = self.extract_mesh(image)
            if meshes:
                # Match mesh to detected face by proximity
                face_center = np.array([
                    (result.bbox[0] + result.bbox[2]) / 2,
                    (result.bbox[1] + result.bbox[3]) / 2,
                ])
                best_mesh = min(
                    meshes,
                    key=lambda m: np.linalg.norm(m[:, :2].mean(axis=0) - face_center)
                )
                result._mesh = best_mesh

        return result

    # ------------------------------------------------------------------
    # Batch processing helpers
    # ------------------------------------------------------------------

    def extract_identity_anchor(
        self,
        image: np.ndarray,
    ) -> np.ndarray | None:
        """Extract ArcFace identity anchor for LFA.

        Args:
            image: RGB face image or full frame.

        Returns:
            (512,) normalized identity embedding, or None.
        """
        face = self.detect_largest(image, extract_identity=True)
        if face is None or face.identity_embedding is None:
            return None
        return face.identity_embedding

    def extract_mouth_region(
        self,
        image: np.ndarray,
        expand_ratio: float = 0.15,
    ) -> np.ndarray | None:
        """Extract mouth region crop from a full frame.

        Args:
            image: RGB image (H, W, 3).
            expand_ratio: Extra padding around estimated mouth bbox.

        Returns:
            Cropped mouth region (H', W', 3) or None.
        """
        result = self.detect_largest(image)
        if result is None:
            return None

        H, W = image.shape[:2]

        # Get mouth bbox from 5-point landmarks
        if result.mouth_bbox is not None:
            x1, y1, x2, y2 = result.mouth_bbox
        else:
            # Fallback: lower 1/3 of face
            fx1, fy1, fx2, fy2 = result.bbox
            face_h = fy2 - fy1
            x1 = fx1 + face_h * 0.2
            y1 = fy1 + face_h * 0.55
            x2 = fx2 - face_h * 0.2
            y2 = fy1 + face_h * 0.90

        # Expand
        mw = x2 - x1
        mh = y2 - y1
        x1 = max(0, int(x1 - mw * expand_ratio))
        y1 = max(0, int(y1 - mh * expand_ratio))
        x2 = min(W, int(x2 + mw * expand_ratio))
        y2 = min(H, int(y2 + mh * expand_ratio))

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    def get_mouth_mask(
        self,
        image: np.ndarray,
        H_lat: int,
        W_lat: int,
    ) -> np.ndarray | None:
        """Generate mouth attention mask for LipSyncBridge.

        Creates a Gaussian-weighted mask centered on the detected mouth
        position, sized for the latent spatial grid.

        Args:
            image: RGB image (H, W, 3).
            H_lat: Latent grid height.
            W_lat: Latent grid width.

        Returns:
            (H_lat, W_lat) float32 mask, or None if no face detected.
        """
        result = self.detect_largest(image)
        if result is None:
            return None

        if result.mouth_bbox is not None:
            mx1, my1, mx2, my2 = result.mouth_bbox
        else:
            fx1, fy1, fx2, fy2 = result.bbox
            fh = fy2 - fy1
            mx1, my1 = fx1 + fh * 0.2, fy1 + fh * 0.55
            mx2, my2 = fx2 - fh * 0.2, fy1 + fh * 0.90

        H_img, W_img = image.shape[:2]

        # Map to normalized coordinates
        mouth_cy = ((my1 + my2) / 2) / H_img
        mouth_cx = ((mx1 + mx2) / 2) / W_img
        mouth_sy = ((my2 - my1) / 2) / H_img * 2.0
        mouth_sx = ((mx2 - mx1) / 2) / W_img * 2.0

        # Create Gaussian on latent grid
        ys = np.linspace(0, 1, H_lat)
        xs = np.linspace(0, 1, W_lat)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")

        sigma_y = max(mouth_sy, 0.03)
        sigma_x = max(mouth_sx, 0.05)

        mask = np.exp(
            -((gy - mouth_cy) ** 2) / (2 * sigma_y ** 2)
            -((gx - mouth_cx) ** 2) / (2 * sigma_x ** 2)
        )
        mask = mask / (mask.max() + 1e-8)

        return mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init, cached)
# ---------------------------------------------------------------------------
_global_analyzer: FaceAnalyzer | None = None


def get_face_analyzer(device: str = "auto") -> FaceAnalyzer:
    """Get or create the global FaceAnalyzer singleton."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = FaceAnalyzer(device=device)
    return _global_analyzer
