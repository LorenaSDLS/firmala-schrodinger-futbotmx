from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.F_simulation.field_registration import FieldRegistration
from src.I_field_geometry.field_geometry import FieldGeometryResult, SIDE_NAMES
from src.I_field_geometry.hologram_calibration import HologramCalibration, HologramKeyframe


@dataclass(frozen=True)
class HologramPose:
    frame_index: int
    field_to_image: np.ndarray | None
    image_to_field: np.ndarray | None
    corners_image: np.ndarray | None
    confidence: float
    state: str
    source: str
    measured: bool
    registration_quality: float
    registration_updated: bool
    anchor_left: int | None
    anchor_right: int | None
    anchor_disagreement_px: float = 0.0
    stale_gap_frames: int = 0

    @property
    def valid(self) -> bool:
        return self.image_to_field is not None and self.corners_image is not None

    @property
    def trusted(self) -> bool:
        return self.valid and self.state in {"anchored", "tracking"}


class AssistedHologramTrajectory:
    """Track a user-proven metric field hologram through a handheld video.

    The user supplies one or more exact field projections.  Frame-to-frame
    registration only estimates how the camera moved; it never re-identifies
    the field from thresholds.  Multiple manual keyframes close drift and allow
    the offline solution to use evidence from both the past and the future.
    """

    CACHE_VERSION = 3

    def __init__(
        self,
        calibration: HologramCalibration,
        matrices_current_to_reference: np.ndarray,
        qualities: np.ndarray,
        updated: np.ndarray,
    ) -> None:
        self.calibration = calibration
        self.matrices = np.asarray(matrices_current_to_reference, dtype=np.float64)
        self.qualities = np.asarray(qualities, dtype=np.float32).reshape(-1)
        self.updated = np.asarray(updated, dtype=bool).reshape(-1)
        if self.matrices.ndim != 3 or self.matrices.shape[1:] != (3, 3):
            raise ValueError("La trayectoria de cámara tiene una forma inválida.")
        if len(self.matrices) != len(self.qualities) or len(self.matrices) != len(self.updated):
            raise ValueError("La trayectoria de cámara está desincronizada.")
        self.frame_count = len(self.matrices)
        self.field_corners = np.float32(
            [
                [0.0, 0.0],
                [calibration.field_width, 0.0],
                [calibration.field_width, calibration.field_height],
                [0.0, calibration.field_height],
            ]
        )
        self.anchor_map = {item.frame_index: item for item in calibration.keyframes}
        self.anchor_indices = np.asarray(sorted(self.anchor_map), dtype=np.int32)
        self._poses = [self._solve_frame(index) for index in range(self.frame_count)]

    @staticmethod
    def _cache_identity(video_path: Path, calibration: HologramCalibration) -> dict[str, Any]:
        stat = video_path.stat()
        anchor_digest = hashlib.sha256(
            json.dumps(
                [item.to_dict() for item in calibration.keyframes],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "version": AssistedHologramTrajectory.CACHE_VERSION,
            "video_path": str(video_path.resolve()),
            "video_size": int(stat.st_size),
            "video_mtime_ns": int(stat.st_mtime_ns),
            "frame_width": int(calibration.frame_width),
            "frame_height": int(calibration.frame_height),
            "total_frames": int(calibration.total_frames),
            "anchor_digest": anchor_digest,
        }

    @classmethod
    def from_video(
        cls,
        video_path: str | Path,
        calibration: HologramCalibration,
        cache_path: str | Path | None = None,
        processing_max_width: int = 420,
        tracking_stride: int | None = None,
    ) -> "AssistedHologramTrajectory":
        video_path = Path(video_path).expanduser().resolve()
        cache = Path(cache_path) if cache_path is not None else None
        identity = cls._cache_identity(video_path, calibration)
        if cache is not None and cache.exists():
            try:
                loaded = np.load(cache, allow_pickle=False)
                metadata = json.loads(str(loaded["metadata"].item()))
                if metadata == identity:
                    return cls(
                        calibration,
                        loaded["matrices"],
                        loaded["qualities"],
                        loaded["updated"],
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or calibration.fps or 30.0)
        stride = int(tracking_stride or max(1, int(np.ceil(fps / 24.0))))
        calibration = calibration.scaled_to(width, height)
        registration = FieldRegistration(
            frame_width=width,
            frame_height=height,
            enabled=True,
            max_corners=260,
            processing_max_width=processing_max_width,
            allow_projective=False,
            deterministic_fit=True,
        )
        sample_indices: list[int] = []
        sample_matrices: list[np.ndarray] = []
        sample_qualities: list[float] = []
        sample_updated: list[bool] = []
        progress = tqdm(total=total if total > 0 else None, desc="Rastreando holograma", unit="frame")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                should_track = frame_index % stride == 0 or (total > 0 and frame_index == total - 1)
                if should_track:
                    result = registration.update(frame, frame_index=frame_index)
                    sample_indices.append(frame_index)
                    sample_matrices.append(result.matrix.copy())
                    sample_qualities.append(float(result.quality))
                    sample_updated.append(bool(result.updated))
                frame_index += 1
                progress.update(1)
        finally:
            progress.close()
            capture.release()
        if not sample_matrices:
            raise RuntimeError("El video no contiene cuadros legibles.")

        frame_total = frame_index
        matrices = np.empty((frame_total, 3, 3), dtype=np.float64)
        qualities = np.zeros(frame_total, dtype=np.float32)
        updated = np.zeros(frame_total, dtype=np.uint8)
        source_corners = np.float32(
            [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        )
        for sample_position, sample_index in enumerate(sample_indices):
            matrices[sample_index] = sample_matrices[sample_position]
            qualities[sample_index] = sample_qualities[sample_position]
            updated[sample_index] = int(sample_updated[sample_position])
            if sample_position + 1 >= len(sample_indices):
                continue
            next_index = sample_indices[sample_position + 1]
            first_projected = cv2.perspectiveTransform(
                source_corners.reshape(1, -1, 2), sample_matrices[sample_position]
            ).reshape(4, 2)
            second_projected = cv2.perspectiveTransform(
                source_corners.reshape(1, -1, 2), sample_matrices[sample_position + 1]
            ).reshape(4, 2)
            span = max(1, next_index - sample_index)
            for intermediate in range(sample_index + 1, next_index):
                alpha = (intermediate - sample_index) / span
                projected = (1.0 - alpha) * first_projected + alpha * second_projected
                matrices[intermediate] = cv2.getPerspectiveTransform(
                    source_corners, projected.astype(np.float32)
                )
                qualities[intermediate] = min(
                    sample_qualities[sample_position], sample_qualities[sample_position + 1]
                )
                updated[intermediate] = int(
                    sample_updated[sample_position] and sample_updated[sample_position + 1]
                )
        # A one-frame video or a truncated final interval still needs values.
        last_index = sample_indices[-1]
        if last_index + 1 < frame_total:
            matrices[last_index + 1 :] = sample_matrices[-1]
            qualities[last_index + 1 :] = sample_qualities[-1]
            updated[last_index + 1 :] = 0

        trajectory = cls(calibration, matrices, qualities, updated)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                metadata=json.dumps(identity, sort_keys=True),
                matrices=trajectory.matrices,
                qualities=trajectory.qualities,
                updated=trajectory.updated.astype(np.uint8),
            )
        return trajectory

    def __len__(self) -> int:
        return len(self._poses)

    def pose(self, frame_index: int) -> HologramPose:
        if not self._poses:
            raise IndexError("La trayectoria holográfica está vacía.")
        return self._poses[int(np.clip(frame_index, 0, len(self._poses) - 1))]

    def _propagate_anchor(self, anchor: HologramKeyframe, target_index: int) -> np.ndarray | None:
        anchor_index = int(np.clip(anchor.frame_index, 0, self.frame_count - 1))
        target_index = int(np.clip(target_index, 0, self.frame_count - 1))
        current_to_ref = self.matrices[target_index]
        anchor_to_ref = self.matrices[anchor_index]
        try:
            ref_to_current = np.linalg.inv(current_to_ref)
        except np.linalg.LinAlgError:
            return None
        field_to_anchor = anchor.field_to_image(self.calibration.field_spec)
        candidate = ref_to_current @ anchor_to_ref @ field_to_anchor
        if not np.isfinite(candidate).all() or abs(float(candidate[2, 2])) < 1e-10:
            return None
        return candidate / candidate[2, 2]

    def _interval_statistics(self, first: int, second: int) -> tuple[float, float, int]:
        low, high = sorted((int(first), int(second)))
        low = max(0, low)
        high = min(self.frame_count - 1, high)
        if high <= low:
            return 1.0, 0.0, 0
        q = self.qualities[low + 1 : high + 1]
        u = self.updated[low + 1 : high + 1]
        positives = q[q > 0]
        median_quality = float(np.median(positives)) if positives.size else 0.0
        bad_fraction = float(1.0 - np.mean(u)) if u.size else 0.0
        max_gap = 0
        current = 0
        for value in u:
            if value:
                current = 0
            else:
                current += 1
                max_gap = max(max_gap, current)
        return median_quality, bad_fraction, max_gap

    def _anchor_confidence(self, anchor: HologramKeyframe, target: int) -> tuple[float, int]:
        median_quality, bad_fraction, max_gap = self._interval_statistics(anchor.frame_index, target)
        quality_term = 0.48 + 0.52 * median_quality
        failure_term = max(0.08, 1.0 - 0.72 * bad_fraction)
        gap_term = float(np.exp(-max(0, max_gap - 4) / 20.0))
        confidence = anchor.confidence * quality_term * failure_term * gap_term
        if target == anchor.frame_index:
            confidence = 1.0
            max_gap = 0
        return float(np.clip(confidence, 0.0, 1.0)), int(max_gap)

    def _valid_corners(self, corners: np.ndarray) -> bool:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(points).all() or not cv2.isContourConvex(points):
            return False
        area = abs(float(cv2.contourArea(points)))
        frame_area = float(self.calibration.frame_width * self.calibration.frame_height)
        if area < 0.002 * frame_area or area > 35.0 * frame_area:
            return False
        span = np.ptp(points, axis=0)
        return bool(span[0] > 20.0 and span[1] > 20.0)

    def _candidate_corners(self, anchor: HologramKeyframe, target: int) -> np.ndarray | None:
        matrix = self._propagate_anchor(anchor, target)
        if matrix is None:
            return None
        corners = cv2.perspectiveTransform(
            self.field_corners.reshape(1, -1, 2), matrix
        ).reshape(4, 2)
        return corners if self._valid_corners(corners) else None

    def _solve_frame(self, frame_index: int) -> HologramPose:
        if frame_index in self.anchor_map:
            anchor = self.anchor_map[frame_index]
            field_to_image = anchor.field_to_image(self.calibration.field_spec)
            return HologramPose(
                frame_index=frame_index,
                field_to_image=field_to_image,
                image_to_field=np.linalg.inv(field_to_image),
                corners_image=anchor.corners_image.copy(),
                confidence=1.0,
                state="anchored",
                source="holograma_v11_keyframe",
                measured=True,
                registration_quality=float(self.qualities[frame_index]),
                registration_updated=bool(self.updated[frame_index]),
                anchor_left=frame_index,
                anchor_right=frame_index,
            )

        position = int(np.searchsorted(self.anchor_indices, frame_index))
        left_index = int(self.anchor_indices[position - 1]) if position > 0 else None
        right_index = int(self.anchor_indices[position]) if position < len(self.anchor_indices) else None
        left_anchor = self.anchor_map.get(left_index) if left_index is not None else None
        right_anchor = self.anchor_map.get(right_index) if right_index is not None else None

        candidates: list[tuple[np.ndarray, float, int, int]] = []
        for anchor in (left_anchor, right_anchor):
            if anchor is None:
                continue
            corners = self._candidate_corners(anchor, frame_index)
            if corners is None:
                continue
            confidence, gap = self._anchor_confidence(anchor, frame_index)
            candidates.append((corners, confidence, gap, anchor.frame_index))
        if not candidates:
            return HologramPose(
                frame_index, None, None, None, 0.0, "lost", "holograma_v11_sin_pose",
                False, float(self.qualities[frame_index]), bool(self.updated[frame_index]),
                left_index, right_index,
            )

        disagreement = 0.0
        if len(candidates) == 1:
            corners, confidence, max_gap, _anchor_index = candidates[0]
        else:
            left_candidate = next((item for item in candidates if item[3] == left_index), candidates[0])
            right_candidate = next((item for item in candidates if item[3] == right_index), candidates[-1])
            left_corners, left_conf, left_gap, _ = left_candidate
            right_corners, right_conf, right_gap, _ = right_candidate
            span = max(1, int(right_index) - int(left_index))
            alpha = float(np.clip((frame_index - int(left_index)) / span, 0.0, 1.0))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            temporal_left = 1.0 - alpha
            temporal_right = alpha
            weight_left = max(1e-6, temporal_left * left_conf)
            weight_right = max(1e-6, temporal_right * right_conf)
            corners = (weight_left * left_corners + weight_right * right_corners) / (weight_left + weight_right)
            diagonal = float(np.hypot(self.calibration.frame_width, self.calibration.frame_height))
            disagreement = float(np.mean(np.linalg.norm(left_corners - right_corners, axis=1)))
            agreement_term = float(np.exp(-disagreement / max(1.0, 0.035 * diagonal)))
            confidence = float(
                np.clip(
                    (temporal_left * left_conf + temporal_right * right_conf) * agreement_term,
                    0.0,
                    1.0,
                )
            )
            max_gap = max(left_gap, right_gap)

        if not self._valid_corners(corners):
            return HologramPose(
                frame_index, None, None, None, 0.0, "lost", "holograma_v11_pose_degenerada",
                False, float(self.qualities[frame_index]), bool(self.updated[frame_index]),
                left_index, right_index, disagreement, max_gap,
            )
        field_to_image = cv2.getPerspectiveTransform(self.field_corners, corners.astype(np.float32))
        try:
            image_to_field = np.linalg.inv(field_to_image)
        except np.linalg.LinAlgError:
            return HologramPose(
                frame_index, None, None, None, 0.0, "lost", "holograma_v11_no_invertible",
                False, float(self.qualities[frame_index]), bool(self.updated[frame_index]),
                left_index, right_index, disagreement, max_gap,
            )

        if confidence >= 0.42 and max_gap <= 32:
            state = "tracking"
        elif confidence >= 0.20 and max_gap <= 90:
            state = "coasting"
        else:
            state = "lost"
        source = f"holograma_v11_{state}"
        return HologramPose(
            frame_index=frame_index,
            field_to_image=field_to_image,
            image_to_field=image_to_field,
            corners_image=corners.astype(np.float32),
            confidence=confidence,
            state=state,
            source=source,
            measured=False,
            registration_quality=float(self.qualities[frame_index]),
            registration_updated=bool(self.updated[frame_index]),
            anchor_left=left_index,
            anchor_right=right_index,
            anchor_disagreement_px=disagreement,
            stale_gap_frames=max_gap,
        )

    def geometry_result(
        self,
        frame_index: int,
        surface_mask: np.ndarray | None = None,
    ) -> FieldGeometryResult:
        pose = self.pose(frame_index)
        coverage = 0.0
        if surface_mask is not None and surface_mask.size:
            coverage = float(np.mean(np.asarray(surface_mask) > 0))
        visible = {side: False for side in SIDE_NAMES}
        statuses = {side: pose.state for side in SIDE_NAMES}
        confidences = {side: float(pose.confidence) for side in SIDE_NAMES}
        if pose.corners_image is not None:
            corners = pose.corners_image
            for side, (first, second) in {
                "far": (1, 2), "right": (2, 3), "near": (3, 0), "left": (0, 1)
            }.items():
                segment = corners[[first, second]]
                min_xy = np.min(segment, axis=0)
                max_xy = np.max(segment, axis=0)
                visible[side] = bool(
                    max_xy[0] >= 0 and max_xy[1] >= 0
                    and min_xy[0] < self.calibration.frame_width
                    and min_xy[1] < self.calibration.frame_height
                )
        return FieldGeometryResult(
            valid=bool(pose.valid and pose.state != "lost"),
            trusted=bool(pose.trusted),
            measured=bool(pose.measured),
            propagated=not pose.measured,
            confidence=float(pose.confidence),
            corners_image=None if pose.corners_image is None else pose.corners_image.copy(),
            homography_image_to_field=(
                None if pose.image_to_field is None else pose.image_to_field.copy()
            ),
            homography_field_to_image=(
                None if pose.field_to_image is None else pose.field_to_image.copy()
            ),
            mask_coverage=coverage,
            source=pose.source,
            line_support={},
            side_visible=visible,
            side_status=statuses,
            side_confidence=confidences,
            side_lines={},
            rejected_frame_sides=[],
            border_evidence_score=0.0,
            white_alignment_score=0.0,
            goal_consistency_score=0.0,
            visible_template_fraction=float(np.mean(list(visible.values()))),
            manual_line_score=1.0 if pose.measured else 0.0,
            registration_scope="full" if pose.valid else "none",
            geometry_state="global" if pose.valid else "surface",
            hard_anchor_score=1.0 if pose.measured else float(pose.confidence),
            hard_anchor_count=1 if pose.measured else 0,
            feature_match_score=float(pose.registration_quality),
            feature_match_count=0,
            feature_matches={
                "anchor_disagreement_px": float(pose.anchor_disagreement_px),
                "stale_gap_frames": float(pose.stale_gap_frames),
            },
            pose_admission_state=pose.state,
            pose_candidate_streak=0,
            registration_stale_frames=int(pose.stale_gap_frames),
            local_lock_active=True,
            field_width=self.calibration.field_width,
            field_height=self.calibration.field_height,
        )
