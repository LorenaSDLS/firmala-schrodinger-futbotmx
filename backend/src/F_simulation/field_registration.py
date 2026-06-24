"""Frame-to-frame camera-motion compensation for handheld match videos.

This does not require a fixed camera or known field corners. It tracks visual
features on the green/white playing surface and accumulates a robust affine
transform from the current frame back to the first frame. The stabilized point
is therefore less sensitive to camera pan, rotation and moderate zoom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class RegistrationResult:
    matrix: np.ndarray
    valid: bool
    updated: bool
    quality: float
    tracked_points: int
    inlier_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "updated": bool(self.updated),
            "quality": round(float(self.quality), 5),
            "tracked_points": int(self.tracked_points),
            "inlier_ratio": round(float(self.inlier_ratio), 5),
            "matrix": [[round(float(v), 8) for v in row] for row in self.matrix],
        }


class FieldRegistration:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        enabled: bool = True,
        max_corners: int = 320,
        processing_max_width: int = 720,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.enabled = bool(enabled)
        self.max_corners = int(max_corners)
        self.processing_scale = min(1.0, float(processing_max_width) / max(1, self.frame_width))
        self.processing_width = max(1, int(round(self.frame_width * self.processing_scale)))
        self.processing_height = max(1, int(round(self.frame_height * self.processing_scale)))
        self.previous_gray: np.ndarray | None = None
        self.previous_mask: np.ndarray | None = None
        self.current_to_reference = np.eye(3, dtype=np.float64)
        self.last_valid_increment = np.eye(3, dtype=np.float64)
        self.frame_index = -1
        self.last_result = RegistrationResult(
            matrix=self.current_to_reference.copy(),
            valid=True,
            updated=True,
            quality=1.0,
            tracked_points=0,
            inlier_ratio=1.0,
        )

    @staticmethod
    def _field_mask(
        frame: np.ndarray,
        semantic_mask: np.ndarray | None = None,
        exclusion_boxes: list[list[float]] | None = None,
    ) -> np.ndarray:
        if semantic_mask is not None:
            mask = (semantic_mask > 0).astype(np.uint8) * 255
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, np.array([27, 25, 25]), np.array([108, 255, 255]))
            white = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 82, 255]))
            mask = cv2.bitwise_or(green, white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Moving robots, the ball and hands should not drive camera motion.
        for box in exclusion_boxes or []:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            pad_x = max(5, int(round(0.12 * max(1.0, x2 - x1))))
            pad_y = max(5, int(round(0.12 * max(1.0, y2 - y1))))
            cv2.rectangle(
                mask,
                (max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)),
                (min(mask.shape[1] - 1, int(x2) + pad_x), min(mask.shape[0] - 1, int(y2) + pad_y)),
                0,
                thickness=cv2.FILLED,
            )
        return mask

    @staticmethod
    def _is_sane_increment(matrix: np.ndarray, diagonal: float) -> bool:
        linear = matrix[:2, :2]
        det = float(np.linalg.det(linear))
        if det <= 0.0:
            return False
        scale = float(np.sqrt(det))
        rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        translation = float(np.linalg.norm(matrix[:2, 2]))
        return (
            0.78 <= scale <= 1.28
            and abs(rotation) <= 12.0
            and translation <= 0.22 * diagonal
        )

    def update(
        self,
        frame: np.ndarray,
        semantic_mask: np.ndarray | None = None,
        exclusion_boxes: list[list[float]] | None = None,
        frame_index: int | None = None,
    ) -> RegistrationResult:
        # ``preview_generator`` passes the source-video frame index so that
        # camera registration and field geometry stay synchronized when a
        # run starts at a non-zero frame or skips frames.  Older callers may
        # omit it; in that case we keep the original sequential behaviour.
        if frame_index is None:
            self.frame_index += 1
        else:
            self.frame_index = int(frame_index)
        if not self.enabled:
            self.last_result = RegistrationResult(
                matrix=np.eye(3, dtype=np.float64),
                valid=False,
                updated=False,
                quality=0.0,
                tracked_points=0,
                inlier_ratio=0.0,
            )
            return self.last_result

        if self.processing_scale < 1.0:
            working = cv2.resize(
                frame,
                (self.processing_width, self.processing_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = frame
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        semantic_small = None
        if semantic_mask is not None:
            semantic_small = (semantic_mask > 0).astype(np.uint8) * 255
            if self.processing_scale < 1.0:
                semantic_small = cv2.resize(
                    semantic_small,
                    (self.processing_width, self.processing_height),
                    interpolation=cv2.INTER_NEAREST,
                )
        boxes_small = None
        if exclusion_boxes:
            scale = self.processing_scale
            boxes_small = [
                [float(value) * scale for value in box]
                for box in exclusion_boxes
            ]
        mask = self._field_mask(
            working,
            semantic_mask=semantic_small,
            exclusion_boxes=boxes_small,
        )

        if self.previous_gray is None:
            self.previous_gray = gray
            self.previous_mask = mask
            self.last_result = RegistrationResult(
                matrix=self.current_to_reference.copy(),
                valid=True,
                updated=True,
                quality=1.0,
                tracked_points=0,
                inlier_ratio=1.0,
            )
            return self.last_result

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.008,
            minDistance=8,
            blockSize=7,
            mask=self.previous_mask,
        )

        valid = False
        quality = 0.0
        tracked_count = 0
        inlier_ratio = 0.0
        increment = None

        if previous_points is not None and len(previous_points) >= 12:
            current_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if current_points is not None and status is not None:
                status = status.reshape(-1).astype(bool)
                previous_good = previous_points.reshape(-1, 2)[status]
                current_good = current_points.reshape(-1, 2)[status]
                error_good = (
                    errors.reshape(-1)[status]
                    if errors is not None
                    else np.zeros(len(previous_good), dtype=np.float32)
                )
                finite = (
                    np.isfinite(previous_good).all(axis=1)
                    & np.isfinite(current_good).all(axis=1)
                    & (error_good < 40.0)
                )
                previous_good = previous_good[finite]
                current_good = current_good[finite]
                tracked_count = int(len(previous_good))

                if tracked_count >= 10:
                    # Maps points in the current frame to the previous frame.
                    affine, inliers = cv2.estimateAffinePartial2D(
                        current_good,
                        previous_good,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.8,
                        maxIters=2500,
                        confidence=0.995,
                        refineIters=15,
                    )
                    if affine is not None:
                        increment_small = np.eye(3, dtype=np.float64)
                        increment_small[:2, :] = affine
                        scale = self.processing_scale
                        to_small = np.array(
                            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                            dtype=np.float64,
                        )
                        to_full = np.array(
                            [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
                            dtype=np.float64,
                        )
                        increment = to_full @ increment_small @ to_small
                        if inliers is not None and len(inliers):
                            inlier_ratio = float(np.mean(inliers.reshape(-1) > 0))
                        diagonal = float(np.hypot(self.frame_width, self.frame_height))
                        valid = (
                            inlier_ratio >= 0.42
                            and self._is_sane_increment(increment, diagonal)
                        )
                        quality = min(
                            1.0,
                            inlier_ratio * min(1.0, tracked_count / 80.0),
                        )

        if valid and increment is not None:
            self.last_valid_increment = increment
            self.current_to_reference = self.current_to_reference @ increment
        else:
            # Do not extrapolate camera motion aggressively. Keeping the last
            # transform is safer than injecting a bad jump into every object.
            quality = 0.0

        self.previous_gray = gray
        self.previous_mask = mask
        self.last_result = RegistrationResult(
            matrix=self.current_to_reference.copy(),
            valid=True,
            updated=valid,
            quality=quality,
            tracked_points=tracked_count,
            inlier_ratio=inlier_ratio,
        )
        return self.last_result

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([float(x), float(y), 1.0], dtype=np.float64)
        transformed = self.current_to_reference @ point
        denominator = transformed[2] if abs(transformed[2]) > 1e-9 else 1.0
        return float(transformed[0] / denominator), float(transformed[1] / denominator)

    def annotate_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        result = detection.copy()
        box = list(map(float, result.get("bbox_xyxy", [0, 0, 0, 0])))
        x1, y1, x2, y2 = box
        group = str(result.get("class_group", "")).lower()
        if group == "robot":
            anchor_x, anchor_y = (x1 + x2) * 0.5, y2
            anchor_type = "bottom_center"
        else:
            anchor_x, anchor_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            anchor_type = "center"

        stabilized_x, stabilized_y = self.transform_point(anchor_x, anchor_y)
        result.update(
            {
                "anchor_type": anchor_type,
                "anchor_x_px": round(anchor_x, 3),
                "anchor_y_px": round(anchor_y, 3),
                "stabilized_x_px": round(stabilized_x, 3),
                "stabilized_y_px": round(stabilized_y, 3),
                "registration_valid": bool(self.last_result.valid),
                "registration_quality": round(float(self.last_result.quality), 5),
            }
        )
        return result
