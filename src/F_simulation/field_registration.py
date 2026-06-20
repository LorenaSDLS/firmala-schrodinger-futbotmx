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
    model_type: str = "identity"
    forward_backward_error: float = 0.0
    reprojection_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "updated": bool(self.updated),
            "quality": round(float(self.quality), 5),
            "tracked_points": int(self.tracked_points),
            "inlier_ratio": round(float(self.inlier_ratio), 5),
            "model_type": str(self.model_type),
            "forward_backward_error": round(float(self.forward_backward_error), 5),
            "reprojection_error": round(float(self.reprojection_error), 5),
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
        allow_projective: bool = True,
        deterministic_fit: bool = False,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.enabled = bool(enabled)
        self.max_corners = int(max_corners)
        self.allow_projective = bool(allow_projective)
        self.deterministic_fit = bool(deterministic_fit)
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
            green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            white = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 82, 255]))
            # White walls, shirts and windows must never drive the camera pose.
            # Keep white pixels only when they touch the green carpet region;
            # this preserves field markings while rejecting background people.
            near_surface = cv2.dilate(green, np.ones((31, 31), np.uint8))
            white_markings = cv2.bitwise_and(white, near_surface)
            mask = cv2.bitwise_or(green, white_markings)

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

    @staticmethod
    def _is_sane_homography(
        matrix: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """Reject projective increments that can inject catastrophic drift.

        Consecutive views of the same planar table are related by a homography,
        but the per-frame transform must still be modest.  The check is carried
        out on the transformed image quadrilateral instead of interpreting the
        upper-left 2x2 block as an affine transform.
        """
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            return False
        if abs(float(value[2, 2])) < 1e-10:
            return False
        value = value / value[2, 2]
        corners = np.float32(
            [[[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]]
        )
        projected = cv2.perspectiveTransform(corners, value)[0]
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected.astype(np.float32)):
            return False
        original_area = max(1.0, float(width * height))
        area_ratio = abs(float(cv2.contourArea(projected.astype(np.float32)))) / original_area
        diagonal = float(np.hypot(width, height))
        mean_motion = float(np.mean(np.linalg.norm(projected - corners[0], axis=1)))
        maximum_motion = float(np.max(np.linalg.norm(projected - corners[0], axis=1)))
        projective_strength = max(
            abs(float(value[2, 0])) * width,
            abs(float(value[2, 1])) * height,
        )
        return (
            0.62 <= area_ratio <= 1.62
            and mean_motion <= 0.18 * diagonal
            and maximum_motion <= 0.32 * diagonal
            and projective_strength <= 0.12
        )

    @staticmethod
    def _balanced_correspondences(
        previous_points: np.ndarray,
        current_points: np.ndarray,
        fb_error: np.ndarray,
        width: int,
        height: int,
        max_points: int = 160,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep a deterministic, spatially distributed subset for robust fitting.

        Hundreds of almost collinear field features can make generic RANSAC
        unnecessarily expensive and numerically unstable.  The table is split
        into a coarse grid, the best forward/backward tracks are retained per
        cell, and the final set is capped.  This preserves field coverage while
        giving every fit a strict upper bound.
        """
        count = int(len(previous_points))
        if count <= max_points:
            return previous_points, current_points, fb_error
        cols, rows = 8, 6
        x = np.clip((current_points[:, 0] / max(1.0, float(width)) * cols).astype(int), 0, cols - 1)
        y = np.clip((current_points[:, 1] / max(1.0, float(height)) * rows).astype(int), 0, rows - 1)
        selected: list[int] = []
        per_cell = max(2, int(np.ceil(max_points / float(cols * rows))))
        for cell_y in range(rows):
            for cell_x in range(cols):
                indices = np.flatnonzero((x == cell_x) & (y == cell_y))
                if indices.size == 0:
                    continue
                order = indices[np.argsort(fb_error[indices], kind="stable")]
                selected.extend(order[:per_cell].tolist())
        if len(selected) < max_points:
            used = np.zeros(count, dtype=bool)
            used[np.asarray(selected, dtype=int)] = True
            remaining = np.flatnonzero(~used)
            order = remaining[np.argsort(fb_error[remaining], kind="stable")]
            selected.extend(order[: max_points - len(selected)].tolist())
        selected_array = np.asarray(selected[:max_points], dtype=int)
        return (
            previous_points[selected_array],
            current_points[selected_array],
            fb_error[selected_array],
        )

    @staticmethod
    def _deterministic_similarity(
        source_points: np.ndarray,
        destination_points: np.ndarray,
        threshold: float = 2.8,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Bounded robust similarity fit without randomized RANSAC.

        The median displacement removes gross optical-flow outliers first. A
        two-pass Umeyama fit then estimates rotation, uniform scale and
        translation. Runtime is linear in the number of tracks and contains no
        randomized iteration loop, which is useful while building an
        interactive multiframe calibration cache.
        """
        source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
        destination = np.asarray(destination_points, dtype=np.float64).reshape(-1, 2)
        if len(source) < 4 or len(source) != len(destination):
            return None, None

        displacement = destination - source
        median_displacement = np.median(displacement, axis=0)
        radial = np.linalg.norm(displacement - median_displacement, axis=1)
        median_radial = float(np.median(radial))
        mad = 1.4826 * float(np.median(np.abs(radial - median_radial)))
        initial_threshold = max(float(threshold), median_radial + 3.5 * max(0.35, mad))
        active = radial <= initial_threshold
        if int(np.count_nonzero(active)) < 4:
            order = np.argsort(radial, kind="stable")
            active = np.zeros(len(source), dtype=bool)
            active[order[: min(len(source), max(4, len(source) // 2))]] = True

        def fit(mask: np.ndarray) -> np.ndarray | None:
            src = source[mask]
            dst = destination[mask]
            if len(src) < 4:
                return None
            src_mean = np.mean(src, axis=0)
            dst_mean = np.mean(dst, axis=0)
            src_centered = src - src_mean
            dst_centered = dst - dst_mean
            variance = float(np.sum(src_centered * src_centered))
            if variance < 1e-8:
                return None
            covariance = dst_centered.T @ src_centered
            try:
                u, singular, vt = np.linalg.svd(covariance)
            except np.linalg.LinAlgError:
                return None
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
                singular[-1] *= -1.0
            scale = float(np.sum(singular) / variance)
            if not np.isfinite(scale) or scale <= 0.0:
                return None
            translation = dst_mean - scale * (rotation @ src_mean)
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, :2] = scale * rotation
            matrix[:2, 2] = translation
            return matrix

        matrix = fit(active)
        if matrix is None:
            return None, None
        projected = cv2.perspectiveTransform(
            source.astype(np.float32).reshape(1, -1, 2), matrix
        ).reshape(-1, 2)
        error = np.linalg.norm(projected - destination, axis=1)
        refined = error <= max(float(threshold), 2.5 * max(0.35, float(np.median(error))))
        if int(np.count_nonzero(refined)) >= 4:
            refined_matrix = fit(refined)
            if refined_matrix is not None:
                matrix = refined_matrix
                active = refined
        return matrix[:2, :], active.astype(np.uint8).reshape(-1, 1)

    @staticmethod
    def _model_statistics(
        matrix: np.ndarray,
        current_points: np.ndarray,
        previous_points: np.ndarray,
        inliers: np.ndarray | None,
    ) -> tuple[float, float]:
        if inliers is None or len(inliers) == 0:
            return 0.0, float("inf")
        mask = inliers.reshape(-1) > 0
        ratio = float(np.mean(mask))
        if not np.any(mask):
            return ratio, float("inf")
        projected = cv2.perspectiveTransform(
            current_points[mask].astype(np.float32).reshape(1, -1, 2),
            np.asarray(matrix, dtype=np.float64),
        ).reshape(-1, 2)
        error = float(np.median(np.linalg.norm(projected - previous_points[mask], axis=1)))
        return ratio, error

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
        model_type = "held"
        forward_backward_error = 0.0
        reprojection_error = 0.0

        if previous_points is not None and len(previous_points) >= 12:
            lk_parameters = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            current_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                previous_points,
                None,
                **lk_parameters,
            )
            if current_points is not None and status is not None:
                backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.previous_gray,
                    current_points,
                    None,
                    **lk_parameters,
                )
                status = status.reshape(-1).astype(bool)
                if backward_status is not None:
                    status &= backward_status.reshape(-1).astype(bool)
                previous_all = previous_points.reshape(-1, 2)
                current_all = current_points.reshape(-1, 2)
                backward_all = (
                    backward_points.reshape(-1, 2)
                    if backward_points is not None
                    else np.full_like(previous_all, np.nan)
                )
                previous_good = previous_all[status]
                current_good = current_all[status]
                backward_good = backward_all[status]
                error_good = (
                    errors.reshape(-1)[status]
                    if errors is not None
                    else np.zeros(len(previous_good), dtype=np.float32)
                )
                fb_error = np.linalg.norm(backward_good - previous_good, axis=1)
                previous_xy = np.rint(previous_good).astype(np.int32)
                current_xy = np.rint(current_good).astype(np.int32)
                previous_inside = (
                    (previous_xy[:, 0] >= 0)
                    & (previous_xy[:, 0] < self.processing_width)
                    & (previous_xy[:, 1] >= 0)
                    & (previous_xy[:, 1] < self.processing_height)
                )
                current_inside = (
                    (current_xy[:, 0] >= 0)
                    & (current_xy[:, 0] < self.processing_width)
                    & (current_xy[:, 1] >= 0)
                    & (current_xy[:, 1] < self.processing_height)
                )
                previous_support = np.zeros(len(previous_good), dtype=bool)
                current_support = np.zeros(len(current_good), dtype=bool)
                valid_previous_indexes = np.flatnonzero(previous_inside)
                valid_current_indexes = np.flatnonzero(current_inside)
                if valid_previous_indexes.size:
                    previous_support[valid_previous_indexes] = (
                        self.previous_mask[
                            previous_xy[valid_previous_indexes, 1],
                            previous_xy[valid_previous_indexes, 0],
                        ] > 0
                    )
                if valid_current_indexes.size:
                    current_support[valid_current_indexes] = (
                        mask[
                            current_xy[valid_current_indexes, 1],
                            current_xy[valid_current_indexes, 0],
                        ] > 0
                    )
                finite = (
                    np.isfinite(previous_good).all(axis=1)
                    & np.isfinite(current_good).all(axis=1)
                    & np.isfinite(backward_good).all(axis=1)
                    & previous_support
                    & current_support
                    & (error_good < 40.0)
                    & (fb_error < 1.8)
                )
                previous_good = previous_good[finite]
                current_good = current_good[finite]
                fb_error = fb_error[finite]
                tracked_count = int(len(previous_good))
                if tracked_count:
                    forward_backward_error = float(np.median(fb_error))

                if tracked_count >= 10:
                    previous_good, current_good, fb_error = self._balanced_correspondences(
                        previous_good,
                        current_good,
                        fb_error,
                        self.processing_width,
                        self.processing_height,
                        max_points=160,
                    )
                    tracked_count = int(len(previous_good))
                    if tracked_count:
                        forward_backward_error = float(np.median(fb_error))

                    # Consecutive 60-fps frames are normally explained by a
                    # partial affine increment. Fit that bounded model first.
                    if self.deterministic_fit:
                        affine, affine_inliers = self._deterministic_similarity(
                            current_good, previous_good, threshold=2.8
                        )
                    else:
                        # Keep the robust fit deterministic and strictly bounded.
                        # OpenCV's global RNG can otherwise choose a pathological
                        # sequence on nearly collinear table markings.
                        cv2.setRNGSeed(10_000 + max(0, int(frame_index)))
                        affine, affine_inliers = cv2.estimateAffinePartial2D(
                            current_good,
                            previous_good,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.8,
                            maxIters=160,
                            confidence=0.985,
                            refineIters=5,
                        )
                    affine_matrix = None
                    affine_ratio = 0.0
                    affine_error = float("inf")
                    if affine is not None:
                        affine_matrix = np.eye(3, dtype=np.float64)
                        affine_matrix[:2, :] = affine
                        affine_ratio, affine_error = self._model_statistics(
                            affine_matrix, current_good, previous_good, affine_inliers
                        )

                    increment_small = affine_matrix
                    inliers = affine_inliers
                    model_type = "affine" if affine_matrix is not None else "held"
                    chosen_ratio = affine_ratio
                    chosen_error = affine_error

                    # Attempt a projective increment only when affine evidence
                    # is genuinely insufficient. USAC/MAGSAC with a small,
                    # spatially balanced sample has deterministic runtime and
                    # avoids the pathological standard-RANSAC stalls observed
                    # on the real V9 failure video.
                    needs_projective = (
                        self.allow_projective
                        and tracked_count >= 20
                        and (
                            affine_matrix is None
                            or affine_ratio < 0.78
                            or affine_error > 0.75
                        )
                    )
                    if needs_projective:
                        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                        cv2.setRNGSeed(20_000 + max(0, int(frame_index)))
                        homography, homography_inliers = cv2.findHomography(
                            current_good,
                            previous_good,
                            method=method,
                            ransacReprojThreshold=2.5,
                            maxIters=120,
                            confidence=0.985,
                        )
                        if homography is not None and homography_inliers is not None:
                            h_ratio, h_error = self._model_statistics(
                                homography, current_good, previous_good, homography_inliers
                            )
                            h_sane = self._is_sane_homography(
                                homography, self.processing_width, self.processing_height
                            )
                            materially_better = (
                                affine_matrix is None
                                or h_error + 0.08 < 0.82 * affine_error
                                or h_ratio > affine_ratio + 0.12
                            )
                            if h_sane and h_ratio >= 0.46 and materially_better:
                                increment_small = homography.astype(np.float64)
                                inliers = homography_inliers
                                model_type = "homography"
                                chosen_ratio = h_ratio
                                chosen_error = h_error

                    if increment_small is not None:
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
                        inlier_ratio = float(chosen_ratio)
                        reprojection_error = float(chosen_error)
                        diagonal = float(np.hypot(self.frame_width, self.frame_height))
                        sane = (
                            self._is_sane_homography(increment, self.frame_width, self.frame_height)
                            if model_type == "homography"
                            else self._is_sane_increment(increment, diagonal)
                        )
                        valid = inlier_ratio >= 0.42 and sane and reprojection_error <= 3.2
                        quality = float(
                            np.clip(
                                inlier_ratio
                                * min(1.0, tracked_count / 90.0)
                                * np.exp(-forward_backward_error / 1.8)
                                * np.exp(-reprojection_error / 3.0),
                                0.0,
                                1.0,
                            )
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
            model_type=model_type,
            forward_backward_error=forward_backward_error,
            reprojection_error=reprojection_error,
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
