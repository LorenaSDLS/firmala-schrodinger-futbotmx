from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from pathlib import Path

import cv2
import numpy as np

from src.I_field_geometry.field_segmenter import FieldMaskResult
from src.I_field_geometry.template_registration import GoalAnchoredTemplateRegistrar
from src.I_field_geometry.calibration import FieldCalibration
from src.I_field_geometry.field_template import build_template_points


SIDE_NAMES = ("far", "right", "near", "left")
SIDE_CORNER_INDEXES = {
    "far": (0, 1),
    "right": (1, 2),
    "near": (2, 3),
    "left": (3, 0),
}


@dataclass
class FieldGeometryResult:
    valid: bool
    trusted: bool
    measured: bool
    propagated: bool
    confidence: float
    corners_image: np.ndarray | None
    homography_image_to_field: np.ndarray | None
    homography_field_to_image: np.ndarray | None
    mask_coverage: float
    source: str
    line_support: dict[str, int]
    side_visible: dict[str, bool]
    side_status: dict[str, str] = field(default_factory=dict)
    side_confidence: dict[str, float] = field(default_factory=dict)
    side_lines: dict[str, np.ndarray] = field(default_factory=dict)
    rejected_frame_sides: list[str] = field(default_factory=list)
    border_evidence_score: float = 0.0
    white_alignment_score: float = 0.0
    goal_consistency_score: float = 0.0
    visible_template_fraction: float = 0.0
    manual_line_score: float = 0.0
    registration_scope: str = "none"
    geometry_state: str = "surface"
    local_homography_image_to_local: np.ndarray | None = None
    hard_anchor_score: float = 0.0
    hard_anchor_count: int = 0
    feature_match_score: float = 0.0
    feature_match_count: int = 0
    feature_matches: dict[str, float] = field(default_factory=dict)
    field_width: float = 100.0
    field_height: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "trusted": bool(self.trusted),
            "measured": bool(self.measured),
            "propagated": bool(self.propagated),
            "confidence": round(float(self.confidence), 6),
            "source": self.source,
            "mask_coverage": round(float(self.mask_coverage), 6),
            "line_support": {key: int(value) for key, value in self.line_support.items()},
            "side_visible": {key: bool(value) for key, value in self.side_visible.items()},
            "side_status": {key: str(value) for key, value in self.side_status.items()},
            "side_confidence": {
                key: round(float(value), 6) for key, value in self.side_confidence.items()
            },
            "side_lines": {
                key: [round(float(value), 8) for value in line]
                for key, line in self.side_lines.items()
            },
            "rejected_frame_sides": list(self.rejected_frame_sides),
            "border_evidence_score": round(float(self.border_evidence_score), 6),
            "white_alignment_score": round(float(self.white_alignment_score), 6),
            "goal_consistency_score": round(float(self.goal_consistency_score), 6),
            "visible_template_fraction": round(float(self.visible_template_fraction), 6),
            "manual_line_score": round(float(self.manual_line_score), 6),
            "registration_scope": str(self.registration_scope),
            "geometry_state": str(self.geometry_state),
            "hard_anchor_score": round(float(self.hard_anchor_score), 6),
            "hard_anchor_count": int(self.hard_anchor_count),
            "feature_match_score": round(float(self.feature_match_score), 6),
            "feature_match_count": int(self.feature_match_count),
            "feature_matches": {key: round(float(value), 6) for key, value in self.feature_matches.items()},
            "local_homography_image_to_local": (
                [[round(float(value), 10) for value in row] for row in self.local_homography_image_to_local]
                if self.local_homography_image_to_local is not None else None
            ),
            "field_width": round(float(self.field_width), 6),
            "field_height": round(float(self.field_height), 6),
            "corners_image": (
                [[round(float(x), 3), round(float(y), 3)] for x, y in self.corners_image]
                if self.corners_image is not None
                else None
            ),
            "homography_image_to_field": (
                [[round(float(value), 10) for value in row] for row in self.homography_image_to_field]
                if self.homography_image_to_field is not None
                else None
            ),
        }


@dataclass
class _SideObservation:
    line: np.ndarray
    points: np.ndarray
    support: int
    confidence: float
    span_ratio: float


class FieldGeometryEstimator:
    """Robust field-surface registration from a segmentation mask.

    The segmenter says *which pixels are table surface*.  It does not, by
    itself, prove that the edge of the visible mask is a physical table edge.
    This estimator therefore accepts a side only when the mask boundary has
    image evidence of a dark/black rail immediately outside it.  Image borders
    are explicitly rejected and missing sides are propagated or extrapolated.

    Canonical coordinates use ``x`` from the camera-near goal (0) to the far
    goal (field_width), and ``y`` across the table (0..field_height).
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        field_width: float = 100.0,
        field_height: float = 60.0,
        camera_margin_ratio: float = 0.032,
        minimum_side_points: int = 6,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.field_width = float(field_width)
        self.field_height = float(field_height)
        self.camera_margin_ratio = float(np.clip(camera_margin_ratio, 0.01, 0.10))
        self.minimum_side_points = max(4, int(minimum_side_points))

        self.template_registrar = GoalAnchoredTemplateRegistrar(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )
        self.manual_calibration: FieldCalibration | None = None
        self.manual_calibration_active = False
        if calibration_path is not None:
            calibration_file = Path(calibration_path).expanduser().resolve()
            if not calibration_file.exists():
                raise FileNotFoundError(
                    f"No se encontró la calibración de cancha: {calibration_file}"
                )
            self.manual_calibration = FieldCalibration.load(calibration_file).scaled_to(
                self.frame_width, self.frame_height
            )
            # The saved physical scale remains authoritative when a calibration
            # file is used.
            self.field_width = float(self.manual_calibration.field_width)
            self.field_height = float(self.manual_calibration.field_height)

        self.reference_to_field: np.ndarray | None = None
        self.reference_to_local: np.ndarray | None = None
        self.last_homography: np.ndarray | None = None
        self.last_local_homography: np.ndarray | None = None
        self.last_corners: np.ndarray | None = None
        self.frames_since_measurement = 10_000
        self.last_surface_mask_image: np.ndarray | None = None
        self.last_result = self._empty_result("sin_calibracion")

    def _empty_result(self, source: str, coverage: float = 0.0) -> FieldGeometryResult:
        return FieldGeometryResult(
            valid=False,
            trusted=False,
            measured=False,
            propagated=False,
            confidence=0.0,
            corners_image=None,
            homography_image_to_field=None,
            homography_field_to_image=None,
            mask_coverage=coverage,
            source=source,
            line_support={},
            side_visible={side: False for side in SIDE_NAMES},
            side_status={side: "desconocido" for side in SIDE_NAMES},
            side_confidence={side: 0.0 for side in SIDE_NAMES},
            side_lines={},
            rejected_frame_sides=[],
            field_width=self.field_width,
            field_height=self.field_height,
        )

    def _local_result(
        self,
        source: str,
        coverage: float,
        local_homography: np.ndarray | None,
        confidence: float = 0.35,
        manual_line_score: float = 0.0,
        hard_anchor_score: float = 0.0,
        hard_anchor_count: int = 0,
        feature_match_score: float = 0.0,
        feature_match_count: int = 0,
        feature_matches: dict[str, float] | None = None,
        side_lines: dict[str, np.ndarray] | None = None,
        rejected_frame_sides: list[str] | None = None,
        measured: bool = True,
        propagated: bool = False,
    ) -> FieldGeometryResult:
        result = FieldGeometryResult(
            valid=False,
            trusted=False,
            measured=bool(measured),
            propagated=bool(propagated),
            confidence=float(np.clip(confidence, 0.0, 0.49)),
            corners_image=None,
            homography_image_to_field=None,
            homography_field_to_image=None,
            mask_coverage=float(coverage),
            source=source,
            line_support={},
            side_visible={side: False for side in SIDE_NAMES},
            side_status={side: "evidencia_local" for side in SIDE_NAMES},
            side_confidence={side: 0.0 for side in SIDE_NAMES},
            side_lines=side_lines or {},
            rejected_frame_sides=list(rejected_frame_sides or []),
            border_evidence_score=0.0,
            white_alignment_score=float(feature_match_score),
            goal_consistency_score=0.0,
            visible_template_fraction=0.0,
            manual_line_score=float(manual_line_score),
            registration_scope="local" if local_homography is not None else "surface",
            geometry_state="local" if local_homography is not None else "surface",
            local_homography_image_to_local=(
                None if local_homography is None
                else np.asarray(local_homography, dtype=np.float64).reshape(3, 3)
            ),
            hard_anchor_score=float(hard_anchor_score),
            hard_anchor_count=int(hard_anchor_count),
            feature_match_score=float(feature_match_score),
            feature_match_count=int(feature_match_count),
            feature_matches=dict(feature_matches or {}),
            field_width=self.field_width,
            field_height=self.field_height,
        )
        self.last_result = result
        return result

    def _trusted_result_from_corners(
        self,
        corners: np.ndarray,
        homography: np.ndarray,
        source: str,
        measured: bool,
        confidence: float = 0.995,
        coverage: float = 0.0,
    ) -> FieldGeometryResult:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
        inverse = self._safe_inverse(homography)
        if inverse is None:
            return self._empty_result("calibracion_invalida", coverage)
        lines = self._lines_from_corners(corners)
        status_value = "calibrado" if measured else "propagado_calibracion"
        result = FieldGeometryResult(
            valid=True,
            trusted=True,
            measured=bool(measured),
            propagated=not bool(measured),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            corners_image=corners,
            homography_image_to_field=homography,
            homography_field_to_image=inverse,
            mask_coverage=float(coverage),
            source=source,
            line_support={side: 100 for side in SIDE_NAMES},
            side_visible={side: bool(measured) for side in SIDE_NAMES},
            side_status={side: status_value for side in SIDE_NAMES},
            side_confidence={side: float(confidence) for side in SIDE_NAMES},
            side_lines=lines,
            rejected_frame_sides=[],
            border_evidence_score=1.0 if measured else 0.92,
            white_alignment_score=1.0 if measured else 0.90,
            goal_consistency_score=1.0,
            visible_template_fraction=1.0,
            manual_line_score=1.0 if measured else 0.95,
            registration_scope="full",
            geometry_state="global",
            hard_anchor_score=1.0 if measured else 0.95,
            hard_anchor_count=4,
            feature_match_score=1.0 if measured else 0.90,
            feature_match_count=4,
            field_width=self.field_width,
            field_height=self.field_height,
        )
        self.last_homography = homography.copy()
        self.last_corners = corners.copy()
        self.last_result = result
        return result

    def _activate_manual_calibration(
        self,
        frame_index: int | None,
        current_to_reference: np.ndarray | None,
    ) -> FieldGeometryResult | None:
        calibration = self.manual_calibration
        if calibration is None or self.manual_calibration_active:
            return None
        if frame_index is None or frame_index != calibration.source_frame_index:
            return None
        current_to_reference = (
            np.asarray(current_to_reference, dtype=np.float64)
            if current_to_reference is not None
            else np.eye(3, dtype=np.float64)
        )
        inverse_registration = self._safe_inverse(current_to_reference)
        if inverse_registration is None:
            return None
        self.manual_calibration_active = True
        self.frames_since_measurement = 0
        if not calibration.is_complete:
            # Partial V8 annotations are semantic constraints only. They never
            # fabricate the missing corners or claim a full homography.
            return None
        homography = calibration.homography_image_to_field.copy()
        self.reference_to_field = homography @ inverse_registration
        return self._trusted_result_from_corners(
            calibration.corners_image,
            homography,
            source=calibration.source,
            measured=True,
            confidence=0.999,
        )

    @property
    def needs_reacquisition(self) -> bool:
        if (
            self.last_result.geometry_state == "local"
            and self.last_result.local_homography_image_to_local is not None
            and self.last_result.confidence >= 0.18
        ):
            return False
        return (not self.last_result.valid) or self.last_result.confidence < 0.28

    @property
    def canonical_corners(self) -> np.ndarray:
        # Order: far-left, far-right, near-right, near-left.
        return np.float32(
            [
                [self.field_width, 0.0],
                [self.field_width, self.field_height],
                [0.0, self.field_height],
                [0.0, 0.0],
            ]
        )

    @staticmethod
    def _largest_clean_mask(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(binary)
        contour = max(contours, key=cv2.contourArea)
        cleaned = np.zeros_like(binary)
        cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
        return cleaned

    @staticmethod
    def _safe_inverse(matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None
        return inverse if np.isfinite(inverse).all() else None

    @staticmethod
    def _normalize_line(line: np.ndarray) -> np.ndarray:
        line = np.asarray(line, dtype=np.float64).reshape(3)
        norm = float(np.hypot(line[0], line[1]))
        if norm < 1e-9:
            return line
        return line / norm

    @staticmethod
    def _orient_line_toward_interior(line: np.ndarray, interior: np.ndarray) -> np.ndarray:
        line = FieldGeometryEstimator._normalize_line(line)
        value = float(line[0] * interior[0] + line[1] * interior[1] + line[2])
        # Interior is represented by negative signed distance.
        return -line if value > 0.0 else line

    @staticmethod
    def _robust_fit_line(points: np.ndarray) -> tuple[np.ndarray, int, float] | None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4:
            return None
        retained = points
        for _ in range(5):
            vx, vy, x0, y0 = cv2.fitLine(
                retained, cv2.DIST_HUBER, 0, 0.01, 0.01
            ).reshape(-1)
            direction = np.array([vx, vy], dtype=np.float64)
            direction /= max(np.linalg.norm(direction), 1e-9)
            origin = np.array([x0, y0], dtype=np.float64)
            delta = retained.astype(np.float64) - origin
            distances = np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            threshold = max(2.0, median + 2.8 * max(mad, 0.8))
            next_retained = retained[distances <= threshold]
            if len(next_retained) < 4 or len(next_retained) == len(retained):
                break
            retained = next_retained

        vx, vy, x0, y0 = cv2.fitLine(
            retained, cv2.DIST_HUBER, 0, 0.01, 0.01
        ).reshape(-1)
        a = float(vy)
        b = float(-vx)
        c = -(a * float(x0) + b * float(y0))
        line = FieldGeometryEstimator._normalize_line(np.array([a, b, c]))
        residuals = np.abs(retained @ line[:2] + line[2])
        residual = float(np.median(residuals)) if len(residuals) else 999.0
        return line, int(len(retained)), residual

    @staticmethod
    def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
        a1, b1, c1 = first
        a2, b2, c2 = second
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-7:
            return None
        x = (b1 * c2 - b2 * c1) / determinant
        y = (c1 * a2 - c2 * a1) / determinant
        point = np.array([x, y], dtype=np.float32)
        return point if np.isfinite(point).all() else None

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        order_y = np.argsort(corners[:, 1])
        far = corners[order_y[:2]]
        near = corners[order_y[2:]]
        far = far[np.argsort(far[:, 0])]
        near = near[np.argsort(near[:, 0])]
        far_left, far_right = far
        near_left, near_right = near
        return np.float32([far_left, far_right, near_right, near_left])

    @staticmethod
    def _line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
        first_h = np.array([float(first[0]), float(first[1]), 1.0])
        second_h = np.array([float(second[0]), float(second[1]), 1.0])
        line = np.cross(first_h, second_h)
        if float(np.hypot(line[0], line[1])) < 1e-8:
            return None
        return FieldGeometryEstimator._normalize_line(line)

    def _lines_from_corners(self, corners: np.ndarray) -> dict[str, np.ndarray]:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        lines: dict[str, np.ndarray] = {}
        interior = np.mean(corners, axis=0)
        for side, (first_index, second_index) in SIDE_CORNER_INDEXES.items():
            line = self._line_from_points(corners[first_index], corners[second_index])
            if line is not None:
                lines[side] = self._orient_line_toward_interior(line, interior)
        return lines

    def _corners_from_lines(self, lines: dict[str, np.ndarray]) -> np.ndarray | None:
        if not set(SIDE_NAMES).issubset(lines):
            return None
        corners = [
            self._intersection(lines["far"], lines["left"]),
            self._intersection(lines["far"], lines["right"]),
            self._intersection(lines["near"], lines["right"]),
            self._intersection(lines["near"], lines["left"]),
        ]
        if any(point is None for point in corners):
            return None
        # The intersections are already in semantic order:
        # far-left, far-right, near-right, near-left. Reordering by image Y
        # breaks as soon as one or more corners lie outside the frame.
        return np.asarray(corners, dtype=np.float32)

    @staticmethod
    def _quad_area(corners: np.ndarray) -> float:
        return abs(float(cv2.contourArea(np.asarray(corners, dtype=np.float32))))

    def _camera_margins(self) -> tuple[int, int]:
        margin_x = max(8, int(round(self.camera_margin_ratio * self.frame_width)))
        margin_y = max(8, int(round(self.camera_margin_ratio * self.frame_height)))
        return margin_x, margin_y

    def _is_point_at_camera_edge(self, point: np.ndarray, side: str) -> bool:
        margin_x, margin_y = self._camera_margins()
        x, y = map(float, point)
        if side == "far":
            return y <= margin_y
        if side == "near":
            return y >= self.frame_height - 1 - margin_y
        if side == "left":
            return x <= margin_x
        if side == "right":
            return x >= self.frame_width - 1 - margin_x
        return False

    def _physical_edge_support_map(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical-border evidence, dark-rail mask, and image-edge map."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # The rail is usually black or neutral dark gray.  The second clause
        # accepts gray rails under bright exposure without accepting green turf.
        dark = ((value < 132) | ((value < 170) & (saturation < 72))).astype(np.uint8)

        binary = (mask > 0).astype(np.uint8)
        eroded = cv2.erode(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        boundary = cv2.subtract(binary, eroded)
        dilated = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            iterations=1,
        )
        outside_band = ((dilated > 0) & (binary == 0)).astype(np.uint8)
        dark_outside = ((dark > 0) & (outside_band > 0)).astype(np.uint8)

        # Distance to a dark pixel outside the segmented surface.
        no_dark = (dark_outside == 0).astype(np.uint8)
        distance_to_dark = cv2.distanceTransform(no_dark, cv2.DIST_L2, 3)

        edges = cv2.Canny(gray, 45, 135)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        distance_score = np.exp(-distance_to_dark / 8.0)
        edge_score = (edges > 0).astype(np.float32)
        support = boundary.astype(np.float32) * distance_score * (0.68 + 0.32 * edge_score)

        margin_x, margin_y = self._camera_margins()
        support[:margin_y, :] = 0.0
        support[-margin_y:, :] = 0.0
        support[:, :margin_x] = 0.0
        support[:, -margin_x:] = 0.0
        return support, dark_outside * 255, edges

    @staticmethod
    def _local_support(score_map: np.ndarray, point: np.ndarray, radius: int = 16) -> float:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        height, width = score_map.shape[:2]
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        patch = score_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        # A maximum catches a rail that is a few pixels outside an imperfect mask.
        return float(0.55 * np.max(patch) + 0.45 * np.mean(patch))

    def _raw_envelope_points(self, mask: np.ndarray) -> dict[str, np.ndarray]:
        height, width = mask.shape[:2]
        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            return {}
        points: dict[str, list[list[float]]] = {side: [] for side in SIDE_NAMES}

        for start, end in zip(
            np.linspace(0, width, 57, dtype=int)[:-1],
            np.linspace(0, width, 57, dtype=int)[1:],
        ):
            selection = (xs >= start) & (xs < end)
            if not np.any(selection):
                continue
            current_x = float(np.median(xs[selection]))
            current_y = ys[selection]
            points["far"].append([current_x, float(np.percentile(current_y, 2.5))])
            points["near"].append([current_x, float(np.percentile(current_y, 97.5))])

        for start, end in zip(
            np.linspace(0, height, 49, dtype=int)[:-1],
            np.linspace(0, height, 49, dtype=int)[1:],
        ):
            selection = (ys >= start) & (ys < end)
            if not np.any(selection):
                continue
            current_y = float(np.median(ys[selection]))
            current_x = xs[selection]
            points["left"].append([float(np.percentile(current_x, 2.5)), current_y])
            points["right"].append([float(np.percentile(current_x, 97.5)), current_y])

        return {
            side: np.asarray(values, dtype=np.float32)
            for side, values in points.items()
            if len(values) >= 4
        }

    def _classify_physical_segment(
        self,
        first: np.ndarray,
        second: np.ndarray,
        interior: np.ndarray,
    ) -> tuple[str, np.ndarray] | None:
        line = self._line_from_points(first, second)
        if line is None:
            return None
        line = self._orient_line_toward_interior(line, interior)
        dx = float(second[0] - first[0])
        dy = float(second[1] - first[1])
        if abs(dx) + abs(dy) < 1e-6:
            return None

        # Tangent orientation separates the two projective line families.
        # Horizontal-ish segments are end borders (far/near); steep segments
        # are side borders (left/right).  The outward normal, encoded by the
        # oriented line coefficients, selects the concrete side.
        if abs(dx) >= 1.08 * abs(dy):
            side = "far" if line[1] < 0.0 else "near"
        elif abs(dy) >= 1.08 * abs(dx):
            side = "left" if line[0] < 0.0 else "right"
        else:
            normal_x = float(line[0] * self.frame_width)
            normal_y = float(line[1] * self.frame_height)
            if abs(normal_y) >= abs(normal_x):
                side = "far" if normal_y < 0.0 else "near"
            else:
                side = "left" if normal_x < 0.0 else "right"
        return side, line

    def _observe_physical_sides(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[
        dict[str, _SideObservation],
        dict[str, int],
        dict[str, bool],
        list[str],
        np.ndarray,
    ]:
        support_map, _dark_mask, _edge_map = self._physical_edge_support_map(frame, mask)
        raw_points = self._raw_envelope_points(mask)
        side_visible = {side: False for side in SIDE_NAMES}
        rejected_frame_sides: list[str] = []
        for side, points in raw_points.items():
            edge_fraction = float(
                np.mean([self._is_point_at_camera_edge(point, side) for point in points])
            )
            if edge_fraction >= 0.30:
                rejected_frame_sides.append(side)

        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            return {}, {}, side_visible, rejected_frame_sides, support_map
        interior = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)

        binary_support = (support_map >= 0.045).astype(np.uint8) * 255
        binary_support = cv2.morphologyEx(
            binary_support,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        )
        diagonal = float(np.hypot(self.frame_width, self.frame_height))
        hough = cv2.HoughLinesP(
            binary_support,
            1,
            np.pi / 360.0,
            threshold=max(18, int(round(0.014 * diagonal))),
            minLineLength=max(45, int(round(0.045 * diagonal))),
            maxLineGap=max(18, int(round(0.025 * diagonal))),
        )

        grouped_points: dict[str, list[np.ndarray]] = {side: [] for side in SIDE_NAMES}
        grouped_lengths: dict[str, float] = {side: 0.0 for side in SIDE_NAMES}
        grouped_support: dict[str, list[float]] = {side: [] for side in SIDE_NAMES}

        if hough is not None:
            for segment in hough[:, 0, :]:
                x1, y1, x2, y2 = map(float, segment)
                first = np.array([x1, y1], dtype=np.float32)
                second = np.array([x2, y2], dtype=np.float32)
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 0.035 * diagonal:
                    continue
                classified = self._classify_physical_segment(first, second, interior)
                if classified is None:
                    continue
                side, _line = classified

                # Sample the complete segment, not just its endpoints.  This
                # gives the robust fit a length-weighted set of rail pixels.
                samples = max(3, int(round(length / 22.0)))
                for t in np.linspace(0.0, 1.0, samples):
                    point = (1.0 - t) * first + t * second
                    if self._is_point_at_camera_edge(point, side):
                        continue
                    local = self._local_support(support_map, point, radius=10)
                    if local < 0.035:
                        continue
                    grouped_points[side].append(point)
                    grouped_support[side].append(local)
                grouped_lengths[side] += length

        observations: dict[str, _SideObservation] = {}
        support_counts: dict[str, int] = {}
        height, width = mask.shape[:2]
        for side in SIDE_NAMES:
            if len(grouped_points[side]) < self.minimum_side_points:
                support_counts[side] = len(grouped_points[side])
                continue
            points = np.asarray(grouped_points[side], dtype=np.float32)
            fitted = self._robust_fit_line(points)
            if fitted is None:
                support_counts[side] = len(points)
                continue
            line, retained, residual = fitted
            if retained < self.minimum_side_points:
                support_counts[side] = retained
                continue
            line = self._orient_line_toward_interior(line, interior)
            if side in {"far", "near"}:
                span = float(np.ptp(points[:, 0])) / max(1.0, float(width))
                length_norm = grouped_lengths[side] / max(1.0, float(width))
            else:
                span = float(np.ptp(points[:, 1])) / max(1.0, float(height))
                length_norm = grouped_lengths[side] / max(1.0, float(height))
            if span < 0.10 or length_norm < 0.10:
                support_counts[side] = retained
                continue

            mean_support = float(np.mean(grouped_support[side]))
            residual_score = float(np.exp(-residual / 5.5))
            confidence = float(
                np.clip(
                    0.38 * min(1.0, mean_support / 0.32)
                    + 0.29 * min(1.0, span / 0.48)
                    + 0.18 * min(1.0, length_norm / 0.85)
                    + 0.15 * residual_score,
                    0.0,
                    1.0,
                )
            )
            if confidence < 0.28:
                support_counts[side] = retained
                continue
            observations[side] = _SideObservation(
                line=line,
                points=points,
                support=retained,
                confidence=confidence,
                span_ratio=span,
            )
            support_counts[side] = retained
            side_visible[side] = True

        return observations, support_counts, side_visible, rejected_frame_sides, support_map

    def _observe_mask_sides_without_image(
        self,
        mask: np.ndarray,
    ) -> tuple[
        dict[str, _SideObservation],
        dict[str, int],
        dict[str, bool],
        list[str],
    ]:
        """Compatibility fallback for complete synthetic or precomputed masks.

        Without the RGB frame we cannot verify a black rail, so only sides that
        are clearly separated from the camera margins are accepted.  A clipped
        mask still cannot bootstrap a homography.
        """
        raw_points = self._raw_envelope_points(mask)
        observations: dict[str, _SideObservation] = {}
        support_counts: dict[str, int] = {}
        side_visible = {side: False for side in SIDE_NAMES}
        rejected: list[str] = []
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return observations, support_counts, side_visible, rejected
        interior = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
        height, width = mask.shape[:2]

        for side, points in raw_points.items():
            edge_fraction = float(
                np.mean([self._is_point_at_camera_edge(point, side) for point in points])
            )
            if edge_fraction >= 0.30:
                rejected.append(side)
            selected = np.asarray(
                [point for point in points if not self._is_point_at_camera_edge(point, side)],
                dtype=np.float32,
            )
            support_counts[side] = int(len(selected))
            if len(selected) < self.minimum_side_points:
                continue
            if side in {"far", "near"}:
                span = float(np.ptp(selected[:, 0])) / max(1.0, float(width))
            else:
                span = float(np.ptp(selected[:, 1])) / max(1.0, float(height))
            if span < 0.12:
                continue
            fitted = self._robust_fit_line(selected)
            if fitted is None:
                continue
            line, retained, residual = fitted
            line = self._orient_line_toward_interior(line, interior)
            confidence = float(
                np.clip(
                    0.58 + 0.22 * min(1.0, span / 0.55)
                    + 0.20 * np.exp(-residual / 5.5),
                    0.0,
                    0.88,
                )
            )
            observations[side] = _SideObservation(
                line=line,
                points=selected,
                support=retained,
                confidence=confidence,
                span_ratio=span,
            )
            support_counts[side] = retained
            side_visible[side] = True
        return observations, support_counts, side_visible, rejected

    def _fit_raw_quad(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for fraction in np.linspace(0.008, 0.085, 36):
            polygon = cv2.approxPolyDP(hull, float(fraction * perimeter), True)
            if len(polygon) == 4:
                return self._order_corners(polygon.reshape(4, 2))

        rectangle = cv2.minAreaRect(hull)
        return self._order_corners(cv2.boxPoints(rectangle))

    def _extrapolated_bootstrap_candidates(
        self,
        raw_corners: np.ndarray,
        observed: dict[str, _SideObservation],
        rejected_sides: Iterable[str],
    ) -> list[np.ndarray]:
        rejected = set(rejected_sides)
        observed_names = set(observed)
        missing = {side for side in SIDE_NAMES if side not in observed_names}
        base = self._order_corners(raw_corners).astype(np.float64)
        candidates: list[np.ndarray] = []

        # Extend clipped sides from the visible quadrilateral without repeatedly
        # multiplying already-extended coordinates.  This keeps inferred corners
        # plausible while still allowing them to lie outside the video.
        longitudinal_steps = (0.18, 0.32, 0.50, 0.72)
        lateral_steps = (0.12, 0.24, 0.38, 0.55)
        for long_step in longitudinal_steps:
            for lateral_step in lateral_steps:
                far_left, far_right, near_right, near_left = [point.copy() for point in base]

                if "near" in missing or "near" in rejected:
                    near_left = near_left + long_step * (near_left - far_left)
                    near_right = near_right + long_step * (near_right - far_right)
                if "far" in missing or "far" in rejected:
                    far_left = far_left + long_step * (far_left - near_left)
                    far_right = far_right + long_step * (far_right - near_right)
                if "left" in missing or "left" in rejected:
                    far_left = far_left + lateral_step * (far_left - far_right)
                    near_left = near_left + lateral_step * (near_left - near_right)
                if "right" in missing or "right" in rejected:
                    far_right = far_right + lateral_step * (far_right - far_left)
                    near_right = near_right + lateral_step * (near_right - near_left)

                # The four points remain in semantic order while they are
                # extended off-screen.  Image-Y sorting would scramble them.
                candidate = np.float32([far_left, far_right, near_right, near_left])
                candidate_lines = self._lines_from_corners(candidate)
                candidate_lines.update({side: obs.line for side, obs in observed.items()})
                rebuilt = self._corners_from_lines(candidate_lines)
                if rebuilt is not None and self._candidate_is_sane(rebuilt):
                    candidates.append(rebuilt)
        return candidates

    def _predicted_homography(self, current_to_reference: np.ndarray | None) -> np.ndarray | None:
        if self.reference_to_field is None or current_to_reference is None:
            return None
        candidate = self.reference_to_field @ current_to_reference
        return candidate if np.isfinite(candidate).all() else None

    def _predicted_corners(self, current_to_reference: np.ndarray | None) -> np.ndarray | None:
        predicted_h = self._predicted_homography(current_to_reference)
        inverse = self._safe_inverse(predicted_h)
        if inverse is None:
            return None
        corners = cv2.perspectiveTransform(
            self.canonical_corners.reshape(1, -1, 2), inverse
        ).reshape(-1, 2)
        return corners.astype(np.float32) if np.isfinite(corners).all() else None

    @staticmethod
    def _blend_lines(
        predicted: np.ndarray,
        observed: np.ndarray,
        observed_weight: float,
    ) -> np.ndarray:
        predicted = FieldGeometryEstimator._normalize_line(predicted)
        observed = FieldGeometryEstimator._normalize_line(observed)
        if float(np.dot(predicted[:2], observed[:2])) < 0.0:
            observed = -observed
        weight = float(np.clip(observed_weight, 0.05, 0.92))
        blended = (1.0 - weight) * predicted + weight * observed
        return FieldGeometryEstimator._normalize_line(blended)

    def _white_mask(self, frame: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        white = ((saturation < 88) & (value > 145) & (field_mask > 0)).astype(np.uint8) * 255
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return white

    def _white_alignment_score(
        self,
        frame: np.ndarray | None,
        mask: np.ndarray,
        homography: np.ndarray,
    ) -> float:
        if frame is None:
            return 0.0
        white = self._white_mask(frame, mask)
        if np.count_nonzero(white) < 80:
            return 0.0
        output_width, output_height = 500, 300
        scale = np.array(
            [
                [output_width / max(self.field_width, 1e-9), 0.0, 0.0],
                [0.0, output_height / max(self.field_height, 1e-9), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        warped = cv2.warpPerspective(
            white,
            scale @ homography,
            (output_width, output_height),
            flags=cv2.INTER_NEAREST,
        )
        lines = cv2.HoughLinesP(
            warped,
            1,
            np.pi / 180.0,
            threshold=28,
            minLineLength=28,
            maxLineGap=12,
        )
        if lines is None:
            return 0.0
        aligned_length = 0.0
        total_length = 0.0
        for segment in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, segment)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 10.0:
                continue
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
            distance_to_axis = min(angle, abs(90.0 - angle), abs(180.0 - angle))
            axis_score = float(np.exp(-0.5 * (distance_to_axis / 9.0) ** 2))
            total_length += length
            aligned_length += length * axis_score
        if total_length < 40.0:
            return 0.0
        density_factor = min(1.0, total_length / 650.0)
        return float(np.clip((aligned_length / total_length) * (0.55 + 0.45 * density_factor), 0.0, 1.0))

    def _goal_consistency_score(
        self,
        homography: np.ndarray,
        goal_detections: Iterable[dict[str, Any]] | None,
    ) -> float:
        """Score goal placement without letting one good goal hide a bad one.

        Goal detections cover the physical goal structure, not an exact point on
        the goal line.  They are therefore treated as *soft* end/orientation
        evidence.  With two goals, both must land at opposite longitudinal ends
        of the canonical field; mapping both to the same end is strongly
        penalized.  A single visible goal can only provide weak evidence.
        """
        goals: list[tuple[np.ndarray, float, float]] = []
        for detection in goal_detections or []:
            if str(detection.get("class_group", "")).lower() != "goal":
                continue
            box = detection.get("bbox_xyxy", [0, 0, 0, 0])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue
            image_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            transformed = cv2.perspectiveTransform(
                image_center.reshape(1, 1, 2), homography
            )[0, 0]
            if not np.isfinite(transformed).all():
                continue
            x_norm = float(transformed[0]) / max(self.field_width, 1e-9)
            y_norm = float(transformed[1]) / max(self.field_height, 1e-9)
            goals.append((image_center, x_norm, y_norm))

        if not goals:
            return 0.0

        def end_and_lateral_score(x_norm: float, y_norm: float, target_x: float) -> float:
            # Goal centers are expected slightly inside the canonical end, not
            # exactly on x=0/1, because the detector box includes goal depth.
            end_score = float(np.exp(-0.5 * ((x_norm - target_x) / 0.18) ** 2))
            lateral_score = float(np.exp(-0.5 * ((y_norm - 0.5) / 0.42) ** 2))
            return 0.84 * end_score + 0.16 * lateral_score

        if len(goals) == 1:
            image_center, x_norm, y_norm = goals[0]
            target_x = 0.92 if image_center[1] < 0.58 * self.frame_height else 0.08
            # One goal helps orientation, but cannot validate the whole field.
            return float(min(0.45, end_and_lateral_score(x_norm, y_norm, target_x)))

        # In the current camera convention, the visually higher goal is the far
        # goal and the visually lower one is the near goal.  Use the two most
        # separated detections in image Y when more than two candidates exist.
        ordered = sorted(goals, key=lambda item: float(item[0][1]))
        far_goal = ordered[0]
        near_goal = ordered[-1]
        far_score = end_and_lateral_score(far_goal[1], far_goal[2], 0.92)
        near_score = end_and_lateral_score(near_goal[1], near_goal[2], 0.08)

        longitudinal_separation = far_goal[1] - near_goal[1]
        separation_score = float(
            np.clip((longitudinal_separation - 0.18) / 0.55, 0.0, 1.0)
        )
        opposite_ends_score = float(
            np.exp(-0.5 * ((longitudinal_separation - 0.84) / 0.32) ** 2)
        )
        both_score = 0.55 * min(far_score, near_score) + 0.45 * (
            0.5 * (far_score + near_score)
        )
        return float(
            np.clip(
                0.62 * both_score
                + 0.23 * separation_score
                + 0.15 * opposite_ends_score,
                0.0,
                1.0,
            )
        )

    def _mask_recall(self, corners: np.ndarray, mask: np.ndarray) -> float:
        quad_mask = np.zeros_like(mask)
        clipped = corners.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], -3 * self.frame_width, 4 * self.frame_width)
        clipped[:, 1] = np.clip(clipped[:, 1], -3 * self.frame_height, 4 * self.frame_height)
        cv2.fillConvexPoly(quad_mask, np.round(clipped).astype(np.int32), 255)
        intersection = np.count_nonzero((quad_mask > 0) & (mask > 0))
        return float(intersection / max(1.0, float(np.count_nonzero(mask))))

    def _candidate_is_sane(self, corners: np.ndarray) -> bool:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
            return False
        area = self._quad_area(corners)
        frame_area = float(self.frame_width * self.frame_height)
        if area < 0.08 * frame_area or area > 12.0 * frame_area:
            return False
        if np.any(corners[:, 0] < -1.8 * self.frame_width) or np.any(
            corners[:, 0] > 2.8 * self.frame_width
        ):
            return False
        if np.any(corners[:, 1] < -1.8 * self.frame_height) or np.any(
            corners[:, 1] > 2.8 * self.frame_height
        ):
            return False
        return True

    def _sample_points_on_line(
        self,
        line: np.ndarray,
        samples: int = 5,
    ) -> np.ndarray:
        segment = _line_segment_in_frame(
            line,
            self.frame_width,
            self.frame_height,
        )
        if segment is None:
            return np.empty((0, 2), dtype=np.float64)
        first = np.asarray(segment[0], dtype=np.float64)
        second = np.asarray(segment[1], dtype=np.float64)
        return np.asarray(
            [(1.0 - t) * first + t * second for t in np.linspace(0.08, 0.92, samples)],
            dtype=np.float64,
        )

    def _goal_anchor_constraints(
        self,
        goal_detections: Iterable[dict[str, Any]] | None,
        observed_lines: dict[str, np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        goals: list[tuple[np.ndarray, float]] = []
        for detection in goal_detections or []:
            if str(detection.get("class_group", "")).lower() != "goal":
                continue
            box = detection.get("bbox_xyxy", [0, 0, 0, 0])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            # Use the box center as a soft orientation anchor.  The box can be
            # clipped and includes goal depth, so it must not be treated as an
            # exact point on the physical goal line.
            anchor = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            confidence = float(detection.get("confidence", 0.5))
            goals.append((anchor, confidence))
        if not goals:
            return []

        constraints: list[tuple[np.ndarray, np.ndarray, float]] = []
        if len(goals) >= 2:
            # Prefer distance to the observed far border when available; image
            # vertical order is the fallback for handheld portrait footage.
            far_line = observed_lines.get("far")
            if far_line is not None:
                ordered = sorted(
                    goals,
                    key=lambda item: abs(
                        float(item[0][0] * far_line[0] + item[0][1] * far_line[1] + far_line[2])
                    ),
                )
                far_goal = ordered[0]
                near_goal = max(ordered[1:], key=lambda item: item[0][1])
            else:
                ordered = sorted(goals, key=lambda item: item[0][1])
                far_goal, near_goal = ordered[0], ordered[-1]
            constraints.append(
                (
                    far_goal[0],
                    np.array([0.92 * self.field_width, 0.5 * self.field_height]),
                    far_goal[1],
                )
            )
            constraints.append(
                (
                    near_goal[0],
                    np.array([0.08 * self.field_width, 0.5 * self.field_height]),
                    near_goal[1],
                )
            )
        else:
            anchor, confidence = goals[0]
            target_x = 0.92 * self.field_width if anchor[1] < 0.58 * self.frame_height else 0.08 * self.field_width
            constraints.append(
                (
                    anchor,
                    np.array([target_x, 0.5 * self.field_height]),
                    confidence * 0.65,
                )
            )
        return constraints

    def _white_segments_for_refinement(
        self,
        frame: np.ndarray | None,
        mask: np.ndarray,
        initial_h: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        if frame is None:
            return []
        white = self._white_mask(frame, mask)
        lines = cv2.HoughLinesP(
            white,
            1,
            np.pi / 180.0,
            threshold=38,
            minLineLength=max(35, int(0.055 * np.hypot(self.frame_width, self.frame_height))),
            maxLineGap=18,
        )
        if lines is None:
            return []
        segments: list[tuple[float, np.ndarray, np.ndarray, str]] = []
        for raw in lines[:, 0, :]:
            first = np.array([float(raw[0]), float(raw[1])], dtype=np.float64)
            second = np.array([float(raw[2]), float(raw[3])], dtype=np.float64)
            length = float(np.linalg.norm(second - first))
            if length < 30.0:
                continue
            transformed = cv2.perspectiveTransform(
                np.float32([[first, second]]), initial_h
            )[0].astype(np.float64)
            dx = abs(float(transformed[1, 0] - transformed[0, 0])) / max(self.field_width, 1e-9)
            dy = abs(float(transformed[1, 1] - transformed[0, 1])) / max(self.field_height, 1e-9)
            family = "horizontal" if dy <= dx else "vertical"
            segments.append((length, first, second, family))
        segments.sort(key=lambda item: item[0], reverse=True)
        return [(first, second, family) for _length, first, second, family in segments[:18]]

    def _refine_with_lines_goals_and_white_marks(
        self,
        corners: np.ndarray,
        frame: np.ndarray | None,
        mask: np.ndarray,
        observations: dict[str, _SideObservation],
        goal_detections: Iterable[dict[str, Any]] | None,
    ) -> tuple[np.ndarray, bool]:
        """Refine a provisional homography from rail lines and goal anchors.

        The solve is linear in a normalized image/field coordinate system.
        This is substantially more stable than optimizing raw homography
        coefficients and works even when two physical corners are off-screen.
        """
        if len(observations) < 2:
            return corners, False
        observed_lines = {side: observation.line for side, observation in observations.items()}
        goal_constraints = self._goal_anchor_constraints(goal_detections, observed_lines)
        if not goal_constraints:
            return corners, False

        image_normalizer = np.array(
            [
                [1.0 / max(self.frame_width, 1), 0.0, 0.0],
                [0.0, 1.0 / max(self.frame_height, 1), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        field_denormalizer = np.array(
            [
                [self.field_width, 0.0, 0.0],
                [0.0, self.field_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        initial_h = cv2.getPerspectiveTransform(
            corners.astype(np.float32), self.canonical_corners
        ).astype(np.float64)
        initial_g = (
            np.array(
                [
                    [1.0 / max(self.field_width, 1e-9), 0.0, 0.0],
                    [0.0, 1.0 / max(self.field_height, 1e-9), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            @ initial_h
            @ np.array(
                [
                    [self.frame_width, 0.0, 0.0],
                    [0.0, self.frame_height, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
        initial_g /= max(abs(float(initial_g[2, 2])), 1e-9)
        initial_parameters = np.array(
            [
                initial_g[0, 0], initial_g[0, 1], initial_g[0, 2],
                initial_g[1, 0], initial_g[1, 1], initial_g[1, 2],
                initial_g[2, 0], initial_g[2, 1],
            ],
            dtype=np.float64,
        )

        rows: list[np.ndarray] = []
        values: list[float] = []

        def append_point_equations(
            image_point: np.ndarray,
            target_u: float,
            target_v: float,
            weight: float,
        ) -> None:
            u = float(image_point[0]) / max(self.frame_width, 1)
            v = float(image_point[1]) / max(self.frame_height, 1)
            root_weight = float(np.sqrt(max(weight, 1e-6)))
            rows.append(
                root_weight
                * np.array([u, v, 1.0, 0.0, 0.0, 0.0, -target_u * u, -target_u * v])
            )
            values.append(root_weight * target_u)
            rows.append(
                root_weight
                * np.array([0.0, 0.0, 0.0, u, v, 1.0, -target_v * u, -target_v * v])
            )
            values.append(root_weight * target_v)

        side_targets = {
            "far": (0, 1.0),
            "near": (0, 0.0),
            "left": (1, 0.0),
            "right": (1, 1.0),
        }
        for side, observation in observations.items():
            axis, target = side_targets[side]
            points = self._sample_points_on_line(observation.line, samples=7)
            weight = 1.6 + 3.0 * observation.confidence
            for point in points:
                u = float(point[0]) / max(self.frame_width, 1)
                v = float(point[1]) / max(self.frame_height, 1)
                root_weight = float(np.sqrt(weight))
                if axis == 0:
                    rows.append(
                        root_weight
                        * np.array([u, v, 1.0, 0.0, 0.0, 0.0, -target * u, -target * v])
                    )
                else:
                    rows.append(
                        root_weight
                        * np.array([0.0, 0.0, 0.0, u, v, 1.0, -target * u, -target * v])
                    )
                values.append(root_weight * target)

        for image_point, field_target, confidence in goal_constraints:
            append_point_equations(
                image_point,
                float(field_target[0]) / max(self.field_width, 1e-9),
                float(field_target[1]) / max(self.field_height, 1e-9),
                weight=1.2 + 2.2 * float(np.clip(confidence, 0.0, 1.0)),
            )

        # Weak Tikhonov regularization keeps the unseen sides close to the best
        # mask/white-line bootstrap while the observed rails and goals dominate.
        regularization = 0.035
        for index in range(8):
            row = np.zeros(8, dtype=np.float64)
            row[index] = np.sqrt(regularization)
            rows.append(row)
            values.append(np.sqrt(regularization) * initial_parameters[index])

        matrix = np.vstack(rows)
        vector = np.asarray(values, dtype=np.float64)
        try:
            parameters, _residuals, rank, singular_values = np.linalg.lstsq(
                matrix, vector, rcond=None
            )
        except np.linalg.LinAlgError:
            return corners, False
        if rank < 8 or not np.isfinite(parameters).all():
            return corners, False
        if len(singular_values) and singular_values[-1] < 1e-8:
            return corners, False

        refined_g = np.array(
            [
                [parameters[0], parameters[1], parameters[2]],
                [parameters[3], parameters[4], parameters[5]],
                [parameters[6], parameters[7], 1.0],
            ],
            dtype=np.float64,
        )
        refined_h = field_denormalizer @ refined_g @ image_normalizer
        inverse = self._safe_inverse(refined_h)
        if inverse is None:
            return corners, False
        refined_corners = cv2.perspectiveTransform(
            self.canonical_corners.reshape(1, -1, 2), inverse
        ).reshape(-1, 2)
        refined_corners = np.asarray(refined_corners, dtype=np.float32).reshape(4, 2)
        if not self._candidate_is_sane(refined_corners):
            return corners, False

        # Validate that both goal anchors improved, not just one of them.
        def goal_error(homography: np.ndarray) -> float:
            errors: list[float] = []
            for image_point, target, _confidence in goal_constraints:
                transformed = cv2.perspectiveTransform(
                    np.float32([[image_point]]), homography
                )[0, 0]
                normalized_error = np.array(
                    [
                        (float(transformed[0]) - float(target[0])) / max(self.field_width, 1e-9),
                        (float(transformed[1]) - float(target[1])) / max(self.field_height, 1e-9),
                    ]
                )
                errors.append(float(np.linalg.norm(normalized_error)))
            return float(np.mean(errors)) if errors else 1e9

        initial_error = goal_error(initial_h)
        refined_error = goal_error(refined_h)
        if not np.isfinite(refined_error) or refined_error >= 0.72 * initial_error:
            return corners, False
        return refined_corners, True

    def _candidate_score(
        self,
        corners: np.ndarray,
        frame: np.ndarray | None,
        mask: np.ndarray,
        observations: dict[str, _SideObservation],
        goal_detections: Iterable[dict[str, Any]] | None,
        predicted_corners: np.ndarray | None,
    ) -> tuple[float, float, float, float]:
        if not self._candidate_is_sane(corners):
            return 0.0, 0.0, 0.0, 0.0
        homography = cv2.getPerspectiveTransform(
            corners.astype(np.float32), self.canonical_corners
        )
        if not np.isfinite(homography).all():
            return 0.0, 0.0, 0.0, 0.0

        recall = self._mask_recall(corners, mask)
        border_scores: list[float] = []
        lines = self._lines_from_corners(corners)
        diagonal = float(np.hypot(self.frame_width, self.frame_height))
        for side, observation in observations.items():
            line = lines.get(side)
            if line is None:
                continue
            distances = np.abs(observation.points @ line[:2] + line[2])
            border_scores.append(float(np.exp(-np.median(distances) / 7.0)))
        border_score = float(np.mean(border_scores)) if border_scores else 0.0
        white_score = self._white_alignment_score(frame, mask, homography)
        goal_score = self._goal_consistency_score(homography, goal_detections)

        temporal_score = 0.5
        if predicted_corners is not None:
            difference = float(
                np.mean(np.linalg.norm(corners - predicted_corners, axis=1))
            )
            temporal_score = float(np.exp(-difference / max(1.0, 0.14 * diagonal)))

        observed_fraction = len(observations) / 4.0
        score = (
            0.30 * border_score
            + 0.21 * white_score
            + 0.18 * recall
            + 0.18 * temporal_score
            + 0.13 * goal_score
        )
        score *= 0.68 + 0.32 * observed_fraction
        return float(score), border_score, white_score, goal_score

    def _build_candidate_lines(
        self,
        observations: dict[str, _SideObservation],
        predicted_corners: np.ndarray | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, float]]:
        predicted_lines = (
            self._lines_from_corners(predicted_corners)
            if predicted_corners is not None
            else {}
        )
        lines: dict[str, np.ndarray] = {}
        status: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for side in SIDE_NAMES:
            observation = observations.get(side)
            predicted = predicted_lines.get(side)
            if observation is not None and predicted is not None:
                weight = 0.24 + 0.62 * observation.confidence
                lines[side] = self._blend_lines(predicted, observation.line, weight)
                status[side] = "observado"
                confidence[side] = observation.confidence
            elif observation is not None:
                lines[side] = observation.line
                status[side] = "observado"
                confidence[side] = observation.confidence
            elif predicted is not None:
                lines[side] = predicted
                status[side] = "propagado"
                confidence[side] = max(0.12, self.last_result.side_confidence.get(side, 0.4) * 0.97)
            else:
                status[side] = "desconocido"
                confidence[side] = 0.0
        return lines, status, confidence

    def update(
        self,
        segmentation: FieldMaskResult | None,
        current_to_reference: np.ndarray | None,
        frame: np.ndarray | None = None,
        goal_detections: Iterable[dict[str, Any]] | None = None,
        exclusion_boxes: Iterable[list[float]] | None = None,
        frame_index: int | None = None,
    ) -> FieldGeometryResult:
        self.frames_since_measurement += 1
        activated = self._activate_manual_calibration(
            frame_index=frame_index,
            current_to_reference=current_to_reference,
        )
        if activated is not None:
            return activated
        predicted_corners = self._predicted_corners(current_to_reference)

        if (
            segmentation is None
            and self.reference_to_local is not None
            and current_to_reference is not None
            and self.last_result.geometry_state == "local"
        ):
            local_h = self.reference_to_local @ np.asarray(current_to_reference, dtype=np.float64)
            self.last_local_homography = local_h.copy()
            return self._local_result(
                source="orientacion_local_propagada",
                coverage=float(self.last_result.mask_coverage),
                local_homography=local_h,
                confidence=max(0.18, float(self.last_result.confidence) * 0.985),
                manual_line_score=float(self.last_result.manual_line_score),
                hard_anchor_score=float(self.last_result.hard_anchor_score),
                hard_anchor_count=int(self.last_result.hard_anchor_count),
                feature_match_score=float(self.last_result.feature_match_score),
                feature_match_count=int(self.last_result.feature_match_count),
                feature_matches=dict(self.last_result.feature_matches),
                side_lines=dict(self.last_result.side_lines),
                rejected_frame_sides=list(self.last_result.rejected_frame_sides),
                measured=False,
                propagated=True,
            )

        # A line-assisted calibration is the authoritative geometric seed.
        # Propagate it with camera registration and never replace it with a
        # discontinuous automatic guess.  This mode exists specifically for
        # crops where the visible evidence is mathematically insufficient to
        # recover all four field sides from one frame.
        if (
            self.manual_calibration_active
            and self.manual_calibration is not None
            and self.manual_calibration.is_complete
            and predicted_corners is not None
        ):
            homography = cv2.getPerspectiveTransform(
                np.asarray(predicted_corners, dtype=np.float32),
                self.canonical_corners,
            )
            confidence = max(
                0.55,
                float(self.last_result.confidence) * 0.997,
            )
            return self._trusted_result_from_corners(
                predicted_corners,
                homography,
                source="calibracion_asistida_propagada",
                measured=False,
                confidence=confidence,
                coverage=float(self.last_result.mask_coverage),
            )

        coverage = 0.0
        segmentation_confidence = 0.0
        observations: dict[str, _SideObservation] = {}
        support: dict[str, int] = {}
        side_visible = {side: False for side in SIDE_NAMES}
        rejected_frame_sides: list[str] = []
        cleaned: np.ndarray | None = None

        if segmentation is not None:
            cleaned = self._largest_clean_mask(segmentation.mask)
            self.last_surface_mask_image = (cleaned > 0).astype(np.uint8) * 255
            coverage = float(np.count_nonzero(cleaned)) / max(1.0, float(cleaned.size))
            segmentation_confidence = float(segmentation.confidence)
            if np.count_nonzero(cleaned) > 100:
                if frame is not None:
                    (
                        observations,
                        support,
                        side_visible,
                        rejected_frame_sides,
                        _support_map,
                    ) = self._observe_physical_sides(frame, cleaned)
                else:
                    (
                        observations,
                        support,
                        side_visible,
                        rejected_frame_sides,
                    ) = self._observe_mask_sides_without_image(cleaned)

        # V8: global coordinates are unlocked only by semantically identified
        # feature anchors. Rails and the surface mask remain excellent local
        # evidence, but they may not invent a canonical field location.
        goal_list = list(goal_detections or [])
        manual_segments = None
        if self.manual_calibration_active and self.manual_calibration is not None:
            manual_segments = self.manual_calibration.semantic_segments

        if cleaned is not None and frame is not None:
            template_result = self.template_registrar.register(
                frame=frame,
                field_mask=cleaned,
                goal_detections=goal_list,
                exclusion_boxes=exclusion_boxes,
                predicted_corners=predicted_corners,
                semantic_segments=manual_segments,
            )
            if (
                template_result.trusted
                and template_result.valid
                and template_result.corners_image is not None
                and template_result.homography_image_to_field_normalized is not None
            ):
                # Registrar order is near-left, far-left, far-right, near-right.
                corners = np.asarray(template_result.corners_image, dtype=np.float32)[[1, 2, 3, 0]]
                field_scale = np.array(
                    [[self.field_width, 0.0, 0.0], [0.0, self.field_height, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                homography = field_scale @ template_result.homography_image_to_field_normalized
                inverse = self._safe_inverse(homography)
                if inverse is not None:
                    self.frames_since_measurement = 0
                    if current_to_reference is not None:
                        inverse_registration = self._safe_inverse(current_to_reference)
                        if inverse_registration is not None:
                            self.reference_to_field = homography @ inverse_registration
                    lines = self._lines_from_corners(corners)
                    self.last_homography = homography
                    self.last_corners = corners.copy()
                    self.last_result = FieldGeometryResult(
                        valid=True,
                        trusted=True,
                        measured=True,
                        propagated=False,
                        confidence=float(template_result.confidence),
                        corners_image=corners,
                        homography_image_to_field=homography,
                        homography_field_to_image=inverse,
                        mask_coverage=coverage,
                        source=template_result.source,
                        line_support={
                            "left": int(round(100 * template_result.rail_score)),
                            "right": int(round(100 * template_result.rail_score)),
                        },
                        side_visible={"far": False, "right": True, "near": False, "left": True},
                        side_status={side: "anclado_semantico" for side in SIDE_NAMES},
                        side_confidence={side: float(template_result.feature_match_score) for side in SIDE_NAMES},
                        side_lines=lines,
                        rejected_frame_sides=rejected_frame_sides,
                        border_evidence_score=float(template_result.rail_score),
                        white_alignment_score=float(template_result.template_score),
                        goal_consistency_score=float(template_result.goal_score),
                        visible_template_fraction=float(template_result.visible_template_fraction),
                        manual_line_score=float(template_result.manual_line_score),
                        registration_scope=str(template_result.registration_scope),
                        geometry_state="global",
                        hard_anchor_score=float(template_result.hard_anchor_score),
                        hard_anchor_count=int(template_result.hard_anchor_count),
                        feature_match_score=float(template_result.feature_match_score),
                        feature_match_count=int(template_result.feature_match_count),
                        feature_matches=dict(template_result.feature_matches or {}),
                        field_width=self.field_width,
                        field_height=self.field_height,
                    )
                    return self.last_result

            # A previously trusted global solution may be transported through a
            # short interval, but a new local hypothesis can never overwrite it.
            if self.last_result.trusted and predicted_corners is not None:
                homography = cv2.getPerspectiveTransform(
                    np.asarray(predicted_corners, dtype=np.float32), self.canonical_corners
                )
                confidence = max(0.30, float(self.last_result.confidence) * 0.972)
                propagated = self._trusted_result_from_corners(
                    predicted_corners,
                    homography,
                    source="global_propagada_sin_anclas_nuevas",
                    measured=False,
                    confidence=confidence,
                    coverage=coverage,
                )
                propagated.feature_matches = dict(template_result.feature_matches or {})
                propagated.feature_match_score = float(template_result.feature_match_score)
                propagated.feature_match_count = int(template_result.feature_match_count)
                propagated.hard_anchor_score = float(template_result.hard_anchor_score)
                propagated.hard_anchor_count = int(template_result.hard_anchor_count)
                return propagated

            rail_lines = {}
            for index, line in enumerate(template_result.rail_lines or []):
                rail_lines[f"rail_{index}"] = self._normalize_line(line)
            if template_result.local_homography_image_to_local is not None:
                self.last_local_homography = np.asarray(
                    template_result.local_homography_image_to_local, dtype=np.float64
                ).copy()
                if current_to_reference is not None:
                    inverse_registration = self._safe_inverse(current_to_reference)
                    if inverse_registration is not None:
                        self.reference_to_local = self.last_local_homography @ inverse_registration
            return self._local_result(
                source=template_result.source,
                coverage=coverage,
                local_homography=template_result.local_homography_image_to_local,
                confidence=max(0.18, float(template_result.confidence)),
                manual_line_score=float(template_result.manual_line_score),
                hard_anchor_score=float(template_result.hard_anchor_score),
                hard_anchor_count=int(template_result.hard_anchor_count),
                feature_match_score=float(template_result.feature_match_score),
                feature_match_count=int(template_result.feature_match_count),
                feature_matches=dict(template_result.feature_matches or {}),
                side_lines=rail_lines,
                rejected_frame_sides=rejected_frame_sides,
            )

        candidate_lines, side_status, side_confidence = self._build_candidate_lines(
            observations, predicted_corners
        )
        candidates: list[tuple[np.ndarray, str]] = []
        direct_candidate = self._corners_from_lines(candidate_lines)
        if direct_candidate is not None:
            if not observations and predicted_corners is not None:
                source = "propagada_registro"
            else:
                source = (
                    "rectas_fisicas_temporales"
                    if predicted_corners is not None
                    else "rectas_fisicas"
                )
            candidates.append((direct_candidate, source))

        if (
            cleaned is not None
            and frame is not None
            and predicted_corners is None
            and len(observations) >= 2
        ):
            raw_quad = self._fit_raw_quad(cleaned)
            if raw_quad is not None:
                for candidate in self._extrapolated_bootstrap_candidates(
                    raw_quad,
                    observations,
                    rejected_frame_sides,
                ):
                    candidates.append((candidate, "bootstrap_extrapolado"))

        # When no new segmentation arrived, pure propagation is valid.
        if not candidates and predicted_corners is not None:
            candidates.append((predicted_corners, "propagada_registro"))
        elif not candidates and self.last_corners is not None and self.frames_since_measurement <= 12:
            candidates.append((self.last_corners.copy(), "retenida_temporal"))

        if not candidates or cleaned is None and predicted_corners is None:
            self.last_result = self._empty_result("sin_geometria", coverage)
            return self.last_result

        if cleaned is None:
            # Scoring a propagated candidate does not need a fresh mask.
            cleaned_for_score = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            if self.last_corners is not None:
                cv2.fillConvexPoly(
                    cleaned_for_score,
                    np.round(np.clip(self.last_corners, [-3*self.frame_width, -3*self.frame_height], [4*self.frame_width, 4*self.frame_height])).astype(np.int32),
                    255,
                )
        else:
            cleaned_for_score = cleaned

        scored: list[tuple[float, np.ndarray, str, float, float, float]] = []
        for candidate, source in candidates:
            score, border_score, white_score, goal_score = self._candidate_score(
                candidate,
                frame,
                cleaned_for_score,
                observations,
                goal_detections,
                predicted_corners,
            )
            if source in {"propagada_registro", "retenida_temporal"} and not observations:
                # Propagation confidence comes from the previous accepted state,
                # not from a synthetic mask-recall score.
                score = max(score, float(self.last_result.confidence) * 0.965)
            scored.append((score, candidate, source, border_score, white_score, goal_score))

        score, corners, source, border_score, white_score, goal_score = max(
            scored, key=lambda item: item[0]
        )
        if cleaned is not None and frame is not None and observations:
            refined_corners, refined = self._refine_with_lines_goals_and_white_marks(
                corners,
                frame,
                cleaned,
                observations,
                goal_detections,
            )
            if refined:
                refined_score, refined_border, refined_white, refined_goal = self._candidate_score(
                    refined_corners,
                    frame,
                    cleaned,
                    observations,
                    goal_detections,
                    predicted_corners,
                )
                initial_recall = self._mask_recall(corners, cleaned)
                refined_recall = self._mask_recall(refined_corners, cleaned)
                goal_improved = refined_goal >= max(0.48, goal_score + 0.04)
                enough_geometry = len(observations) >= 3
                if (
                    refined_score >= 0.95 * score
                    and refined_recall >= 0.88 * initial_recall
                    and (enough_geometry or goal_improved)
                ):
                    corners = refined_corners
                    score = max(score, refined_score)
                    border_score = refined_border
                    white_score = refined_white
                    goal_score = refined_goal
                    source = "ajustada_rectas_porterias"
        if score <= 0.0 or not self._candidate_is_sane(corners):
            self.last_result = self._empty_result("candidato_rechazado", coverage)
            return self.last_result

        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        homography = cv2.getPerspectiveTransform(corners.astype(np.float32), self.canonical_corners)
        inverse = self._safe_inverse(homography)
        if inverse is None or not np.isfinite(homography).all():
            self.last_result = self._empty_result("homografia_invalida", coverage)
            return self.last_result

        observed_count = len(observations)
        measured = segmentation is not None and observed_count >= 2 and source != "propagada_registro"
        # A bootstrap with only two real sides is deliberately low confidence.
        measurement_confidence = float(
            np.clip(
                0.18 * segmentation_confidence
                + 0.32 * border_score
                + 0.20 * white_score
                + 0.12 * goal_score
                + 0.18 * min(1.0, observed_count / 3.0),
                0.0,
                1.0,
            )
        )
        if source == "bootstrap_extrapolado":
            measurement_confidence = min(measurement_confidence, 0.62)
        if source in {"propagada_registro", "retenida_temporal"}:
            measured = False
            measurement_confidence = max(
                0.10,
                float(self.last_result.confidence) * (0.968 if source == "propagada_registro" else 0.94),
            )

        # Do not replace a strong state with a weak, discontinuous bootstrap.
        if self.last_corners is not None and predicted_corners is not None and measured:
            diagonal = float(np.hypot(self.frame_width, self.frame_height))
            discrepancy = float(np.mean(np.linalg.norm(corners - predicted_corners, axis=1)))
            if discrepancy > 0.24 * diagonal and measurement_confidence < 0.68:
                corners = predicted_corners
                homography = cv2.getPerspectiveTransform(corners.astype(np.float32), self.canonical_corners)
                inverse = self._safe_inverse(homography)
                source = "propagada_rechazo_salto"
                measured = False
                measurement_confidence = max(0.10, float(self.last_result.confidence) * 0.96)
                candidate_lines = self._lines_from_corners(corners)
                side_status = {side: "propagado" for side in SIDE_NAMES}

        if inverse is None:
            self.last_result = self._empty_result("homografia_invalida", coverage)
            return self.last_result

        if measured:
            self.frames_since_measurement = 0
            if current_to_reference is not None:
                inverse_registration = self._safe_inverse(current_to_reference)
                if inverse_registration is not None:
                    self.reference_to_field = homography @ inverse_registration
        elif self.reference_to_field is None and current_to_reference is not None:
            inverse_registration = self._safe_inverse(current_to_reference)
            if inverse_registration is not None:
                self.reference_to_field = homography @ inverse_registration

        final_lines = self._lines_from_corners(corners)
        for side, observation in observations.items():
            # Keep the actually observed line in diagnostics when it was used.
            if side_status.get(side) == "observado":
                final_lines[side] = observation.line

        self.last_homography = homography
        self.last_corners = corners.copy()
        if source in {
            "propagada_registro",
            "propagada_rechazo_salto",
            "retenida_temporal",
        }:
            trusted = bool(
                self.last_result.trusted
                and measurement_confidence >= 0.30
                and source != "retenida_temporal"
            )
        else:
            # Legacy mask-extrema/rail bootstraps remain useful diagnostics, but
            # they are no longer allowed to unlock events or Mesa Replay.  Only
            # the multiline template solver above, an assisted calibration, or
            # propagation of an already trusted state can do that.
            trusted = False
        self.last_result = FieldGeometryResult(
            valid=True,
            trusted=trusted,
            measured=measured,
            propagated=not measured,
            confidence=measurement_confidence,
            corners_image=corners,
            homography_image_to_field=homography,
            homography_field_to_image=inverse,
            mask_coverage=coverage,
            source=source,
            line_support=support,
            side_visible=side_visible,
            side_status=side_status,
            side_confidence=side_confidence,
            side_lines=final_lines,
            rejected_frame_sides=rejected_frame_sides,
            border_evidence_score=border_score,
            white_alignment_score=white_score,
            goal_consistency_score=goal_score,
            field_width=self.field_width,
            field_height=self.field_height,
        )
        return self.last_result

    def transform_point(self, x: float, y: float) -> tuple[float, float] | None:
        result = self.last_result
        if not result.valid or result.homography_image_to_field is None:
            return None
        point = np.float32([[[float(x), float(y)]]])
        transformed = cv2.perspectiveTransform(point, result.homography_image_to_field)[0, 0]
        if not np.isfinite(transformed).all():
            return None
        return float(transformed[0]), float(transformed[1])

    def _point_has_surface_support(
        self,
        x: float,
        y: float,
        group: str,
    ) -> bool:
        mask = self.last_surface_mask_image
        if mask is None:
            return bool(
                self.manual_calibration_active
                and self.manual_calibration is not None
                and self.manual_calibration.is_complete
            )
        height, width = mask.shape[:2]
        px = int(round(float(x)))
        py = int(round(float(y)))
        if px < 0 or py < 0 or px >= width or py >= height:
            return False
        # Bottom centers and ball centers can sit a few pixels outside the raw
        # segmentation because of rail thickness, shadows or mask stride.
        radius = 10 if group == "ball" else 18
        if group == "goal":
            radius = 34
        x1, x2 = max(0, px - radius), min(width, px + radius + 1)
        y1, y2 = max(0, py - radius), min(height, py + radius + 1)
        return bool(np.any(mask[y1:y2, x1:x2] > 0))

    def annotate_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        result = detection.copy()
        box = list(map(float, result.get("bbox_xyxy", [0, 0, 0, 0])))
        x1, y1, x2, y2 = box
        group = str(result.get("class_group", "")).lower()
        if group == "robot":
            anchor = ((x1 + x2) * 0.5, y2)
            anchor_type = "bottom_center"
        else:
            anchor = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
            anchor_type = "center"

        transformed = self.transform_point(*anchor)
        geometry = self.last_result
        supported = self._point_has_surface_support(anchor[0], anchor[1], group)
        result["field_anchor_type"] = anchor_type
        result["field_transform_supported"] = bool(supported)
        result["field_transform_valid"] = bool(
            transformed is not None and geometry.valid and geometry.trusted and supported
        )
        result["field_transform_extrapolated"] = bool(
            transformed is not None and geometry.valid and geometry.trusted and not supported
        )
        result["field_transform_provisional"] = bool(
            transformed is not None and geometry.valid and not geometry.trusted
        )
        result["field_transform_confidence"] = round(float(geometry.confidence), 6)
        result["field_transform_source"] = geometry.source
        if transformed is not None:
            field_x, field_y = transformed
            x_norm = field_x / max(self.field_width, 1e-9)
            y_norm = field_y / max(self.field_height, 1e-9)
            result.update(
                {
                    "field_x": round(field_x, 6),
                    "field_y": round(field_y, 6),
                    "field_x_norm": round(x_norm, 8),
                    "field_y_norm": round(y_norm, 8),
                    "inside_surface": bool(0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0),
                }
            )

            if group == "goal" and geometry.homography_image_to_field is not None:
                box_points = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
                transformed_box = cv2.perspectiveTransform(
                    box_points, geometry.homography_image_to_field
                )[0]
                result["field_polygon"] = [
                    [round(float(px), 5), round(float(py), 5)]
                    for px, py in transformed_box
                ]
        return result


def _line_segment_in_frame(
    line: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b, c = map(float, line)
    points: list[tuple[int, int]] = []
    if abs(b) > 1e-8:
        for x in (0.0, float(width - 1)):
            y = -(a * x + c) / b
            if -1.0 <= y <= height:
                points.append((int(round(x)), int(round(y))))
    if abs(a) > 1e-8:
        for y in (0.0, float(height - 1)):
            x = -(b * y + c) / a
            if -1.0 <= x <= width:
                points.append((int(round(x)), int(round(y))))
    unique: list[tuple[int, int]] = []
    for point in points:
        if point not in unique:
            unique.append(point)
    if len(unique) < 2:
        return None
    # Pick the farthest pair.
    best = None
    best_distance = -1.0
    for first in unique:
        for second in unique:
            distance = float(np.hypot(first[0] - second[0], first[1] - second[1]))
            if distance > best_distance:
                best_distance = distance
                best = (first, second)
    return best


def _draw_dashed_line(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 14,
) -> None:
    vector = np.array(second, dtype=np.float64) - np.array(first, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return
    direction = vector / length
    position = 0.0
    draw = True
    while position < length:
        next_position = min(length, position + dash_length)
        if draw:
            start = np.array(first) + direction * position
            end = np.array(first) + direction * next_position
            cv2.line(
                image,
                tuple(np.round(start).astype(int)),
                tuple(np.round(end).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
        draw = not draw
        position = next_position


def draw_field_geometry_overlay(
    frame: np.ndarray,
    segmentation: FieldMaskResult | None,
    geometry: FieldGeometryResult,
) -> np.ndarray:
    output = frame.copy()
    if segmentation is not None:
        mask = segmentation.mask > 0
        tint = np.zeros_like(output)
        tint[:, :] = (255, 190, 40)
        output[mask] = cv2.addWeighted(output[mask], 0.72, tint[mask], 0.28, 0)
        if len(segmentation.polygon) >= 3:
            cv2.polylines(
                output,
                [np.round(segmentation.polygon).astype(np.int32)],
                True,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )

    # Explicitly mark camera margins that were rejected as physical sides.
    margin_x = max(8, int(round(0.032 * output.shape[1])))
    margin_y = max(8, int(round(0.032 * output.shape[0])))
    red = (40, 40, 255)
    for side in geometry.rejected_frame_sides:
        if side == "far":
            cv2.line(output, (0, margin_y), (output.shape[1] - 1, margin_y), red, 2)
        elif side == "near":
            y = output.shape[0] - 1 - margin_y
            cv2.line(output, (0, y), (output.shape[1] - 1, y), red, 2)
        elif side == "left":
            cv2.line(output, (margin_x, 0), (margin_x, output.shape[0] - 1), red, 2)
        elif side == "right":
            x = output.shape[1] - 1 - margin_x
            cv2.line(output, (x, 0), (x, output.shape[0] - 1), red, 2)

    for side in SIDE_NAMES:
        line = geometry.side_lines.get(side)
        if line is None:
            continue
        segment = _line_segment_in_frame(line, output.shape[1], output.shape[0])
        if segment is None:
            continue
        status = geometry.side_status.get(side, "desconocido")
        if status == "observado":
            cv2.line(output, segment[0], segment[1], (60, 255, 60), 3, cv2.LINE_AA)
        elif status == "propagado":
            _draw_dashed_line(output, segment[0], segment[1], (0, 220, 255), 2)
        else:
            _draw_dashed_line(output, segment[0], segment[1], (170, 170, 170), 1)

    if geometry.valid and geometry.corners_image is not None:
        corners = np.round(geometry.corners_image).astype(np.int32)
        labels = ["LEJ-IZQ", "LEJ-DER", "CER-DER", "CER-IZQ"]
        for index, (x, y) in enumerate(corners):
            if -100 <= x <= output.shape[1] + 100 and -100 <= y <= output.shape[0] + 100:
                cv2.circle(output, (x, y), 6, (255, 120, 20), -1, cv2.LINE_AA)
                cv2.putText(
                    output,
                    labels[index],
                    (
                        max(3, min(output.shape[1] - 105, x + 8)),
                        max(20, min(output.shape[0] - 5, y - 8)),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 120, 20),
                    2,
                    cv2.LINE_AA,
                )

    if geometry.geometry_state == "global" and geometry.trusted:
        status = "GLOBAL ANCLADA" if geometry.measured else "GLOBAL / PROPAGADA"
        status_color = (60, 255, 60)
    elif geometry.geometry_state == "local":
        status = "ORIENTACION LOCAL - POSICION GLOBAL DESCONOCIDA"
        status_color = (0, 210, 255)
    elif geometry.geometry_state == "surface":
        status = "SOLO SUPERFICIE - SIN COORDENADAS"
        status_color = (0, 160, 255)
    elif geometry.valid:
        status = "PROVISIONAL BLOQUEADA"
        status_color = (0, 190, 255)
    else:
        status = "NO VALIDA"
        status_color = (0, 0, 255)
    cv2.putText(
        output,
        f"Cancha: {status} | alcance {geometry.registration_scope} | conf {geometry.confidence:.2f} | {geometry.source}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        status_color,
        2,
        cv2.LINE_AA,
    )
    observed = sum(value == "observado" for value in geometry.side_status.values())
    propagated = sum(value == "propagado" for value in geometry.side_status.values())
    cv2.putText(
        output,
        (
            f"Lados fisicos {observed}/4 | propagados {propagated} | "
            f"borde {geometry.border_evidence_score:.2f} | blancas {geometry.white_alignment_score:.2f} | "
            f"anclas {geometry.hard_anchor_count}:{geometry.hard_anchor_score:.2f} | rasgos {geometry.feature_match_count}:{geometry.feature_match_score:.2f}"
        ),
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Verde=riel observado  Amarillo=propagado  Rojo=margen de camara rechazado",
        (18, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return output


def _draw_canonical_template(
    image: np.ndarray,
    color: tuple[int, int, int] = (238, 238, 238),
    thickness: int = 2,
) -> None:
    height, width = image.shape[:2]
    template = build_template_points(density=260)
    points = template.points.copy()
    points[:, 0] *= max(1, width - 1)
    points[:, 1] *= max(1, height - 1)
    for index in range(1, len(points)):
        if template.groups[index] != template.groups[index - 1]:
            continue
        first = points[index - 1]
        second = points[index]
        if float(np.linalg.norm(second - first)) > 0.18 * max(width, height):
            continue
        cv2.line(
            image,
            tuple(np.rint(first).astype(int)),
            tuple(np.rint(second).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _render_local_rectification(
    frame: np.ndarray,
    geometry: FieldGeometryResult,
    segmentation: FieldMaskResult | None,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Render honest local orientation without a canonical field position."""
    canvas = np.full((output_height, output_width, 3), (28, 28, 28), dtype=np.uint8)
    raw_mask = None
    if segmentation is not None:
        raw_mask = (segmentation.mask > 0).astype(np.uint8) * 255
        if raw_mask.shape != frame.shape[:2]:
            raw_mask = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    local_h = geometry.local_homography_image_to_local
    if local_h is not None:
        if raw_mask is not None and np.any(raw_mask):
            ys, xs = np.nonzero(raw_mask)
            step = max(1, len(xs) // 5000)
            points = np.column_stack([xs[::step], ys[::step]]).astype(np.float32)
        else:
            h, w = frame.shape[:2]
            points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        transformed = cv2.perspectiveTransform(points.reshape(1, -1, 2), local_h).reshape(-1, 2)
        finite = transformed[np.isfinite(transformed).all(axis=1)]
        if len(finite) >= 4:
            minimum = np.percentile(finite, 1.0, axis=0)
            maximum = np.percentile(finite, 99.0, axis=0)
            span = np.maximum(maximum - minimum, 1.0)
            scale = min((output_width - 36) / span[0], (output_height - 82) / span[1])
            fit = np.array([
                [scale, 0.0, 18.0 - scale * minimum[0]],
                [0.0, scale, 64.0 - scale * minimum[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            warp_h = fit @ local_h
            warped = cv2.warpPerspective(frame, warp_h, (output_width, output_height))
            if raw_mask is not None:
                warped_mask = cv2.warpPerspective(raw_mask, warp_h, (output_width, output_height), flags=cv2.INTER_NEAREST)
                support = warped_mask > 0
                canvas[support] = warped[support]
            else:
                support = np.any(warped > 3, axis=2)
                canvas[support] = warped[support]
            # Local coordinate axes only. They do not claim Mesa Replay position.
            cv2.line(canvas, (50, output_height - 42), (170, output_height - 42), (70, 230, 255), 3, cv2.LINE_AA)
            cv2.line(canvas, (50, output_height - 42), (50, output_height - 142), (255, 150, 60), 3, cv2.LINE_AA)
            cv2.putText(canvas, "direccion campo", (178, output_height - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "ancho local", (57, output_height - 148), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 150, 60), 1, cv2.LINE_AA)
    elif raw_mask is not None and np.any(raw_mask):
        visible = cv2.findNonZero(raw_mask)
        x, y, w, h = cv2.boundingRect(visible)
        crop = frame[y:y+h, x:x+w]
        crop_mask = raw_mask[y:y+h, x:x+w]
        scale = min((output_width - 36) / max(1, w), (output_height - 82) / max(1, h))
        resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
        resized_mask = cv2.resize(crop_mask, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
        ox = (output_width - resized.shape[1]) // 2
        oy = 62 + max(0, (output_height - 62 - resized.shape[0]) // 2)
        region = canvas[oy:oy+resized.shape[0], ox:ox+resized.shape[1]]
        region[resized_mask > 0] = resized[resized_mask > 0]

    cv2.rectangle(canvas, (0, 0), (output_width, 52), (0, 0, 0), cv2.FILLED)
    state_text = "ORIENTACION LOCAL" if local_h is not None else "SOLO SUPERFICIE"
    cv2.putText(canvas, f"V8 {state_text} - posicion global desconocida", (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"anclas {geometry.hard_anchor_count} | coincidencias {geometry.feature_match_count} | {geometry.source}", (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def render_rectified_debug(
    frame: np.ndarray,
    geometry: FieldGeometryResult,
    detections: list[dict[str, Any]],
    segmentation: FieldMaskResult | None = None,
    output_width: int = 1000,
    output_height: int = 600,
    goal_depth_ratio: float = 0.10,
) -> np.ndarray:
    """Render V8 progressive geometry without inventing a global field."""

    if geometry.geometry_state != "global" or not geometry.trusted:
        return _render_local_rectification(
            frame, geometry, segmentation, output_width + 2 * max(45, int(round(output_width * goal_depth_ratio))), output_height
        )

    goal_margin = max(45, int(round(output_width * goal_depth_ratio)))
    total_width = output_width + 2 * goal_margin
    field_canvas = np.full((output_height, output_width, 3), (55, 132, 88), dtype=np.uint8)
    _draw_canonical_template(field_canvas)

    support_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    if geometry.valid and geometry.homography_image_to_field is not None:
        scale = np.array(
            [
                [output_width / max(geometry.field_width, 1e-9), 0.0, 0.0],
                [0.0, output_height / max(geometry.field_height, 1e-9), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        warp_h = scale @ geometry.homography_image_to_field
        warped = cv2.warpPerspective(
            frame,
            warp_h,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if segmentation is not None:
            raw_mask = (segmentation.mask > 0).astype(np.uint8) * 255
            if raw_mask.shape != frame.shape[:2]:
                raw_mask = cv2.resize(
                    raw_mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            support_mask = cv2.warpPerspective(
                raw_mask,
                warp_h,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            support_mask = (np.any(warped > 3, axis=2)).astype(np.uint8) * 255
        support_mask = cv2.morphologyEx(
            support_mask,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8),
        )
        support = support_mask > 0
        if np.any(support):
            # Preserve the real video only where the segmenter says surface is
            # visible. The rest remains the schematic canonical field.
            field_canvas[support] = cv2.addWeighted(
                warped[support], 0.88, field_canvas[support], 0.12, 0.0
            )
    else:
        cv2.putText(
            field_canvas,
            "Sin homografia: solo caracteristicas parciales",
            (55, output_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw only detections whose anchors are backed by visible segmented
    # surface. Extrapolated coordinates remain in JSON diagnostics but are not
    # presented as reliable replay positions.
    for detection in detections:
        if not detection.get("field_transform_valid"):
            continue
        x_norm = detection.get("field_x_norm")
        y_norm = detection.get("field_y_norm")
        if x_norm is None or y_norm is None:
            continue
        x = int(round(float(x_norm) * output_width))
        y = int(round(float(y_norm) * output_height))
        group = str(detection.get("class_group", ""))
        if group == "robot":
            color, radius = (255, 170, 0), 11
        elif group == "ball":
            color, radius = (0, 120, 255), 8
        elif group == "goal":
            color, radius = (0, 240, 255), 7
        else:
            continue
        if 0 <= x < output_width and 0 <= y < output_height:
            cv2.circle(field_canvas, (x, y), radius, color, -1, cv2.LINE_AA)

    support_fraction = float(np.count_nonzero(support_mask)) / max(1.0, float(support_mask.size))
    partial = bool(geometry.registration_scope == "partial" or support_fraction < 0.72)
    visible = cv2.findNonZero((support_mask > 0).astype(np.uint8))

    if partial and visible is not None:
        x, y, w, h = cv2.boundingRect(visible)
        pad_x = max(18, int(round(0.10 * w)))
        pad_y = max(18, int(round(0.10 * h)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(output_width, x + w + pad_x)
        y2 = min(output_height, y + h + pad_y)
        crop = field_canvas[y1:y2, x1:x2]
        if crop.size:
            canvas = cv2.resize(crop, (total_width, output_height), interpolation=cv2.INTER_LINEAR)
            # Full-field locator map: schematic only, with the visible viewport.
            mini_w, mini_h = 240, 140
            mini = np.full((mini_h, mini_w, 3), (55, 132, 88), dtype=np.uint8)
            _draw_canonical_template(mini, thickness=1)
            cv2.rectangle(
                mini,
                (int(round(x1 / output_width * mini_w)), int(round(y1 / output_height * mini_h))),
                (int(round(x2 / output_width * mini_w)), int(round(y2 / output_height * mini_h))),
                (0, 240, 255),
                2,
            )
            mx = total_width - mini_w - 16
            my = 16
            canvas[my : my + mini_h, mx : mx + mini_w] = mini
            cv2.rectangle(canvas, (mx, my), (mx + mini_w, my + mini_h), (245, 245, 245), 1)
        else:
            canvas = cv2.resize(field_canvas, (total_width, output_height))
    else:
        canvas = np.full((output_height, total_width, 3), (25, 25, 25), dtype=np.uint8)
        canvas[:, goal_margin : goal_margin + output_width] = field_canvas
        goal_height = int(output_height * 0.34)
        goal_y1 = (output_height - goal_height) // 2
        goal_y2 = goal_y1 + goal_height
        cv2.rectangle(canvas, (2, goal_y1), (goal_margin, goal_y2), (0, 220, 255), 3)
        cv2.rectangle(
            canvas,
            (goal_margin + output_width, goal_y1),
            (total_width - 3, goal_y2),
            (255, 0, 210),
            3,
        )

    status = "PARCIAL" if partial else "COMPLETA"
    trust = "CONFIABLE" if geometry.trusted else "PROVISIONAL"
    cv2.rectangle(canvas, (0, 0), (total_width, 48), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        canvas,
        (
            f"V8 GLOBAL {status} / {trust} | visible {support_fraction * 100:.1f}% | "
            f"{geometry.source} | conf {geometry.confidence:.2f}"
        ),
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (0, 255, 255) if geometry.valid else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas

