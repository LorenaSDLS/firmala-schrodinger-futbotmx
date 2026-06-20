from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
import heapq
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - OpenCV fallback remains available.
    cKDTree = None

from src.I_field_geometry.feature_constraints import (
    CANONICAL_LINES_NORMALIZED,
    CANONICAL_SEGMENTS_NORMALIZED,
    TRANSVERSE_FEATURES,
    score_manual_anchors,
    local_rectification_from_segments,
    semantic_family_counts,
)
from src.I_field_geometry.visual_evidence import (
    AdaptiveFieldEvidenceExtractor,
    VisualLineCandidate,
)
from src.I_field_geometry.field_template import (
    FieldTemplateConfig,
    TemplatePointSet,
    build_template_points,
)


@dataclass(frozen=True)
class RailLineCandidate:
    line: np.ndarray
    segment: np.ndarray
    support: float
    length_ratio: float


@dataclass
class TemplateRegistrationResult:
    valid: bool
    trusted: bool
    confidence: float
    corners_image: np.ndarray | None
    homography_image_to_field_normalized: np.ndarray | None
    homography_field_to_image_normalized: np.ndarray | None
    source: str
    template_score: float = 0.0
    mask_score: float = 0.0
    goal_score: float = 0.0
    rail_score: float = 0.0
    visible_template_fraction: float = 0.0
    rail_lines: list[np.ndarray] | None = None
    goal_anchors: list[np.ndarray] | None = None
    manual_line_score: float = 0.0
    registration_scope: str = "none"
    geometry_state: str = "surface"
    local_homography_image_to_local: np.ndarray | None = None
    hard_anchor_score: float = 0.0
    hard_anchor_count: int = 0
    feature_match_score: float = 0.0
    feature_match_count: int = 0
    feature_matches: dict[str, float] | None = None
    reverse_template_score: float = 0.0
    boundary_alignment_score: float = 0.0
    candidate_margin: float = 0.0
    temporal_evidence_frames: int = 0
    marking_pixel_fraction: float = 0.0
    candidate_count: int = 0
    physical_boundary_score: float = 0.0
    physical_boundary_count: int = 0
    physical_boundary_scores: dict[str, float] | None = None


def normalize_line(line: np.ndarray) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64).reshape(3)
    norm = float(np.hypot(line[0], line[1]))
    return line / max(norm, 1e-12)


def line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    line = np.cross(
        np.array([float(first[0]), float(first[1]), 1.0]),
        np.array([float(second[0]), float(second[1]), 1.0]),
    )
    if float(np.hypot(line[0], line[1])) < 1e-8:
        return None
    return normalize_line(line)


def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))
    if abs(float(point[2])) < 1e-10:
        return None
    point = point[:2] / point[2]
    return point.astype(np.float64) if np.isfinite(point).all() else None


def _signed_distance(line: np.ndarray, point: np.ndarray) -> float:
    line = normalize_line(line)
    return float(line[0] * point[0] + line[1] * point[1] + line[2])


def _point_inside_pair(
    first: np.ndarray,
    second: np.ndarray,
    point: np.ndarray,
    first_reference: np.ndarray | None = None,
    second_reference: np.ndarray | None = None,
) -> bool:
    # The lines are unoriented and may converge. A point is in their projective
    # strip when it lies on the same side of line 1 as line 2's midpoint, and
    # on the same side of line 2 as line 1's midpoint.
    if first_reference is None or second_reference is None:
        d1 = _signed_distance(first, point)
        d2 = _signed_distance(second, point)
        return d1 * d2 <= 0.0 or min(abs(d1), abs(d2)) < 12.0
    return (
        _signed_distance(first, point) * _signed_distance(first, second_reference) >= -1e-6
        and _signed_distance(second, point) * _signed_distance(second, first_reference) >= -1e-6
    )


def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)


def _safe_inverse(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    return inverse if np.isfinite(inverse).all() else None


def _quad_is_sane(corners: np.ndarray, width: int, height: int) -> bool:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        return False
    area = abs(float(cv2.contourArea(corners)))
    frame_area = float(width * height)
    if not (0.015 * frame_area <= area <= 60.0 * frame_area):
        return False
    # A field corner may legitimately lie outside the crop, but solutions many
    # frame-heights away are the classic projective degeneracy that previously
    # collapsed Mesa Replay.  Keep enough room for handheld crops without
    # accepting effectively infinite quadrilaterals.
    if np.any(corners[:, 0] < -6.0 * width) or np.any(corners[:, 0] > 7.0 * width):
        return False
    if np.any(corners[:, 1] < -6.0 * height) or np.any(corners[:, 1] > 7.0 * height):
        return False
    near_width = float(np.linalg.norm(corners[3] - corners[0]))
    far_width = float(np.linalg.norm(corners[2] - corners[1]))
    if min(near_width, far_width) < 0.08 * np.hypot(width, height):
        return False
    if max(near_width, far_width) / max(1.0, min(near_width, far_width)) > 15.0:
        return False
    return True


def _goal_boxes(goal_detections: Iterable[dict[str, Any]] | None) -> list[tuple[np.ndarray, np.ndarray, float]]:
    goals: list[tuple[np.ndarray, np.ndarray, float]] = []
    for detection in goal_detections or []:
        if str(detection.get("class_group", "")).lower() != "goal":
            continue
        box = detection.get("bbox_xyxy", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        goals.append((center, np.array([x1, y1, x2, y2], dtype=np.float64), float(detection.get("confidence", 0.5))))
    if len(goals) <= 2:
        return goals
    # Keep a confident, spatially separated pair rather than two duplicate boxes.
    best_pair: tuple[float, tuple, tuple] | None = None
    for first, second in combinations(goals, 2):
        separation = float(np.linalg.norm(first[0] - second[0]))
        score = separation * (0.45 + 0.55 * min(first[2], second[2]))
        if best_pair is None or score > best_pair[0]:
            best_pair = (score, first, second)
    return [best_pair[1], best_pair[2]] if best_pair is not None else goals[:2]


def goal_mouth_anchor_candidates(
    goal_detections: Iterable[dict[str, Any]] | None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    goals = _goal_boxes(goal_detections)
    if len(goals) < 2:
        return []
    first_center, first_box, _ = goals[0]
    second_center, second_box, _ = goals[1]

    def facing_point(center: np.ndarray, box: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - center
        if float(np.linalg.norm(direction)) < 1e-8:
            return center.copy()
        x1, y1, x2, y2 = box
        candidates: list[tuple[float, np.ndarray]] = []
        if abs(direction[0]) > 1e-9:
            for x in (x1, x2):
                t = (x - center[0]) / direction[0]
                y = center[1] + t * direction[1]
                if t > 0 and y1 <= y <= y2:
                    candidates.append((t, np.array([x, y], dtype=np.float64)))
        if abs(direction[1]) > 1e-9:
            for y in (y1, y2):
                t = (y - center[1]) / direction[1]
                x = center[0] + t * direction[0]
                if t > 0 and x1 <= x <= x2:
                    candidates.append((t, np.array([x, y], dtype=np.float64)))
        return min(candidates, key=lambda item: item[0])[1] if candidates else center.copy()

    def near_score(box: np.ndarray, center: np.ndarray) -> float:
        area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        maximum_y = max(1.0, max(first_center[1], second_center[1]))
        return float(np.sqrt(area) * (1.0 + 0.25 * center[1] / maximum_y))

    first_is_near = near_score(first_box, first_center) >= near_score(second_box, second_center)

    def orient(first_anchor: np.ndarray, second_anchor: np.ndarray, label: str):
        if first_is_near:
            return first_anchor, second_anchor, label
        return second_anchor, first_anchor, label

    candidates: list[tuple[np.ndarray, np.ndarray, str]] = []

    # Axis-aligned YOLO boxes are most reliable at the inner vertical edge:
    # the left goal opens to the right and the right goal opens to the left.
    ordered = sorted(
        [(first_center, first_box, 0), (second_center, second_box, 1)],
        key=lambda item: item[0][0],
    )
    left_center, left_box, left_index = ordered[0]
    right_center, right_box, right_index = ordered[1]
    left_inner = np.array([left_box[2], 0.5 * (left_box[1] + left_box[3])], dtype=np.float64)
    right_inner = np.array([right_box[0], 0.5 * (right_box[1] + right_box[3])], dtype=np.float64)
    by_index = {left_index: left_inner, right_index: right_inner}
    candidates.append(orient(by_index[0], by_index[1], "bordes_internos"))

    candidates.append(
        orient(
            facing_point(first_center, first_box, second_center),
            facing_point(second_center, second_box, first_center),
            "rayo_hacia_campo",
        )
    )
    candidates.append(orient(first_center.copy(), second_center.copy(), "centros"))

    unique: list[tuple[np.ndarray, np.ndarray, str]] = []
    for near_anchor, far_anchor, label in candidates:
        if all(
            np.linalg.norm(near_anchor - prior[0]) + np.linalg.norm(far_anchor - prior[1]) > 8.0
            for prior in unique
        ):
            unique.append((near_anchor, far_anchor, label))
    return unique


def goal_mouth_anchors(goal_detections: Iterable[dict[str, Any]] | None) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = goal_mouth_anchor_candidates(goal_detections)
    if not candidates:
        return None
    return candidates[0][0], candidates[0][1]



class GoalAnchoredTemplateRegistrar:
    """Register the planar field from goals, physical rails and field markings.

    The old pipeline assigned semantic names to mask extrema (top=far,
    bottom=near), which is invalid under arbitrary camera rotation. This solver
    treats all candidate rails as unlabeled projective lines, uses the two goal
    mouths to establish the longitudinal direction, and scores complete
    homographies against a known marking template.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        template_config: FieldTemplateConfig | None = None,
        processing_max_dimension: int = 720,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.template_config = template_config or FieldTemplateConfig()
        self.template_points: TemplatePointSet = build_template_points(self.template_config, density=210)
        self.evidence_extractor = AdaptiveFieldEvidenceExtractor(maximum_lines=56)
        self.last_debug: dict[str, Any] = {}
        scale = min(1.0, float(processing_max_dimension) / max(self.frame_width, self.frame_height))
        self.scale = scale
        self.work_width = max(64, int(round(self.frame_width * scale)))
        self.work_height = max(64, int(round(self.frame_height * scale)))
        self.to_work = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.to_full = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _clean_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8) * 255
        if binary.shape[:2] != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((height, width), dtype=np.uint8)
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(binary)
        cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
        return clean

    @staticmethod
    def _white_mask(frame: np.ndarray, field_mask: np.ndarray, exclusion_boxes: Iterable[list[float]] | None = None) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        lightness = lab[:, :, 0]
        white = (((saturation < 92) & (value > 148)) | ((saturation < 125) & (lightness > 185)))
        white = (white & (field_mask > 0)).astype(np.uint8) * 255
        for box in exclusion_boxes or []:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            pad = 8
            cv2.rectangle(white, (max(0, x1 - pad), max(0, y1 - pad)), (min(white.shape[1] - 1, x2 + pad), min(white.shape[0] - 1, y2 + pad)), 0, cv2.FILLED)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return white

    @staticmethod
    def _rail_support(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        dark = ((val < 138) | ((val < 178) & (sat < 80))).astype(np.uint8)
        binary = (mask > 0).astype(np.uint8)
        eroded = cv2.erode(binary, np.ones((7, 7), np.uint8))
        boundary = cv2.subtract(binary, eroded)
        outside = cv2.dilate(binary, np.ones((31, 31), np.uint8)) - binary
        dark_outside = ((outside > 0) & (dark > 0)).astype(np.uint8)
        distance = cv2.distanceTransform((dark_outside == 0).astype(np.uint8), cv2.DIST_L2, 3)
        edges = cv2.dilate(cv2.Canny(gray, 45, 145), np.ones((3, 3), np.uint8))
        score = boundary.astype(np.float32) * np.exp(-distance / 7.0) * (0.55 + 0.45 * (edges > 0))
        margin_x = max(10, int(round(0.025 * mask.shape[1])))
        margin_y = max(10, int(round(0.025 * mask.shape[0])))
        score[:margin_y] = 0
        score[-margin_y:] = 0
        score[:, :margin_x] = 0
        score[:, -margin_x:] = 0
        return score.astype(np.float32)

    def _extract_rail_lines(self, frame: np.ndarray, mask: np.ndarray) -> list[RailLineCandidate]:
        support = self._rail_support(frame, mask)
        binary = (support > 0.055).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        diagonal = float(np.hypot(mask.shape[1], mask.shape[0]))
        lines = cv2.HoughLinesP(
            binary,
            1,
            np.pi / 720.0,
            threshold=max(18, int(0.025 * diagonal)),
            minLineLength=max(45, int(0.11 * diagonal)),
            maxLineGap=max(18, int(0.035 * diagonal)),
        )
        raw: list[RailLineCandidate] = []
        if lines is None:
            return raw
        for segment in lines[:, 0, :]:
            p1 = segment[:2].astype(np.float64)
            p2 = segment[2:].astype(np.float64)
            length = float(np.linalg.norm(p2 - p1))
            if length < 0.10 * diagonal:
                continue
            line = line_from_points(p1, p2)
            if line is None:
                continue
            samples = np.linspace(0.0, 1.0, 48)[:, None] * (p2 - p1) + p1
            xs = np.clip(np.round(samples[:, 0]).astype(int), 0, mask.shape[1] - 1)
            ys = np.clip(np.round(samples[:, 1]).astype(int), 0, mask.shape[0] - 1)
            mean_support = float(np.mean(support[ys, xs]))
            if mean_support < 0.035:
                continue
            raw.append(
                RailLineCandidate(
                    line=line,
                    segment=np.array([p1, p2], dtype=np.float64),
                    support=mean_support,
                    length_ratio=length / diagonal,
                )
            )

        # Deduplicate Hough fragments representing the same physical rail.
        selected: list[RailLineCandidate] = []
        raw.sort(key=lambda item: item.support * (0.35 + item.length_ratio), reverse=True)
        for candidate in raw:
            midpoint = np.mean(candidate.segment, axis=0)
            angle = float(np.arctan2(-candidate.line[0], candidate.line[1]))
            duplicate = False
            for existing in selected:
                existing_midpoint = np.mean(existing.segment, axis=0)
                existing_angle = float(np.arctan2(-existing.line[0], existing.line[1]))
                angle_delta = abs(np.arctan2(np.sin(angle - existing_angle), np.cos(angle - existing_angle)))
                angle_delta = min(angle_delta, abs(np.pi - angle_delta))
                line_distance = abs(_signed_distance(existing.line, midpoint))
                reverse_distance = abs(_signed_distance(candidate.line, existing_midpoint))
                if angle_delta < np.deg2rad(6.0) and min(line_distance, reverse_distance) < 22.0:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
            if len(selected) >= 12:
                break
        return selected


    @staticmethod
    def _deduplicate_lines(
        candidates: list[RailLineCandidate],
        maximum: int,
        angle_tolerance_degrees: float = 3.8,
        distance_tolerance: float = 20.0,
    ) -> list[RailLineCandidate]:
        selected: list[RailLineCandidate] = []
        candidates = sorted(
            candidates,
            key=lambda item: item.support * (0.35 + item.length_ratio),
            reverse=True,
        )
        for candidate in candidates:
            midpoint = np.mean(candidate.segment, axis=0)
            angle = float(np.arctan2(-candidate.line[0], candidate.line[1]))
            duplicate = False
            for existing in selected:
                existing_midpoint = np.mean(existing.segment, axis=0)
                existing_angle = float(np.arctan2(-existing.line[0], existing.line[1]))
                angle_delta = abs(
                    np.arctan2(
                        np.sin(angle - existing_angle),
                        np.cos(angle - existing_angle),
                    )
                )
                angle_delta = min(angle_delta, abs(np.pi - angle_delta))
                line_distance = min(
                    abs(_signed_distance(existing.line, midpoint)),
                    abs(_signed_distance(candidate.line, existing_midpoint)),
                )
                if (
                    angle_delta < np.deg2rad(angle_tolerance_degrees)
                    and line_distance < distance_tolerance
                ):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
            if len(selected) >= maximum:
                break
        return selected

    def _extract_projective_lines(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        white_mask: np.ndarray,
    ) -> tuple[list[RailLineCandidate], list[RailLineCandidate]]:
        """Extract outer-side proposals and generic field-marking lines.

        Outer proposals combine white paint close to the segmented surface
        boundary with dark-rail evidence.  Generic marking lines also include
        center/end lines; intersections between those lines supply the second
        projective vanishing point.  This is deliberately different from the
        V5.4 rail-only bootstrap, which could confuse a cropped image edge with
        a real field side.
        """

        height, width = mask.shape[:2]
        diagonal = float(np.hypot(width, height))
        boundary = cv2.morphologyEx(
            (mask > 0).astype(np.uint8) * 255,
            cv2.MORPH_GRADIENT,
            np.ones((9, 9), np.uint8),
        )
        boundary_distance = cv2.distanceTransform(
            (boundary == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        rail_support = self._rail_support(frame, mask)

        near_boundary = boundary_distance < max(24.0, 0.065 * diagonal)
        outer_white = ((white_mask > 0) & near_boundary).astype(np.uint8) * 255
        rail_binary = (rail_support > 0.024).astype(np.uint8) * 255
        side_binary = cv2.max(outer_white, rail_binary)
        # Never let the image crop itself become a field line.  This is a hard
        # exclusion, not a weak score: there are no pixels beyond a camera
        # margin with which to prove a physical surface transition.
        margin_x = max(10, int(round(0.030 * width)))
        margin_y = max(10, int(round(0.030 * height)))
        side_binary[:margin_y, :] = 0
        side_binary[-margin_y:, :] = 0
        side_binary[:, :margin_x] = 0
        side_binary[:, -margin_x:] = 0
        side_binary = cv2.morphologyEx(
            side_binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        side_edges = cv2.dilate(
            cv2.Canny(side_binary, 30, 105), np.ones((3, 3), np.uint8)
        )

        marking_edges = cv2.dilate(
            cv2.Canny(white_mask, 30, 105), np.ones((3, 3), np.uint8)
        )

        def extract(
            edge_image: np.ndarray,
            minimum_length_ratio: float,
            maximum_gap_ratio: float,
            outer_mode: bool,
        ) -> list[RailLineCandidate]:
            lines = cv2.HoughLinesP(
                edge_image,
                1,
                np.pi / 720.0,
                threshold=max(18, int(0.018 * diagonal)),
                minLineLength=max(34, int(minimum_length_ratio * diagonal)),
                maxLineGap=max(18, int(maximum_gap_ratio * diagonal)),
            )
            output: list[RailLineCandidate] = []
            if lines is None:
                return output
            for raw_segment in lines[:, 0, :]:
                p1 = raw_segment[:2].astype(np.float64)
                p2 = raw_segment[2:].astype(np.float64)
                length = float(np.linalg.norm(p2 - p1))
                if length < minimum_length_ratio * diagonal:
                    continue
                line = line_from_points(p1, p2)
                if line is None:
                    continue
                samples = (
                    np.linspace(0.0, 1.0, 56)[:, None] * (p2 - p1) + p1
                )
                xs = np.clip(np.round(samples[:, 0]).astype(int), 0, width - 1)
                ys = np.clip(np.round(samples[:, 1]).astype(int), 0, height - 1)
                white_fraction = float(np.mean(white_mask[ys, xs] > 0))
                rail_value = float(np.mean(rail_support[ys, xs]))
                boundary_value = float(
                    np.mean(np.exp(-boundary_distance[ys, xs] / 28.0))
                )
                if outer_mode:
                    evidence = (
                        0.30
                        + 0.62 * white_fraction
                        + 0.90 * min(1.0, rail_value / 0.12)
                    ) * (0.48 + 0.52 * boundary_value)
                    if white_fraction < 0.04 and rail_value < 0.018:
                        continue
                else:
                    evidence = 0.42 + 0.58 * white_fraction
                    if white_fraction < 0.08:
                        continue
                support = float(
                    np.clip((length / diagonal) * evidence, 0.0, 1.0)
                )
                output.append(
                    RailLineCandidate(
                        line=line,
                        segment=np.array([p1, p2], dtype=np.float64),
                        support=support,
                        length_ratio=length / diagonal,
                    )
                )
            return output

        sides = self._deduplicate_lines(
            extract(side_edges, 0.043, 0.042, True), maximum=26
        )
        markings = self._deduplicate_lines(
            extract(marking_edges, 0.032, 0.032, False), maximum=38
        )
        return sides, markings

    @staticmethod
    def _marking_vanishing_candidates(
        width: int,
        height: int,
        marking_lines: list[RailLineCandidate],
    ) -> list[np.ndarray]:
        diagonal = float(np.hypot(width, height))
        weighted: list[tuple[float, np.ndarray]] = []
        for first, second in combinations(marking_lines[:32], 2):
            point = intersect_lines(first.line, second.line)
            if point is None or not np.isfinite(point).all():
                continue
            if np.any(np.abs(point) > 9.0 * max(width, height)):
                continue
            angle_first = float(np.arctan2(-first.line[0], first.line[1]))
            angle_second = float(np.arctan2(-second.line[0], second.line[1]))
            delta = abs(
                np.arctan2(
                    np.sin(angle_first - angle_second),
                    np.cos(angle_first - angle_second),
                )
            )
            delta = min(delta, abs(np.pi - delta))
            # Very close image angles make an unstable, numerically distant
            # intersection.  Field-family lines still have visible perspective,
            # so retain moderate and large angular differences.
            if delta < np.deg2rad(4.0):
                continue
            score = (
                first.support
                * second.support
                * (0.35 + 0.65 * min(1.0, delta / np.deg2rad(35.0)))
            )
            weighted.append((float(score), point))

        weighted.sort(key=lambda item: item[0], reverse=True)
        unique: list[np.ndarray] = []
        threshold = max(28.0, 0.035 * diagonal)
        for _score, point in weighted:
            if all(float(np.linalg.norm(point - prior)) > threshold for prior in unique):
                unique.append(point)
            if len(unique) >= 150:
                break

        # Coarse off-frame hypotheses remain a last-resort fallback when only
        # one end-family line is visible.  They are evaluated by the full
        # template/mask/goal objective, never trusted from geometry alone.
        if len(unique) < 18:
            unique.extend(
                GoalAnchoredTemplateRegistrar._vanishing_candidates(
                    width, height, marking_lines
                )
            )
        return unique[:180]

    def _distance_map(self, white_mask: np.ndarray) -> np.ndarray:
        inverted = (white_mask == 0).astype(np.uint8)
        return cv2.distanceTransform(inverted, cv2.DIST_L2, 3)

    def extract_marking_mask(
        self,
        frame: np.ndarray,
        field_mask: np.ndarray,
        exclusion_boxes: Iterable[list[float]] | None = None,
    ) -> np.ndarray:
        """Public full-resolution paint mask used by the temporal accumulator."""
        evidence = self.evidence_extractor.extract(frame, field_mask, exclusion_boxes)
        return evidence.marking_mask

    @staticmethod
    def _as_rail_candidate(candidate: VisualLineCandidate) -> RailLineCandidate:
        return RailLineCandidate(
            line=np.asarray(candidate.line, dtype=np.float64),
            segment=np.asarray(candidate.segment, dtype=np.float64),
            support=float(candidate.support),
            length_ratio=float(candidate.length_ratio),
        )

    @staticmethod
    def _line_pair_quality(
        first: RailLineCandidate,
        second: RailLineCandidate,
        diagonal: float,
    ) -> float:
        midpoint_first = np.mean(first.segment, axis=0)
        midpoint_second = np.mean(second.segment, axis=0)
        separation = min(
            abs(_signed_distance(first.line, midpoint_second)),
            abs(_signed_distance(second.line, midpoint_first)),
        )
        if separation < 0.018 * diagonal:
            return 0.0
        vector_first = first.segment[1] - first.segment[0]
        vector_second = second.segment[1] - second.segment[0]
        angle_first = float(np.arctan2(vector_first[1], vector_first[0]))
        angle_second = float(np.arctan2(vector_second[1], vector_second[0]))
        delta = abs(np.arctan2(np.sin(angle_first - angle_second), np.cos(angle_first - angle_second)))
        delta = min(delta, abs(np.pi - delta))
        # Lines from one projective family may converge, but a near-perpendicular
        # pair almost certainly belongs to different field directions.
        # Strong close-range perspective can make two valid sidelines differ
        # by more than 50 degrees. V8's 48-degree cutoff discarded exactly the
        # converging pair needed in portrait videos. Keep broad hypotheses and
        # let dense template/goal/boundary scoring reject cross-family pairs.
        if delta > np.deg2rad(82.0):
            return 0.0
        separation_score = float(np.clip(separation / (0.16 * diagonal), 0.0, 1.0))
        angle_score = float(np.exp(-0.5 * (delta / np.deg2rad(42.0)) ** 2))
        return float(
            first.support
            * second.support
            * (0.35 + 0.65 * separation_score)
            * (0.45 + 0.55 * angle_score)
        )

    def _rank_line_pairs(
        self,
        lines: list[RailLineCandidate],
        diagonal: float,
        maximum: int,
    ) -> list[tuple[RailLineCandidate, RailLineCandidate, float]]:
        ranked: list[tuple[RailLineCandidate, RailLineCandidate, float]] = []
        for first, second in combinations(lines, 2):
            quality = self._line_pair_quality(first, second, diagonal)
            if quality > 0.0:
                ranked.append((first, second, quality))
        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked[:maximum]

    @staticmethod
    def _goal_mouth_points(
        goal_detections: Iterable[dict[str, Any]] | None,
        field_mask: np.ndarray,
    ) -> list[tuple[np.ndarray, float, np.ndarray, float]]:
        """Return goal mouth points plus the observed goal-axis segment.

        This works with one visible goal, unlike V8's two-goal-only anchor
        routine. The facing point is the intersection of the box boundary and
        the ray from the box centre toward the field centroid.
        """
        moments = cv2.moments((field_mask > 0).astype(np.uint8))
        if abs(float(moments.get("m00", 0.0))) > 1e-8:
            field_center = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
        else:
            height, width = field_mask.shape[:2]
            field_center = np.array([0.5 * width, 0.5 * height], dtype=np.float64)

        output: list[tuple[np.ndarray, float, np.ndarray, float]] = []
        for detection in goal_detections or []:
            if str(detection.get("class_group", "")).lower() != "goal":
                continue
            box = detection.get("bbox_xyxy", [])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue
            center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
            direction = field_center - center
            candidates: list[tuple[float, np.ndarray]] = []
            if abs(float(direction[0])) > 1e-9:
                for x in (x1, x2):
                    t = (x - center[0]) / direction[0]
                    y = center[1] + t * direction[1]
                    if t > 0.0 and y1 <= y <= y2:
                        candidates.append((t, np.array([x, y], dtype=np.float64)))
            if abs(float(direction[1])) > 1e-9:
                for y in (y1, y2):
                    t = (y - center[1]) / direction[1]
                    x = center[0] + t * direction[0]
                    if t > 0.0 and x1 <= x <= x2:
                        candidates.append((t, np.array([x, y], dtype=np.float64)))
            point = min(candidates, key=lambda item: item[0])[1] if candidates else center
            confidence = float(np.clip(detection.get("confidence", 0.5), 0.0, 1.0))
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width >= box_height:
                axis = np.array([[x1, center[1]], [x2, center[1]]], dtype=np.float64)
            else:
                axis = np.array([[center[0], y1], [center[0], y2]], dtype=np.float64)
            aspect = max(box_width, box_height) / max(1.0, min(box_width, box_height))
            axis_reliability = float(np.clip((aspect - 1.35) / 0.90, 0.0, 1.0))
            output.append((point, confidence, axis, axis_reliability))
            if float(np.linalg.norm(point - center)) > 3.0:
                output.append((center, 0.82 * confidence, axis, axis_reliability))
        return output

    @staticmethod
    def _single_goal_score(
        field_to_image: np.ndarray,
        goal_points: list[tuple[np.ndarray, float, np.ndarray, float]],
        diagonal: float,
    ) -> float:
        """Score goal position and orientation against canonical end lines.

        A point-only goal score cannot distinguish a field whose longitudinal
        and transverse line families were swapped.  The long axis of the goal
        bounding box must also agree with the projected near/far end line.
        """
        if not goal_points:
            return 0.50
        endpoint_segments = (
            np.float32([[0.0, 0.0], [0.0, 1.0]]),
            np.float32([[1.0, 0.0], [1.0, 1.0]]),
        )
        projected_segments = [
            _project_points(segment, field_to_image) for segment in endpoint_segments
        ]
        projected_centers = np.asarray(
            [0.5 * (segment[0] + segment[1]) for segment in projected_segments],
            dtype=np.float64,
        )
        if not np.isfinite(projected_centers).all():
            return 0.0

        position_scale = max(10.0, 0.045 * diagonal)
        line_scale = max(8.0, 0.032 * diagonal)
        pair_scores: list[float] = []
        weights: list[float] = []
        for point, confidence, observed_axis, axis_reliability in goal_points:
            observed_axis = np.asarray(observed_axis, dtype=np.float64).reshape(2, 2)
            observed_vector = observed_axis[1] - observed_axis[0]
            observed_norm = float(np.linalg.norm(observed_vector))
            if observed_norm < 2.0:
                continue
            observed_unit = observed_vector / observed_norm
            alternatives: list[float] = []
            for center, projected_segment in zip(projected_centers, projected_segments):
                projected_vector = projected_segment[1] - projected_segment[0]
                projected_norm = float(np.linalg.norm(projected_vector))
                if projected_norm < 1e-8:
                    continue
                projected_unit = projected_vector / projected_norm
                cosine = float(np.clip(abs(np.dot(observed_unit, projected_unit)), 0.0, 1.0))
                angle_error = float(np.degrees(np.arccos(cosine)))
                angle_score = float(np.exp(-0.5 * (angle_error / 16.0) ** 2))
                position_error = float(np.linalg.norm(center - point))
                position_score = float(np.exp(-0.5 * (position_error / position_scale) ** 2))
                projected_line = line_from_points(projected_segment[0], projected_segment[1])
                if projected_line is None:
                    continue
                line_error = abs(float(point @ projected_line[:2] + projected_line[2]))
                line_score = float(np.exp(-0.5 * (line_error / line_scale) ** 2))
                reliability = float(np.clip(axis_reliability, 0.0, 1.0))
                alternatives.append(
                    (0.72 - 0.30 * reliability) * position_score
                    + (0.28 - 0.10 * reliability) * line_score
                    + 0.40 * reliability * angle_score
                )
            if alternatives:
                pair_scores.append(max(alternatives))
                weights.append(0.35 + 0.65 * confidence)
        if not pair_scores:
            return 0.0
        return float(np.average(np.asarray(pair_scores), weights=np.asarray(weights)))

    @staticmethod
    def _render_projected_markings(
        projected: np.ndarray,
        groups: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for group in (1, 2, 3):
            points = projected[groups == group]
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if len(points) < 2:
                continue
            points = np.round(points).astype(np.int32)
            valid = (
                (points[:, 0] >= -2)
                & (points[:, 0] <= width + 1)
                & (points[:, 1] >= -2)
                & (points[:, 1] <= height + 1)
            )
            points = points[valid]
            for point in points:
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    canvas[point[1], point[0]] = 255
        return cv2.dilate(canvas, np.ones((5, 5), np.uint8))

    def _reverse_chamfer_score(
        self,
        projected: np.ndarray,
        groups: np.ndarray,
        white_mask: np.ndarray,
        quad_mask: np.ndarray,
    ) -> float:
        """Score observed paint against projected markings with bounded work.

        V9 rasterized a candidate template and ran a full distance transform for
        every hypothesis. Dense real frames can make hundreds of repeated
        transforms dominate runtime. A small nearest-neighbour query is
        equivalent for the sampled Chamfer term and has a strict point cap.
        """
        height, width = white_mask.shape[:2]
        template = np.asarray(projected, dtype=np.float64)[np.asarray(groups) > 0]
        template = template[
            np.isfinite(template).all(axis=1)
            & (template[:, 0] >= 0)
            & (template[:, 0] < width)
            & (template[:, 1] >= 0)
            & (template[:, 1] < height)
        ]
        if len(template) < 12:
            return 0.0
        if len(template) > 1600:
            template = template[:: max(1, len(template) // 1600)][:1600]

        observed_yx = np.column_stack(np.nonzero((white_mask > 0) & (quad_mask > 0)))
        if len(observed_yx) < 16:
            return 0.0
        if len(observed_yx) > 2200:
            observed_yx = observed_yx[:: max(1, len(observed_yx) // 2200)][:2200]
        observed = observed_yx[:, [1, 0]].astype(np.float64)

        if cKDTree is not None:
            tree = cKDTree(template, compact_nodes=True, balanced_tree=True)
            distances, _ = tree.query(observed, k=1, workers=1)
        else:
            rendered = self._render_projected_markings(projected, groups, width, height)
            if np.count_nonzero(rendered) < 12:
                return 0.0
            distance = cv2.distanceTransform(
                (rendered == 0).astype(np.uint8), cv2.DIST_L2, 3
            )
            distances = distance[observed_yx[:, 0], observed_yx[:, 1]]
        return float(np.mean(np.exp(-0.5 * (np.asarray(distances) / 5.5) ** 2)))

    @staticmethod
    def _directional_boundary_evidence(
        frame: np.ndarray,
        mask: np.ndarray,
        corners: np.ndarray,
    ) -> tuple[float, int, dict[str, float]]:
        """Verify that projected outer sides separate field from dark exterior.

        Distance to a segmentation contour is not enough: an interior white
        line may happen to sit near a noisy contour and was previously allowed
        to masquerade as a field edge.  V10 samples both sides of each projected
        boundary.  The inward samples must remain on the playing surface while
        the outward samples must leave it and, when visible, become darker.
        """
        height, width = mask.shape[:2]
        diagonal = float(np.hypot(width, height))
        polygon = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        center = np.mean(polygon, axis=0)
        field = cv2.morphologyEx(
            (mask > 0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
        side_names = ("left_long", "far_end", "right_long", "near_end")
        side_scores: dict[str, float] = {}
        strong_count = 0

        base_offset = float(np.clip(0.008 * diagonal, 5.0, 14.0))
        offsets = (base_offset, 1.65 * base_offset)
        margin = max(2, int(round(0.004 * diagonal)))

        for index, name in enumerate(side_names):
            first = polygon[index]
            second = polygon[(index + 1) % 4]
            direction = second - first
            length = float(np.linalg.norm(direction))
            if length < 0.04 * diagonal:
                side_scores[name] = 0.0
                continue
            normal = np.array([-direction[1], direction[0]], dtype=np.float64) / max(length, 1e-9)
            midpoint = 0.5 * (first + second)
            if float(np.dot(center - midpoint, normal)) < 0.0:
                normal *= -1.0

            sample_count = int(np.clip(length / max(8.0, 0.018 * diagonal), 18, 54))
            base = (
                np.linspace(0.10, 0.90, sample_count, dtype=np.float64)[:, None]
                * direction[None, :]
                + first[None, :]
            )
            inside_sets = [base + normal[None, :] * offset for offset in offsets]
            outside_sets = [base - normal[None, :] * offset for offset in offsets]
            all_sets = [*inside_sets, *outside_sets]
            valid = np.ones(sample_count, dtype=bool)
            for points in all_sets:
                valid &= (
                    (points[:, 0] >= margin)
                    & (points[:, 0] < width - margin)
                    & (points[:, 1] >= margin)
                    & (points[:, 1] < height - margin)
                )
            if int(np.count_nonzero(valid)) < 8:
                side_scores[name] = 0.0
                continue

            def sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
                selected = points[valid]
                xs = np.clip(np.rint(selected[:, 0]).astype(int), 0, width - 1)
                ys = np.clip(np.rint(selected[:, 1]).astype(int), 0, height - 1)
                return image[ys, xs].astype(np.float32)

            inside_field = np.mean(
                np.vstack([sample(field, points) for points in inside_sets]), axis=0
            )
            outside_field = np.mean(
                np.vstack([sample(field, points) for points in outside_sets]), axis=0
            )
            inside_value = np.mean(
                np.vstack([sample(value, points) for points in inside_sets]), axis=0
            )
            outside_value = np.mean(
                np.vstack([sample(value, points) for points in outside_sets]), axis=0
            )

            inside_ratio = float(np.mean(inside_field > 0.50))
            outside_ratio = float(np.mean(outside_field < 0.35))
            transition = float(np.sqrt(max(0.0, inside_ratio * outside_ratio)))
            contrast = float(
                np.mean(np.clip((inside_value - outside_value - 5.0) / 48.0, 0.0, 1.0))
            )
            outside_dark = float(np.mean(outside_value < 145.0))
            color_support = max(contrast, 0.72 * outside_dark)
            score = float(np.clip(transition * (0.72 + 0.28 * color_support), 0.0, 1.0))
            side_scores[name] = score
            if score >= 0.58:
                strong_count += 1

        nonzero = [value for value in side_scores.values() if value > 0.0]
        overall = float(np.mean(sorted(nonzero, reverse=True)[:2])) if nonzero else 0.0
        return overall, strong_count, side_scores

    def _cheap_candidate_score(
        self,
        corners: np.ndarray,
        mask: np.ndarray,
        white_distance: np.ndarray,
        boundary_distance: np.ndarray,
        goal_points: list[tuple[np.ndarray, float, np.ndarray, float]],
    ) -> float:
        height, width = mask.shape[:2]
        if not _quad_is_sane(corners, width, height):
            return -1.0
        field_to_image = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.asarray(corners, dtype=np.float32),
        )
        sample = self.template_points.points[::5]
        weights = self.template_points.weights[::5]
        groups = self.template_points.groups[::5]
        projected = _project_points(sample, field_to_image)
        inside = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 1)
            & (projected[:, 0] < width - 1)
            & (projected[:, 1] >= 1)
            & (projected[:, 1] < height - 1)
        )
        if np.count_nonzero(inside) < 12:
            return -1.0
        points = projected[inside]
        xs = np.round(points[:, 0]).astype(int)
        ys = np.round(points[:, 1]).astype(int)
        visible_groups = groups[inside]
        visible_weights = weights[inside]
        scores = np.zeros(len(points), dtype=np.float64)
        marking = visible_groups > 0
        boundary = visible_groups == 0
        scores[marking] = np.exp(-0.5 * (white_distance[ys[marking], xs[marking]] / 7.0) ** 2)
        scores[boundary] = np.exp(-0.5 * (boundary_distance[ys[boundary], xs[boundary]] / 6.5) ** 2)
        template = float(np.average(scores, weights=visible_weights))

        # Cheap ranking must stay truly cheap. V9 rasterized a full-frame
        # quadrilateral for every one of thousands of line assignments; on a
        # dense real frame this became both the dominant cost and a source of
        # pathological OpenCV stalls. Sample a fixed canonical interior grid
        # instead. Full mask precision/recall is still computed later for only
        # the strongest hypotheses in ``_candidate_score``.
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.06, 0.94, 9, dtype=np.float32),
            np.linspace(0.08, 0.92, 6, dtype=np.float32),
        )
        interior = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        projected_interior = _project_points(interior, field_to_image)
        visible_interior = (
            np.isfinite(projected_interior).all(axis=1)
            & (projected_interior[:, 0] >= 1)
            & (projected_interior[:, 0] < width - 1)
            & (projected_interior[:, 1] >= 1)
            & (projected_interior[:, 1] < height - 1)
        )
        if np.count_nonzero(visible_interior) < 6:
            field_support = 0.0
        else:
            interior_points = projected_interior[visible_interior]
            interior_x = np.rint(interior_points[:, 0]).astype(int)
            interior_y = np.rint(interior_points[:, 1]).astype(int)
            field_support = float(np.mean(mask[interior_y, interior_x] > 0))
        visible_ratio = float(np.mean(visible_interior))
        support = field_support * (0.68 + 0.32 * visible_ratio)
        goal = self._single_goal_score(field_to_image, goal_points, float(np.hypot(width, height)))
        return float(0.68 * template + 0.22 * support + 0.10 * goal)

    def _unlabeled_grid_candidates(
        self,
        mask: np.ndarray,
        white_distance: np.ndarray,
        boundary_distance: np.ndarray,
        side_candidates: list[RailLineCandidate],
        marking_lines: list[RailLineCandidate],
        manual_segments: dict[str, np.ndarray],
        goal_points: list[tuple[np.ndarray, float, np.ndarray, float]],
        maximum: int = 72,
    ) -> list[tuple[np.ndarray, dict[str, float], str]]:
        """Generate 2x2 line-grid hypotheses without pre-labelling detections.

        Semantic identity is assigned *inside each hypothesis*, then the whole
        projected template decides which assignment is plausible. This removes
        V8's circular dependency: knowing the homography before knowing the line
        name, while requiring the line name before estimating the homography.
        """
        diagonal = float(np.hypot(mask.shape[1], mask.shape[0]))
        transverse_names = ("near", "near_area", "center", "far_area", "far")
        longitudinal_names = ("left", "right")

        manual_lines: dict[str, np.ndarray] = {}
        for name, segment in manual_segments.items():
            if name in CANONICAL_LINES_NORMALIZED:
                value = line_from_points(segment[0], segment[1])
                if value is not None:
                    manual_lines[name] = value

        # Long field sides are usually present in boundary proposals; include a
        # few strong paint lines because some arenas paint the outer border.
        longitudinal_pool = self._deduplicate_lines(
            [*side_candidates[:14], *marking_lines[:10]], maximum=14
        )
        transverse_pool = self._deduplicate_lines(
            [*marking_lines[:22], *side_candidates[:8]], maximum=22
        )

        def family_assignments(
            family_names: tuple[str, ...],
            pool: list[RailLineCandidate],
            maximum_pairs: int,
        ) -> list[tuple[dict[str, np.ndarray], dict[str, float], float]]:
            fixed = {name: manual_lines[name] for name in family_names if name in manual_lines}
            output: list[tuple[dict[str, np.ndarray], dict[str, float], float]] = []
            if len(fixed) >= 2:
                output.append((fixed, {name: 1.0 for name in fixed}, 1.0))
                return output
            if len(fixed) == 1:
                fixed_name = next(iter(fixed))
                for candidate in pool[:16]:
                    # Reject automatic lines coincident with the manual one.
                    midpoint = np.mean(candidate.segment, axis=0)
                    if abs(_signed_distance(fixed[fixed_name], midpoint)) < 0.014 * diagonal:
                        continue
                    for other_name in family_names:
                        if other_name == fixed_name:
                            continue
                        score = 0.72 + 0.28 * candidate.support
                        output.append(
                            (
                                {fixed_name: fixed[fixed_name], other_name: candidate.line},
                                {fixed_name: 1.0, other_name: float(0.68 + 0.22 * candidate.support)},
                                score,
                            )
                        )
                output.sort(key=lambda item: item[2], reverse=True)
                return output[:maximum_pairs]

            pairs = self._rank_line_pairs(pool[:18], diagonal, maximum_pairs)
            for first, second, pair_quality in pairs:
                for first_name in family_names:
                    for second_name in family_names:
                        if first_name == second_name:
                            continue
                        # Avoid nearly adjacent canonical coordinates when the
                        # observed lines are widely separated, and vice versa.
                        first_coordinate = float(-CANONICAL_LINES_NORMALIZED[first_name][2])
                        second_coordinate = float(-CANONICAL_LINES_NORMALIZED[second_name][2])
                        canonical_separation = abs(first_coordinate - second_coordinate)
                        if canonical_separation < 0.14:
                            continue
                        seed_score = float(0.55 + 0.35 * np.clip(pair_quality, 0.0, 1.0))
                        output.append(
                            (
                                {first_name: first.line, second_name: second.line},
                                {first_name: seed_score, second_name: seed_score},
                                pair_quality,
                            )
                        )
            output.sort(key=lambda item: item[2], reverse=True)
            return output[:maximum_pairs]

        longitudinal = family_assignments(longitudinal_names, longitudinal_pool, 24)
        transverse = family_assignments(transverse_names, transverse_pool, 110)
        if not longitudinal or not transverse:
            return []

        heap: list[tuple[float, int, np.ndarray, dict[str, float]]] = []
        counter = 0
        attempted = 0
        for long_lines, long_matches, long_quality in longitudinal:
            for trans_lines, trans_matches, trans_quality in transverse:
                attempted += 1
                semantic_lines = {**long_lines, **trans_lines}
                solved = None
                try:
                    # Imported at module scope; this routine uses intersections
                    # of every selected transverse/longitudinal pair.
                    from src.I_field_geometry.feature_constraints import homography_from_semantic_lines
                    solved = homography_from_semantic_lines(semantic_lines)
                except (ValueError, np.linalg.LinAlgError):
                    solved = None
                if solved is None:
                    continue
                _image_to_field, field_to_image = solved
                corners = _project_points(
                    np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), field_to_image
                ).astype(np.float32)
                score = self._cheap_candidate_score(
                    corners, mask, white_distance, boundary_distance, goal_points
                )
                if score < 0.34:
                    continue
                score += 0.025 * min(long_quality, trans_quality)
                matches = {**long_matches, **trans_matches}
                item = (float(score), counter, corners, matches)
                counter += 1
                if len(heap) < maximum:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        selected = sorted(heap, key=lambda item: item[0], reverse=True)
        return [
            (corners, matches, "rejilla_no_etiquetada")
            for _score, _counter, corners, matches in selected
        ]

    @staticmethod
    def _automatic_feature_matches(
        field_to_image: np.ndarray,
        marking_lines: list[RailLineCandidate],
        diagonal: float,
    ) -> tuple[float, int, dict[str, float]]:
        """Match canonical features to distinct observed segments efficiently.

        V8 called ``score_feature_anchor`` for every feature/segment pair.  That
        helper inverted the same homography on every call, which dominated CPU
        time.  V9 projects all observed segments into field coordinates once and
        all canonical segments into image coordinates once, then evaluates the
        complete score matrix with vectorized NumPy operations.
        """
        candidates = marking_lines[:26]
        if not candidates:
            return 0.0, 0, {}
        try:
            image_to_field = np.linalg.inv(np.asarray(field_to_image, dtype=np.float64))
        except np.linalg.LinAlgError:
            return 0.0, 0, {}

        observed = np.asarray(
            [candidate.segment for candidate in candidates], dtype=np.float32
        ).reshape(-1, 2, 2)
        observed_field = cv2.perspectiveTransform(
            observed.reshape(1, -1, 2), image_to_field
        ).reshape(-1, 2, 2).astype(np.float64)
        observed64 = observed.astype(np.float64)
        observed_vectors = observed64[:, 1] - observed64[:, 0]
        observed_norms = np.linalg.norm(observed_vectors, axis=1)
        safe_observed_vectors = observed_vectors / np.maximum(observed_norms[:, None], 1e-9)

        proposals: list[tuple[float, str, int]] = []
        for name, canonical_segment in CANONICAL_SEGMENTS_NORMALIZED.items():
            projected = cv2.perspectiveTransform(
                np.asarray(canonical_segment, dtype=np.float32).reshape(1, 2, 2),
                np.asarray(field_to_image, dtype=np.float64),
            ).reshape(2, 2).astype(np.float64)
            if not np.isfinite(projected).all():
                continue
            projected_vector = projected[1] - projected[0]
            projected_norm = float(np.linalg.norm(projected_vector))
            if projected_norm < 1e-8:
                continue
            projected_unit = projected_vector / projected_norm
            projected_line = line_from_points(projected[0], projected[1])
            if projected_line is None:
                continue

            perpendicular_error = np.mean(
                np.abs(
                    observed64 @ projected_line[:2]
                    + projected_line[2]
                ),
                axis=1,
            )
            cosine = np.clip(
                np.abs(safe_observed_vectors @ projected_unit), 0.0, 1.0
            )
            angle_error = np.degrees(np.arccos(cosine))

            expected = float(-CANONICAL_LINES_NORMALIZED[name][2])
            if name in TRANSVERSE_FEATURES:
                canonical_error = np.mean(
                    np.abs(observed_field[:, :, 0] - expected), axis=1
                )
                along = observed_field[:, :, 1]
            else:
                canonical_error = np.mean(
                    np.abs(observed_field[:, :, 1] - expected), axis=1
                )
                along = observed_field[:, :, 0]
            along_span = np.abs(along[:, 1] - along[:, 0])
            along_inside = (
                np.isfinite(along).all(axis=1)
                & (np.max(along, axis=1) >= -0.20)
                & (np.min(along, axis=1) <= 1.20)
            )

            distance_scale = max(3.0, 0.0065 * float(diagonal))
            distance_score = np.exp(
                -0.5 * (perpendicular_error / distance_scale) ** 2
            )
            angle_score = np.exp(-0.5 * (angle_error / 5.5) ** 2)
            canonical_score = np.exp(-0.5 * (canonical_error / 0.025) ** 2)
            canonical_score[~np.isfinite(canonical_error)] = 0.0
            span_score = np.clip(along_span / 0.12, 0.0, 1.0)
            scores = (
                0.34 * distance_score
                + 0.28 * angle_score
                + 0.30 * canonical_score
                + 0.08 * span_score
            )
            relaxed = (
                (scores >= 0.64)
                & (perpendicular_error <= max(9.0, 0.016 * diagonal))
                & (angle_error <= 12.0)
                & (canonical_error <= 0.075)
                & (along_span >= 0.025)
                & along_inside
                & (observed_norms >= 8.0)
            )
            for index in np.flatnonzero(relaxed):
                proposals.append((float(scores[index]), name, int(index)))

        proposals.sort(reverse=True)
        used_features: set[str] = set()
        used_segments: set[int] = set()
        matches: dict[str, float] = {}
        for score, name, index in proposals:
            if name in used_features or index in used_segments:
                continue
            matches[name] = score
            used_features.add(name)
            used_segments.add(index)
        return (
            float(np.mean(list(matches.values()))) if matches else 0.0,
            len(matches),
            matches,
        )

    def _candidate_score(
        self,
        corners: np.ndarray,
        frame: np.ndarray,
        mask: np.ndarray,
        white_mask: np.ndarray,
        white_distance: np.ndarray,
        boundary_distance: np.ndarray,
        goal_points: list[tuple[np.ndarray, float, np.ndarray, float]],
        near_anchor: np.ndarray | None,
        far_anchor: np.ndarray | None,
        rail_pair: tuple[RailLineCandidate, RailLineCandidate] | None,
        semantic_segments: dict[str, np.ndarray] | None = None,
        marking_lines: list[RailLineCandidate] | None = None,
        seed_matches: dict[str, float] | None = None,
    ) -> tuple[Any, ...] | None:
        height, width = mask.shape[:2]
        if not _quad_is_sane(corners, width, height):
            return None
        field_to_image = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.asarray(corners, dtype=np.float32),
        )
        image_to_field = _safe_inverse(field_to_image)
        if image_to_field is None:
            return None

        projected = _project_points(self.template_points.points, field_to_image)
        finite = np.isfinite(projected).all(axis=1)
        inside = (
            finite
            & (projected[:, 0] >= 1)
            & (projected[:, 0] < width - 1)
            & (projected[:, 1] >= 1)
            & (projected[:, 1] < height - 1)
        )
        visible_fraction = float(
            np.sum(self.template_points.weights[inside])
            / max(1e-6, np.sum(self.template_points.weights))
        )
        if np.count_nonzero(inside) < 24 or visible_fraction < 0.020:
            return None
        points = projected[inside]
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
        weights = self.template_points.weights[inside]
        groups = self.template_points.groups[inside]

        marking_selector = groups > 0
        boundary_selector = groups == 0
        marking_score = 0.0
        if np.any(marking_selector):
            distances = white_distance[ys[marking_selector], xs[marking_selector]]
            marking_score = float(
                np.average(
                    np.exp(-0.5 * (distances / 6.5) ** 2),
                    weights=weights[marking_selector],
                )
            )
        boundary_score = 0.0
        if np.any(boundary_selector):
            distances = boundary_distance[ys[boundary_selector], xs[boundary_selector]]
            boundary_score = float(
                np.average(
                    np.exp(-0.5 * (distances / 6.5) ** 2),
                    weights=weights[boundary_selector],
                )
            )
        template_score = float(0.76 * marking_score + 0.24 * boundary_score)
        template_score *= 0.62 + 0.38 * min(1.0, visible_fraction / 0.22)

        quad_mask = np.zeros_like(mask)
        clipped = np.asarray(corners, dtype=np.float64).copy()
        clipped[:, 0] = np.clip(clipped[:, 0], -10 * width, 11 * width)
        clipped[:, 1] = np.clip(clipped[:, 1], -10 * height, 11 * height)
        cv2.fillConvexPoly(quad_mask, np.round(clipped).astype(np.int32), 255)
        mask_count = max(1, int(np.count_nonzero(mask)))
        quad_visible_count = max(1, int(np.count_nonzero(quad_mask)))
        intersection = int(np.count_nonzero((quad_mask > 0) & (mask > 0)))
        recall = intersection / mask_count
        precision = intersection / quad_visible_count
        mask_score = float(0.78 * recall + 0.22 * min(1.0, precision / 0.80))

        reverse_score = self._reverse_chamfer_score(
            projected,
            self.template_points.groups,
            white_mask,
            quad_mask,
        )

        diagonal = float(np.hypot(width, height))
        if near_anchor is not None and far_anchor is not None:
            anchors_field = _project_points(np.vstack([near_anchor, far_anchor]), image_to_field)
            if not np.isfinite(anchors_field).all():
                return None
            near_error = float(
                np.linalg.norm(
                    (anchors_field[0] - np.array([0.0, 0.5]))
                    / np.array([1.0, 0.75])
                )
            )
            far_error = float(
                np.linalg.norm(
                    (anchors_field[1] - np.array([1.0, 0.5]))
                    / np.array([1.0, 0.75])
                )
            )
            paired_goal_score = float(
                np.exp(-0.5 * ((near_error + far_error) / 0.20) ** 2)
            )
            goal_score = max(
                paired_goal_score,
                self._single_goal_score(field_to_image, goal_points, diagonal),
            )
        else:
            goal_score = self._single_goal_score(field_to_image, goal_points, diagonal)

        physical_boundary_score, physical_boundary_count, physical_boundary_scores = (
            self._directional_boundary_evidence(frame, mask, corners)
        )
        rail_alignment_score = 0.0
        if rail_pair is not None:
            projected_lines = [
                line_from_points(corners[0], corners[1]),
                line_from_points(corners[3], corners[2]),
            ]
            rail_scores: list[float] = []
            for projected_line, rail in zip(projected_lines, rail_pair):
                if projected_line is None:
                    continue
                sample = (
                    np.linspace(0.0, 1.0, 16)[:, None]
                    * (rail.segment[1] - rail.segment[0])
                    + rail.segment[0]
                )
                median_distance = float(
                    np.median(
                        np.abs(sample @ projected_line[:2] + projected_line[2])
                    )
                )
                rail_scores.append(float(np.exp(-median_distance / 8.0)))
            rail_alignment_score = float(np.mean(rail_scores)) if rail_scores else 0.0
        rail_score = float(
            physical_boundary_score
            * (0.72 + 0.28 * rail_alignment_score)
        )

        manual_reports = score_manual_anchors(
            semantic_segments, field_to_image, diagonal
        )
        manual_score = (
            float(np.mean([item.score for item in manual_reports]))
            if manual_reports
            else 0.0
        )
        hard_anchor_score = (
            float(min([item.score for item in manual_reports]))
            if manual_reports
            else 0.0
        )
        hard_anchor_count = sum(item.hard_pass for item in manual_reports)
        if manual_reports and (
            not all(item.hard_pass for item in manual_reports)
            or hard_anchor_score < 0.72
        ):
            return None

        feature_score, feature_count, feature_matches = self._automatic_feature_matches(
            field_to_image, marking_lines or [], diagonal
        )
        for name, seed_score in (seed_matches or {}).items():
            feature_matches[name] = max(
                float(feature_matches.get(name, 0.0)),
                float(np.clip(seed_score, 0.0, 0.86)),
            )
        feature_count = len(feature_matches)
        feature_score = (
            float(np.mean(list(feature_matches.values())))
            if feature_matches
            else 0.0
        )

        centerline = _project_points(
            np.float32([[0.0, 0.5], [1.0, 0.5]]), field_to_image
        )
        axis_length = float(np.linalg.norm(centerline[1] - centerline[0]))
        axis_score = float(
            np.clip(axis_length / max(1.0, 0.16 * diagonal), 0.0, 1.0)
        )

        if near_anchor is not None and far_anchor is not None and goal_score < 0.08:
            return None
        if semantic_segments:
            score = (
                0.21 * template_score
                + 0.20 * manual_score
                + 0.15 * reverse_score
                + 0.13 * feature_score
                + 0.12 * mask_score
                + 0.07 * goal_score
                + 0.06 * boundary_score
                + 0.04 * rail_score
                + 0.02 * axis_score
            )
        else:
            score = (
                0.24 * template_score
                + 0.18 * reverse_score
                + 0.16 * mask_score
                + 0.13 * feature_score
                + 0.10 * goal_score
                + 0.08 * boundary_score
                + 0.07 * rail_score
                + 0.04 * axis_score
            )
        return (
            float(score),
            template_score,
            mask_score,
            goal_score,
            rail_score,
            visible_fraction,
            manual_score,
            feature_score,
            feature_count,
            hard_anchor_score,
            hard_anchor_count,
            feature_matches,
            reverse_score,
            boundary_score,
            physical_boundary_score,
            physical_boundary_count,
            physical_boundary_scores,
        )

    @staticmethod
    def _vanishing_candidates(width: int, height: int, other_lines: list[RailLineCandidate]) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []
        # Intersections of observed line segments often directly propose the
        # transverse vanishing point. Keep both off-screen and distant points.
        for first, second in combinations(other_lines[:10], 2):
            point = intersect_lines(first.line, second.line)
            if point is not None and np.isfinite(point).all():
                candidates.append(point)

        x_values = np.array([-4.0, -2.0, -1.0, -0.25, 0.5, 1.25, 2.0, 3.0, 5.0]) * width
        y_values = np.array([-4.0, -2.0, -1.0, -0.25, 0.5, 1.25, 2.0, 3.0, 5.0]) * height
        for x in x_values:
            for y in y_values:
                if -0.2 * width <= x <= 1.2 * width and -0.2 * height <= y <= 1.2 * height:
                    continue
                candidates.append(np.array([x, y], dtype=np.float64))
        # De-duplicate coarse proposals.
        unique: list[np.ndarray] = []
        threshold = 0.12 * float(np.hypot(width, height))
        for point in candidates:
            if all(float(np.linalg.norm(point - prior)) > threshold for prior in unique):
                unique.append(point)
        return unique[:160]

    def register(
        self,
        frame: np.ndarray,
        field_mask: np.ndarray,
        goal_detections: Iterable[dict[str, Any]] | None,
        exclusion_boxes: Iterable[list[float]] | None = None,
        predicted_corners: np.ndarray | None = None,
        semantic_segments: dict[str, np.ndarray] | None = None,
        temporal_marking_mask: np.ndarray | None = None,
        temporal_evidence_frames: int = 0,
    ) -> TemplateRegistrationResult:
        """Register the visible field with dense, unlabeled and temporal cues.

        V9 does not require an observed line to be named before a homography is
        available. It generates canonical line-grid assignments, projects the
        complete template, and selects the assignment supported by paint,
        physical boundaries, goals, the surface mask and temporal continuity.
        """
        frame_work = (
            cv2.resize(
                frame,
                (self.work_width, self.work_height),
                interpolation=cv2.INTER_AREA,
            )
            if self.scale < 1.0
            else frame
        )
        mask_work = self._clean_mask(field_mask, self.work_width, self.work_height)
        if np.count_nonzero(mask_work) < 0.035 * mask_work.size:
            return TemplateRegistrationResult(
                False,
                False,
                0.0,
                None,
                None,
                None,
                "mascara_insuficiente",
                temporal_evidence_frames=int(temporal_evidence_frames),
            )

        scaled_boxes = None
        if exclusion_boxes:
            scaled_boxes = [
                [float(value) * self.scale for value in box]
                for box in exclusion_boxes
            ]
        evidence = self.evidence_extractor.extract(
            frame_work, mask_work, scaled_boxes
        )
        white = evidence.marking_mask.copy()
        if temporal_marking_mask is not None:
            prior = (np.asarray(temporal_marking_mask) > 0).astype(np.uint8) * 255
            if prior.shape != white.shape:
                prior = cv2.resize(
                    prior,
                    (self.work_width, self.work_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            prior = cv2.bitwise_and(prior, evidence.search_region)
            prior = cv2.morphologyEx(
                prior, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            white = cv2.max(white, prior)

        white_distance = self._distance_map(white)
        boundary_combined = cv2.max(evidence.boundary_mask, evidence.rail_mask)
        boundary_distance = self._distance_map(boundary_combined)

        evidence_sides = [
            self._as_rail_candidate(item) for item in evidence.boundary_lines
        ]
        evidence_markings = [
            self._as_rail_candidate(item) for item in evidence.marking_lines
        ]
        # Keep V8's Hough proposals as a complementary fallback. The adaptive
        # evidence is primary, but Hough can bridge a broken long line.
        legacy_sides, legacy_markings = self._extract_projective_lines(
            frame_work, mask_work, white
        )
        side_candidates = self._deduplicate_lines(
            [*evidence_sides, *legacy_sides], maximum=28
        )
        marking_lines = self._deduplicate_lines(
            [*evidence_markings, *legacy_markings], maximum=42
        )

        manual_segments_work: dict[str, np.ndarray] = {}
        for name, segment in (semantic_segments or {}).items():
            points = np.asarray(segment, dtype=np.float64).reshape(2, 2) * self.scale
            if float(np.linalg.norm(points[1] - points[0])) >= 5.0:
                manual_segments_work[name] = points

        scaled_goals: list[dict[str, Any]] = []
        for detection in goal_detections or []:
            copied = dict(detection)
            box = copied.get("bbox_xyxy", [])
            if len(box) == 4:
                copied["bbox_xyxy"] = [float(value) * self.scale for value in box]
            scaled_goals.append(copied)
        goal_points = self._goal_mouth_points(scaled_goals, mask_work)

        anchor_candidates = goal_mouth_anchor_candidates(goal_detections)
        scaled_anchor_candidates = [
            (near * self.scale, far * self.scale, label)
            for near, far, label in anchor_candidates
            if label != "rayo_hacia_campo"
        ]
        if not scaled_anchor_candidates:
            scaled_anchor_candidates = [
                (near * self.scale, far * self.scale, label)
                for near, far, label in anchor_candidates
            ]

        candidate_bank: list[
            tuple[
                float,
                np.ndarray,
                tuple[Any, ...],
                tuple[RailLineCandidate, RailLineCandidate] | None,
                np.ndarray | None,
                np.ndarray | None,
                str,
                dict[str, float],
            ]
        ] = []
        diagonal = float(np.hypot(self.work_width, self.work_height))
        predicted_work = (
            None
            if predicted_corners is None
            else np.asarray(predicted_corners, dtype=np.float64) * self.scale
        )

        def consider(
            corners: np.ndarray,
            source_label: str,
            near_anchor: np.ndarray | None = None,
            far_anchor: np.ndarray | None = None,
            rail_pair: tuple[RailLineCandidate, RailLineCandidate] | None = None,
            seed_matches: dict[str, float] | None = None,
        ) -> None:
            corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            # Canonical x starts at the camera-near goal. Under a planar
            # perspective its end line is normally wider in the image. A
            # one-goal clip is otherwise exactly ambiguous under x reflection.
            near_width = float(np.linalg.norm(corners[3] - corners[0]))
            far_width = float(np.linalg.norm(corners[2] - corners[1]))
            if near_width + 1e-6 < far_width:
                corners = corners[[1, 0, 3, 2]]

            near_center = 0.5 * (corners[0] + corners[3])
            far_center = 0.5 * (corners[1] + corners[2])
            field_axis = far_center - near_center
            left_vector = corners[0] - near_center
            handedness = float(
                field_axis[0] * left_vector[1] - field_axis[1] * left_vector[0]
            )
            if handedness > 0.0:
                corners = corners[[3, 2, 1, 0]]
            scored = self._candidate_score(
                corners,
                frame_work,
                mask_work,
                white,
                white_distance,
                boundary_distance,
                goal_points,
                near_anchor,
                far_anchor,
                rail_pair,
                semantic_segments=manual_segments_work,
                marking_lines=marking_lines,
                seed_matches=seed_matches,
            )
            if scored is None:
                return
            total = float(scored[0])
            if predicted_work is not None:
                temporal_difference = float(
                    np.mean(
                        np.linalg.norm(
                            np.asarray(corners, dtype=np.float64) - predicted_work,
                            axis=1,
                        )
                    )
                )
                temporal_score = float(
                    np.exp(
                        -temporal_difference / max(1.0, 0.14 * diagonal)
                    )
                )
                total = 0.92 * total + 0.08 * temporal_score
            components = (total, *scored[1:])
            candidate_bank.append(
                (
                    total,
                    np.asarray(corners, dtype=np.float32),
                    components,
                    rail_pair,
                    near_anchor,
                    far_anchor,
                    source_label,
                    dict(seed_matches or {}),
                )
            )

        # A propagated global pose is a useful candidate, never an automatic
        # acceptance. Current dense evidence must revalidate it.
        if predicted_work is not None:
            consider(predicted_work, "propagada_revalidada")

        # Exact/manual 2x2 seed when enough semantic lines were provided.
        manual_lines: dict[str, np.ndarray] = {}
        for name, segment in manual_segments_work.items():
            line = line_from_points(segment[0], segment[1])
            if line is not None:
                manual_lines[name] = line
        if manual_lines:
            from src.I_field_geometry.feature_constraints import homography_from_semantic_lines
            solved = homography_from_semantic_lines(manual_lines)
            if solved is not None:
                _image_to_field, field_to_image = solved
                corners = _project_points(
                    np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    field_to_image,
                )
                consider(
                    corners,
                    "anclas_manuales_exactas",
                    seed_matches={name: 1.0 for name in manual_lines},
                )

        # V9 core: assign identities within hypotheses, not before pose search.
        grid_candidates = self._unlabeled_grid_candidates(
            mask_work,
            white_distance,
            boundary_distance,
            side_candidates,
            marking_lines,
            manual_segments_work,
            goal_points,
            maximum=20,
        )
        for grid_index, (corners, seed_matches, label) in enumerate(grid_candidates):
            consider(corners, label, seed_matches=seed_matches)

        # Retain the two-goal projective solver when both goals are available.
        # Geometry can generate tens of thousands of combinations, so V9 uses
        # a sparse chamfer pre-score and fully evaluates only the strongest set.
        if scaled_anchor_candidates and len(side_candidates) >= 2:
            vanishing_candidates = self._marking_vanishing_candidates(
                self.work_width, self.work_height, marking_lines
            )
            goal_heap: list[
                tuple[
                    float,
                    int,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    tuple[RailLineCandidate, RailLineCandidate],
                    str,
                ]
            ] = []
            goal_counter = 0
            ranked_side_pairs = self._rank_line_pairs(
                side_candidates[:18], diagonal, maximum=28
            )
            for near_anchor, far_anchor, anchor_label in scaled_anchor_candidates:
                for first, second, _pair_quality in ranked_side_pairs:
                    near_product = _signed_distance(
                        first.line, near_anchor
                    ) * _signed_distance(second.line, near_anchor)
                    far_product = _signed_distance(
                        first.line, far_anchor
                    ) * _signed_distance(second.line, far_anchor)
                    if near_product > 0.0 or far_product > 0.0:
                        continue
                    for vanishing in vanishing_candidates[:70]:
                        near_end = line_from_points(vanishing, near_anchor)
                        far_end = line_from_points(vanishing, far_anchor)
                        if near_end is None or far_end is None:
                            continue
                        for rail_pair in ((first, second), (second, first)):
                            c00 = intersect_lines(near_end, rail_pair[0].line)
                            c01 = intersect_lines(far_end, rail_pair[0].line)
                            c11 = intersect_lines(far_end, rail_pair[1].line)
                            c10 = intersect_lines(near_end, rail_pair[1].line)
                            if any(
                                point is None
                                for point in (c00, c01, c11, c10)
                            ):
                                continue
                            corners = np.float32([c00, c01, c11, c10])
                            cheap = self._cheap_candidate_score(
                                corners,
                                mask_work,
                                white_distance,
                                boundary_distance,
                                goal_points,
                            )
                            if cheap < 0.36:
                                continue
                            item = (
                                float(cheap),
                                goal_counter,
                                corners,
                                near_anchor,
                                far_anchor,
                                rail_pair,
                                anchor_label,
                            )
                            goal_counter += 1
                            if len(goal_heap) < 42:
                                heapq.heappush(goal_heap, item)
                            elif cheap > goal_heap[0][0]:
                                heapq.heapreplace(goal_heap, item)
            for (
                _cheap,
                _counter,
                corners,
                near_anchor,
                far_anchor,
                rail_pair,
                anchor_label,
            ) in sorted(goal_heap, key=lambda item: item[0], reverse=True):
                consider(
                    corners,
                    f"dos_porterias_{anchor_label}",
                    near_anchor,
                    far_anchor,
                    rail_pair,
                )

        if not candidate_bank:
            local_work, local_source, local_count = local_rectification_from_segments(
                manual_segments_work,
                [item.segment for item in marking_lines[:28]],
            )
            local_full = None if local_work is None else local_work @ self.to_work
            source = (
                "anclas_parciales_sin_registro_global_v10"
                if manual_segments_work
                else "orientacion_local_sin_registro_global_v10"
            )
            self.last_debug = {
                "marking_mask": white.copy(),
                "boundary_mask": boundary_combined.copy(),
                "marking_lines": [item.segment.copy() for item in marking_lines[:24]],
                "side_lines": [item.segment.copy() for item in side_candidates[:16]],
                "candidate_corners": [],
                "source": source,
            }
            return TemplateRegistrationResult(
                valid=False,
                trusted=False,
                confidence=0.0,
                corners_image=None,
                homography_image_to_field_normalized=None,
                homography_field_to_image_normalized=None,
                source=source,
                rail_lines=[self.to_work.T @ item.line for item in side_candidates[:8]],
                registration_scope="local" if local_full is not None else "surface",
                geometry_state="local" if local_full is not None else "surface",
                local_homography_image_to_local=local_full,
                hard_anchor_count=len(manual_segments_work),
                feature_match_count=local_count,
                temporal_evidence_frames=int(temporal_evidence_frames),
                marking_pixel_fraction=float(evidence.marking_pixel_fraction),
                candidate_count=0,
            )

        candidate_bank.sort(key=lambda item: item[0], reverse=True)
        best = candidate_bank[0]

        # Dense coordinate-descent refinement of the winning quadrilateral.
        refined = best
        refinement_label = f"{best[6]}_refinada"
        for step in (4.0, 2.0):
            improved = True
            rounds = 0
            while improved and rounds < 1:
                improved = False
                rounds += 1
                base_corners = refined[1].copy()
                for corner_index in range(4):
                    for axis in range(2):
                        for direction in (-1.0, 1.0):
                            proposal = base_corners.copy()
                            proposal[corner_index, axis] += direction * step
                            previous_count = len(candidate_bank)
                            consider(
                                proposal,
                                refinement_label,
                                refined[4],
                                refined[5],
                                refined[3],
                                refined[7],
                            )
                            if len(candidate_bank) > previous_count:
                                candidate = candidate_bank[-1]
                                if candidate[0] > refined[0] + 1e-5:
                                    refined = candidate
                                    base_corners = candidate[1].copy()
                                    improved = True
        candidate_bank.append(refined)
        candidate_bank.sort(key=lambda item: item[0], reverse=True)
        best = candidate_bank[0]

        # Ambiguity is measured against a geometrically distinct runner-up.
        second_score = None
        for candidate in candidate_bank[1:]:
            difference = float(
                np.mean(np.linalg.norm(candidate[1] - best[1], axis=1))
            )
            if difference > 0.025 * diagonal:
                second_score = float(candidate[0])
                break
        candidate_margin = float(
            best[0] - second_score if second_score is not None else 0.12
        )

        (
            score,
            corners_work,
            components,
            rail_pair,
            near_anchor,
            far_anchor,
            source_label,
            _seed_matches,
        ) = best
        (
            _,
            template_score,
            mask_score,
            goal_score,
            rail_score,
            visible_fraction,
            manual_score,
            feature_score,
            feature_count,
            hard_anchor_score,
            hard_anchor_count,
            feature_matches,
            reverse_score,
            boundary_score,
            physical_boundary_score,
            physical_boundary_count,
            physical_boundary_scores,
        ) = components

        corners_full = corners_work / self.scale
        field_to_image = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            corners_full.astype(np.float32),
        )
        image_to_field = _safe_inverse(field_to_image)
        if image_to_field is None:
            return TemplateRegistrationResult(
                False,
                False,
                0.0,
                None,
                None,
                None,
                "inversion_fallida_v10",
            )

        matched_names = set(feature_matches or {}) | set(manual_segments_work)
        transverse_count, longitudinal_count = semantic_family_counts(matched_names)
        structural_support = transverse_count >= 2 and longitudinal_count >= 2
        manual_hard_ok = bool(
            manual_segments_work
            and int(hard_anchor_count) == len(manual_segments_work)
            and hard_anchor_score >= 0.72
        )
        temporal_support = bool(predicted_work is not None)
        ambiguity_threshold = (
            0.005
            if scaled_anchor_candidates
            else (
                0.007
                if manual_hard_ok
                else (0.014 if predicted_work is not None else 0.024)
            )
        )
        ambiguity_ok = bool(
            candidate_margin >= ambiguity_threshold
            or predicted_work is not None
        )
        # Temporal consensus may refine a currently visible marking, but it may
        # not create global trust by itself.  A fresh frame must still contain
        # enough paint evidence to prevent self-confirmation loops.
        evidence_pixels_ok = bool(evidence.marking_pixel_fraction >= 0.0012)

        manual_global_trust = bool(
            manual_hard_ok
            and structural_support
            and manual_score >= 0.72
            and feature_score >= 0.54
            and template_score >= 0.30
            and reverse_score >= 0.20
            and mask_score >= 0.56
            and boundary_score >= 0.18
            and score >= 0.49
            and ambiguity_ok
            and evidence_pixels_ok
        )
        automatic_trust = bool(
            not manual_segments_work
            and structural_support
            and int(feature_count) >= 4
            and feature_score >= 0.60
            and template_score >= 0.37
            and reverse_score >= 0.26
            and mask_score >= 0.62
            and boundary_score >= 0.46
            and physical_boundary_count >= 1
            and physical_boundary_score >= 0.55
            and score >= 0.49
            and ambiguity_ok
            and evidence_pixels_ok
            and (
                physical_boundary_count >= 2
                or goal_score >= 0.42
                or (feature_count >= 5 and reverse_score >= 0.31)
                or temporal_support
            )
        )
        trusted = manual_global_trust or automatic_trust
        confidence = float(
            np.clip(
                0.62 * ((score - 0.38) / 0.30)
                + 0.20 * np.clip(candidate_margin / 0.05, 0.0, 1.0)
                + 0.10 * np.clip(temporal_evidence_frames / 8.0, 0.0, 1.0)
                + 0.08 * goal_score,
                0.0,
                1.0,
            )
        )

        self.last_debug = {
            "marking_mask": white.copy(),
            "boundary_mask": boundary_combined.copy(),
            "marking_lines": [item.segment.copy() for item in marking_lines[:24]],
            "side_lines": [item.segment.copy() for item in side_candidates[:16]],
            "candidate_corners": [item[1].copy() for item in candidate_bank[:8]],
            "best_corners": corners_work.copy(),
            "source": source_label,
            "candidate_margin": candidate_margin,
            "candidate_count": len(candidate_bank),
            "physical_boundary_scores": dict(physical_boundary_scores or {}),
            "physical_boundary_count": int(physical_boundary_count),
            "physical_boundary_score": float(physical_boundary_score),
        }

        if not trusted:
            local_work, local_source, local_count = local_rectification_from_segments(
                manual_segments_work,
                [item.segment for item in marking_lines[:28]],
            )
            local_full = None if local_work is None else local_work @ self.to_work
            return TemplateRegistrationResult(
                valid=False,
                trusted=False,
                confidence=min(confidence, 0.49),
                corners_image=None,
                homography_image_to_field_normalized=None,
                homography_field_to_image_normalized=None,
                source=f"registro_local_v10_{local_source}_{source_label}",
                template_score=float(template_score),
                mask_score=float(mask_score),
                goal_score=float(goal_score),
                rail_score=float(rail_score),
                visible_template_fraction=float(visible_fraction),
                rail_lines=[self.to_work.T @ item.line for item in side_candidates[:8]],
                manual_line_score=float(manual_score),
                registration_scope="local" if local_full is not None else "surface",
                geometry_state="local" if local_full is not None else "surface",
                local_homography_image_to_local=local_full,
                hard_anchor_score=float(hard_anchor_score),
                hard_anchor_count=int(hard_anchor_count),
                feature_match_score=float(feature_score),
                feature_match_count=int(feature_count),
                feature_matches=dict(feature_matches or {}),
                reverse_template_score=float(reverse_score),
                boundary_alignment_score=float(boundary_score),
                candidate_margin=float(candidate_margin),
                temporal_evidence_frames=int(temporal_evidence_frames),
                marking_pixel_fraction=float(evidence.marking_pixel_fraction),
                candidate_count=len(candidate_bank),
                physical_boundary_score=float(physical_boundary_score),
                physical_boundary_count=int(physical_boundary_count),
                physical_boundary_scores=dict(physical_boundary_scores or {}),
            )

        scope = "partial" if visible_fraction < 0.48 else "full"
        source_prefix = (
            "anclas_duras_v10" if manual_segments_work else "plantilla_semantica_v10"
        )
        rails_full = None
        if rail_pair is not None:
            rails_full = [self.to_work.T @ item.line for item in rail_pair]
        goal_anchors_full = None
        if near_anchor is not None and far_anchor is not None:
            goal_anchors_full = [near_anchor / self.scale, far_anchor / self.scale]
        return TemplateRegistrationResult(
            valid=True,
            trusted=True,
            confidence=confidence,
            corners_image=corners_full.astype(np.float32),
            homography_image_to_field_normalized=image_to_field,
            homography_field_to_image_normalized=field_to_image,
            source=f"{source_prefix}_{source_label}",
            template_score=float(template_score),
            mask_score=float(mask_score),
            goal_score=float(goal_score),
            rail_score=float(rail_score),
            visible_template_fraction=float(visible_fraction),
            rail_lines=rails_full,
            goal_anchors=goal_anchors_full,
            manual_line_score=float(manual_score),
            registration_scope=scope,
            geometry_state="global",
            hard_anchor_score=float(hard_anchor_score),
            hard_anchor_count=int(hard_anchor_count),
            feature_match_score=float(feature_score),
            feature_match_count=int(feature_count),
            feature_matches=dict(feature_matches or {}),
            reverse_template_score=float(reverse_score),
            boundary_alignment_score=float(boundary_score),
            candidate_margin=float(candidate_margin),
            temporal_evidence_frames=int(temporal_evidence_frames),
            marking_pixel_fraction=float(evidence.marking_pixel_fraction),
            candidate_count=len(candidate_bank),
            physical_boundary_score=float(physical_boundary_score),
            physical_boundary_count=int(physical_boundary_count),
            physical_boundary_scores=dict(physical_boundary_scores or {}),
        )
