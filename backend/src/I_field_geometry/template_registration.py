from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Any, Iterable

import cv2
import numpy as np

from src.I_field_geometry.feature_constraints import (
    CANONICAL_LINES_NORMALIZED,
    CANONICAL_SEGMENTS_NORMALIZED,
    score_feature_anchor,
    score_manual_anchors,
    local_rectification_from_segments,
    semantic_family_counts,
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

    @staticmethod
    def _automatic_feature_matches(
        field_to_image: np.ndarray,
        marking_lines: list[RailLineCandidate],
        diagonal: float,
    ) -> tuple[float, int, dict[str, float]]:
        """Match canonical straight features to distinct observed segments.

        V8 does not reward a generic cloud of nearby white pixels as semantic
        evidence. A feature match must agree in line position, image direction,
        canonical coordinate and visible span.
        """
        proposals: list[tuple[float, str, int]] = []
        raw_scores: dict[tuple[str, int], float] = {}
        for name in CANONICAL_SEGMENTS_NORMALIZED:
            for index, candidate in enumerate(marking_lines[:34]):
                report = score_feature_anchor(name, candidate.segment, field_to_image, diagonal)
                relaxed_pass = bool(
                    report.score >= 0.64
                    and report.perpendicular_error_px <= max(9.0, 0.016 * diagonal)
                    and report.angle_error_deg <= 12.0
                    and report.canonical_error <= 0.075
                    and report.along_span >= 0.025
                )
                if relaxed_pass:
                    proposals.append((report.score, name, index))
                    raw_scores[(name, index)] = report.score
        proposals.sort(reverse=True)
        used_features: set[str] = set()
        used_segments: set[int] = set()
        matches: dict[str, float] = {}
        for score, name, index in proposals:
            if name in used_features or index in used_segments:
                continue
            matches[name] = float(score)
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
        mask: np.ndarray,
        white_distance: np.ndarray,
        near_anchor: np.ndarray | None,
        far_anchor: np.ndarray | None,
        rail_pair: tuple[RailLineCandidate, RailLineCandidate] | None,
        semantic_segments: dict[str, np.ndarray] | None = None,
        marking_lines: list[RailLineCandidate] | None = None,
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
        inside = finite & (projected[:, 0] >= 1) & (projected[:, 0] < width - 1) & (projected[:, 1] >= 1) & (projected[:, 1] < height - 1)
        visible_fraction = float(np.sum(self.template_points.weights[inside]) / max(1e-6, np.sum(self.template_points.weights)))
        if np.count_nonzero(inside) < 24 or visible_fraction < 0.020:
            return None
        points = projected[inside]
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
        distances = white_distance[ys, xs]
        weights = self.template_points.weights[inside]
        template_score = float(np.average(np.exp(-0.5 * (distances / 6.5) ** 2), weights=weights))
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

        if near_anchor is not None and far_anchor is not None:
            anchors_field = _project_points(np.vstack([near_anchor, far_anchor]), image_to_field)
            if not np.isfinite(anchors_field).all():
                return None
            near_error = float(np.linalg.norm((anchors_field[0] - np.array([0.0, 0.5])) / np.array([1.0, 0.75])))
            far_error = float(np.linalg.norm((anchors_field[1] - np.array([1.0, 0.5])) / np.array([1.0, 0.75])))
            goal_score = float(np.exp(-0.5 * ((near_error + far_error) / 0.20) ** 2))
        else:
            goal_score = 0.50

        rail_score = 0.50
        if rail_pair is not None:
            projected_lines = [line_from_points(corners[0], corners[1]), line_from_points(corners[3], corners[2])]
            rail_scores: list[float] = []
            for projected_line, rail in zip(projected_lines, rail_pair):
                if projected_line is None:
                    continue
                sample = np.linspace(0.0, 1.0, 16)[:, None] * (rail.segment[1] - rail.segment[0]) + rail.segment[0]
                median_distance = float(np.median(np.abs(sample @ projected_line[:2] + projected_line[2])))
                rail_scores.append(float(np.exp(-median_distance / 8.0)))
            rail_score = float(np.mean(rail_scores)) if rail_scores else 0.0

        diagonal = float(np.hypot(width, height))
        manual_reports = score_manual_anchors(semantic_segments, field_to_image, diagonal)
        manual_score = float(np.mean([item.score for item in manual_reports])) if manual_reports else 0.0
        hard_anchor_score = float(min([item.score for item in manual_reports])) if manual_reports else 0.0
        hard_anchor_count = sum(item.hard_pass for item in manual_reports)
        # Manual labels are hard constraints. One mismatched label invalidates
        # the whole global hypothesis; averaging is explicitly forbidden.
        if manual_reports and (
            not all(item.hard_pass for item in manual_reports)
            or hard_anchor_score < 0.72
        ):
            return None

        feature_score, feature_count, feature_matches = self._automatic_feature_matches(
            field_to_image, marking_lines or [], diagonal
        )
        centerline = _project_points(np.float32([[0.0, 0.5], [1.0, 0.5]]), field_to_image)
        axis_length = float(np.linalg.norm(centerline[1] - centerline[0]))
        axis_score = float(np.clip(axis_length / max(1.0, 0.16 * diagonal), 0.0, 1.0))

        if near_anchor is not None and far_anchor is not None and goal_score < 0.08:
            return None
        if semantic_segments:
            score = (
                0.24 * template_score + 0.31 * manual_score + 0.16 * feature_score
                + 0.13 * mask_score + 0.08 * goal_score + 0.05 * rail_score + 0.03 * axis_score
            )
        else:
            score = (
                0.24 * template_score + 0.26 * feature_score + 0.22 * goal_score
                + 0.12 * rail_score + 0.12 * mask_score + 0.04 * axis_score
            )
        return (
            float(score), template_score, mask_score, goal_score, rail_score,
            visible_fraction, manual_score, feature_score, feature_count,
            hard_anchor_score, hard_anchor_count, feature_matches,
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
    ) -> TemplateRegistrationResult:
        """Estimate a field-to-image mapping from whatever evidence is visible.

        V8 deliberately separates *local registration* from *full field
        visibility*. Missing corners are allowed to remain outside the crop;
        manual semantic segments are optional constraints, not a demand for four
        invented borders.
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
                False, False, 0.0, None, None, None, "mascara_insuficiente"
            )

        scaled_boxes = None
        if exclusion_boxes:
            scaled_boxes = [
                [float(value) * self.scale for value in box]
                for box in exclusion_boxes
            ]
        white = self._white_mask(frame_work, mask_work, scaled_boxes)
        white_distance = self._distance_map(white)
        side_candidates, marking_lines = self._extract_projective_lines(
            frame_work, mask_work, white
        )

        manual_segments_work: dict[str, np.ndarray] = {}
        for name, segment in (semantic_segments or {}).items():
            points = np.asarray(segment, dtype=np.float64).reshape(2, 2) * self.scale
            if float(np.linalg.norm(points[1] - points[0])) >= 5.0:
                manual_segments_work[name] = points

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

        # Each candidate is: score, corners_work, components, rail_pair,
        # near_anchor, far_anchor, source_label.
        best: tuple[
            float, np.ndarray, tuple[Any, ...],
            tuple[RailLineCandidate, RailLineCandidate] | None,
            np.ndarray | None, np.ndarray | None, str,
        ] | None = None
        diagonal = float(np.hypot(self.work_width, self.work_height))

        def consider(
            corners: np.ndarray,
            source_label: str,
            near_anchor: np.ndarray | None = None,
            far_anchor: np.ndarray | None = None,
            rail_pair: tuple[RailLineCandidate, RailLineCandidate] | None = None,
        ) -> None:
            nonlocal best
            scored = self._candidate_score(
                corners,
                mask_work,
                white_distance,
                near_anchor,
                far_anchor,
                rail_pair,
                semantic_segments=manual_segments_work,
                marking_lines=marking_lines,
            )
            if scored is None:
                return
            total = float(scored[0])
            if predicted_corners is not None:
                predicted_work = np.asarray(predicted_corners, dtype=np.float64) * self.scale
                temporal_difference = float(
                    np.mean(np.linalg.norm(corners - predicted_work, axis=1))
                )
                temporal_score = float(
                    np.exp(-temporal_difference / max(1.0, 0.18 * diagonal))
                )
                total = 0.95 * total + 0.05 * temporal_score
            if best is None or total > best[0]:
                components = (total, *scored[1:])
                best = (
                    total,
                    np.asarray(corners, dtype=np.float32),
                    components,
                    rail_pair,
                    near_anchor,
                    far_anchor,
                    source_label,
                )

        # 1) A previous locally valid registration is always a legitimate
        # candidate. Manual lines and current white marks decide whether it is
        # still aligned; it is not blindly trusted.
        if predicted_corners is not None:
            consider(
                np.asarray(predicted_corners, dtype=np.float64) * self.scale,
                "propagada_revalidada",
            )

        # 2) Flexible manual boundary constraints. With four lines this is an
        # exact line-homography seed. With two or three lines, only visible
        # missing boundaries are filled from detected rails/paint; if a missing
        # side is not visible, no rectangle is fabricated.
        manual_boundary_lines: dict[str, np.ndarray] = {}
        for name in ("near", "far", "left", "right"):
            segment = manual_segments_work.get(name)
            if segment is not None:
                line = line_from_points(segment[0], segment[1])
                if line is not None:
                    manual_boundary_lines[name] = line

        detected_pool = self._deduplicate_lines(
            [*side_candidates[:18], *marking_lines[:18]], maximum=20
        )
        if predicted_corners is not None:
            p = np.asarray(predicted_corners, dtype=np.float64) * self.scale
            predicted_map = {
                "near": line_from_points(p[0], p[3]),
                "far": line_from_points(p[1], p[2]),
                "left": line_from_points(p[0], p[1]),
                "right": line_from_points(p[3], p[2]),
            }
        else:
            predicted_map = {}

        missing = [
            name for name in ("near", "far", "left", "right")
            if name not in manual_boundary_lines
        ]
        if len(manual_boundary_lines) >= 2 and len(missing) <= 2:
            pool_items: list[tuple[np.ndarray, int]] = [
                (candidate.line, index) for index, candidate in enumerate(detected_pool[:16])
            ]
            # A propagated missing line may be used, but never an image margin.
            for name in missing:
                line = predicted_map.get(name)
                if line is not None:
                    pool_items.append((line, 10_000 + len(pool_items)))

            assignments = [()] if not missing else permutations(pool_items, len(missing))
            tested = 0
            for assignment in assignments:
                if len({item[1] for item in assignment}) != len(assignment):
                    continue
                lines = dict(manual_boundary_lines)
                for name, (line, _identifier) in zip(missing, assignment):
                    lines[name] = line
                try:
                    near_left = intersect_lines(lines["near"], lines["left"])
                    far_left = intersect_lines(lines["far"], lines["left"])
                    far_right = intersect_lines(lines["far"], lines["right"])
                    near_right = intersect_lines(lines["near"], lines["right"])
                except Exception:
                    continue
                if any(point is None for point in (near_left, far_left, far_right, near_right)):
                    continue
                corners = np.float32([near_left, far_left, far_right, near_right])
                consider(corners, "lineas_semanticas_parciales")
                tested += 1
                if tested >= 420:
                    break

        # 3) Fully automatic goal-anchored multiline solver retained from V6,
        # now evaluated with optional manual line evidence and partial-view
        # scoring. It only runs when two goal anchors are actually available.
        if scaled_anchor_candidates and len(side_candidates) >= 2:
            vanishing_candidates = self._marking_vanishing_candidates(
                self.work_width, self.work_height, marking_lines
            )
            anchor_priority = {"bordes_internos": 1.0, "centros": 0.97}
            for near_anchor, far_anchor, anchor_label in scaled_anchor_candidates:
                for first, second in combinations(side_candidates[:22], 2):
                    midpoint_first = np.mean(first.segment, axis=0)
                    midpoint_second = np.mean(second.segment, axis=0)
                    midpoint_distance = abs(_signed_distance(first.line, midpoint_second))
                    if midpoint_distance < 0.052 * diagonal:
                        continue
                    near_product = _signed_distance(first.line, near_anchor) * _signed_distance(second.line, near_anchor)
                    far_product = _signed_distance(first.line, far_anchor) * _signed_distance(second.line, far_anchor)
                    if near_product > 0.0 or far_product > 0.0:
                        continue
                    near_span = abs(_signed_distance(first.line, near_anchor)) + abs(_signed_distance(second.line, near_anchor))
                    far_span = abs(_signed_distance(first.line, far_anchor)) + abs(_signed_distance(second.line, far_anchor))
                    if min(near_span, far_span) < 0.075 * diagonal:
                        continue
                    for vanishing in vanishing_candidates:
                        near_end = line_from_points(vanishing, near_anchor)
                        far_end = line_from_points(vanishing, far_anchor)
                        if near_end is None or far_end is None:
                            continue
                        for rail_pair in ((first, second), (second, first)):
                            c00 = intersect_lines(near_end, rail_pair[0].line)
                            c01 = intersect_lines(far_end, rail_pair[0].line)
                            c11 = intersect_lines(far_end, rail_pair[1].line)
                            c10 = intersect_lines(near_end, rail_pair[1].line)
                            if any(point is None for point in (c00, c01, c11, c10)):
                                continue
                            corners = np.float32([c00, c01, c11, c10])
                            field_axis = far_anchor - near_anchor
                            lateral_vector = c00 - near_anchor
                            cross_value = float(field_axis[0] * lateral_vector[1] - field_axis[1] * lateral_vector[0])
                            if cross_value >= 0.0:
                                continue
                            consider(
                                corners,
                                anchor_label,
                                near_anchor,
                                far_anchor,
                                rail_pair,
                            )

        if best is None:
            local_work, local_source, local_count = local_rectification_from_segments(
                manual_segments_work,
                [item.segment for item in marking_lines[:24]],
            )
            local_full = None if local_work is None else local_work @ self.to_work
            source = "anclas_parciales_sin_registro_global" if manual_segments_work else "orientacion_local_sin_registro_global"
            return TemplateRegistrationResult(
                valid=False,
                trusted=False,
                confidence=0.0,
                corners_image=None,
                homography_image_to_field_normalized=None,
                homography_field_to_image_normalized=None,
                source=source,
                rail_lines=[self.to_work.T @ item.line for item in side_candidates[:8]],
                manual_line_score=0.0,
                registration_scope="local" if local_full is not None else "surface",
                geometry_state="local" if local_full is not None else "surface",
                local_homography_image_to_local=local_full,
                hard_anchor_count=len(manual_segments_work),
                feature_match_count=local_count,
            )

        score, corners_work, components, rail_pair, near_anchor, far_anchor, source_label = best
        (
            _, template_score, mask_score, goal_score, rail_score,
            visible_fraction, manual_score, feature_score, feature_count,
            hard_anchor_score, hard_anchor_count, feature_matches,
        ) = components
        corners_full = corners_work / self.scale
        field_to_image = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            corners_full.astype(np.float32),
        )
        image_to_field = _safe_inverse(field_to_image)
        if image_to_field is None:
            return TemplateRegistrationResult(
                False, False, 0.0, None, None, None, "inversion_fallida"
            )

        has_goals = near_anchor is not None and far_anchor is not None
        matched_names = set(feature_matches or {}) | set(manual_segments_work)
        transverse_count, longitudinal_count = semantic_family_counts(matched_names)
        grid_support = transverse_count >= 2 and longitudinal_count >= 2
        goal_supported_grid = bool(
            has_goals
            and transverse_count >= 1
            and longitudinal_count >= 1
            and (int(feature_count) + int(hard_anchor_count)) >= 3
        )
        structural_support = grid_support or goal_supported_grid

        manual_hard_ok = bool(
            manual_segments_work
            and int(hard_anchor_count) == len(manual_segments_work)
            and hard_anchor_score >= 0.72
        )
        manual_global_trust = bool(
            manual_hard_ok
            and structural_support
            and manual_score >= 0.82
            and feature_score >= 0.58
            and mask_score >= 0.58
            and score >= 0.50
        )
        automatic_trust = bool(
            not manual_segments_work
            and structural_support
            and int(feature_count) >= (2 if has_goals else 4)
            and feature_score >= 0.70
            and template_score >= 0.30
            and mask_score >= 0.67
            and (not has_goals or goal_score >= 0.62)
            and score >= 0.57
        )
        trusted = manual_global_trust or automatic_trust
        confidence = float(np.clip((score - 0.34) / 0.42, 0.0, 1.0))

        # V8 never exposes a merely plausible full homography. If semantic
        # feature identity is insufficient, return only local orientation.
        if not trusted:
            local_work, local_source, local_count = local_rectification_from_segments(
                manual_segments_work,
                [item.segment for item in marking_lines[:24]],
            )
            local_full = None if local_work is None else local_work @ self.to_work
            return TemplateRegistrationResult(
                valid=False,
                trusted=False,
                confidence=min(confidence, 0.49),
                corners_image=None,
                homography_image_to_field_normalized=None,
                homography_field_to_image_normalized=None,
                source=f"registro_local_{local_source}_{source_label}",
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
            )

        scope = "partial" if visible_fraction < 0.48 else "full"
        source_prefix = "anclas_duras" if manual_segments_work else "plantilla_semantica"
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
        )
