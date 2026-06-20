"""Temporal tracking for up to four robots and one orange ball.

The detector works independently on every frame. This module adds track
confirmation, motion prediction, exact global assignment, appearance cues and
short occlusion handling. Unconfirmed detections are never exported, which
prevents a single badge or shirt from becoming a robot in the replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import hypot, log
from typing import Any

import cv2
import numpy as np

from src.C_quick_view.field_selector import select_main_field

from src.C_quick_view.ball_recovery import AdaptiveBallRecovery


ROBOT_NAMES = {"robot", "robots"}
BALL_NAMES = {"orange ball", "ball", "pelota", "balon", "balón"}
FIELD_NAMES = {"field", "playing field", "cancha", "campo"}
GOAL_NAMES = {"goal", "goals", "goal box", "goal_box", "goal mouth", "goal_mouth", "porteria", "portería", "arco"}


def _bbox_center(box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def _bbox_size(box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([max(2.0, x2 - x1), max(2.0, y2 - y1)], dtype=np.float64)


def _box_from_center_size(center: np.ndarray, size: np.ndarray) -> list[float]:
    half = np.maximum(size, 2.0) * 0.5
    return [
        float(center[0] - half[0]),
        float(center[1] - half[1]),
        float(center[0] + half[0]),
        float(center[1] + half[1]),
    ]


def _clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), max(0.0, width - 1.0))
    y1 = min(max(0.0, y1), max(0.0, height - 1.0))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center_crop(box: list[float], fraction: float = 0.72) -> list[float]:
    center = _bbox_center(box)
    size = _bbox_size(box) * float(fraction)
    return _box_from_center_size(center, size)


def _appearance_descriptor(
    frame: np.ndarray | None,
    box: list[float],
) -> np.ndarray | None:
    """HSV descriptor from the central robot region.

    The center crop excludes much of the surrounding field and nearby robots.
    Green pixels are suppressed so appearance changes less during camera motion.
    """
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(_center_crop(box), width, height)
    x1i, y1i, x2i, y2i = map(int, map(round, (x1, y1, x2, y2)))
    if x2i - x1i < 4 or y2i - y1i < 4:
        return None

    crop = frame[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (val > 32) & ~((hue >= 30) & (hue <= 100) & (sat > 45))
    if int(mask.sum()) < 40:
        mask = val > 32

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        mask.astype(np.uint8) * 255,
        [18, 8],
        [0, 180, 0, 256],
    ).astype(np.float64).reshape(-1)
    total = float(hist.sum())
    return hist / total if total > 0.0 else None


def _appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.45
    coefficient = float(np.sqrt(np.maximum(a, 0.0) * np.maximum(b, 0.0)).sum())
    return min(1.0, max(0.0, 1.0 - coefficient))


def _field_support_score(frame: np.ndarray | None, box: list[float]) -> float:
    """Estimate how much green/white playing surface surrounds a robot box."""
    if frame is None or frame.size == 0:
        return 1.0

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    box_width = max(4.0, x2 - x1)
    box_height = max(4.0, y2 - y1)

    # A ring around the lower half is more reliable than sampling only below;
    # robots often touch a wall or goal on one side.
    rx1 = int(max(0, x1 - 0.18 * box_width))
    rx2 = int(min(width, x2 + 0.18 * box_width))
    ry1 = int(max(0, y1 + 0.45 * box_height))
    ry2 = int(min(height, y2 + 0.18 * box_height))
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0

    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([28, 30, 28]), np.array([105, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([179, 88, 255]))
    green_ratio = float((green > 0).mean())
    white_ratio = float((white > 0).mean())
    # White only counts as a field line when green surface is also present.
    # This prevents white badges, shirts or walls from looking like a field.
    return green_ratio + (0.20 * white_ratio if green_ratio >= 0.02 else 0.0)


def _refine_orange_ball(frame: np.ndarray | None, box: list[float]) -> list[float]:
    if frame is None or frame.size == 0:
        return box

    height, width = frame.shape[:2]
    center = _bbox_center(box)
    size = _bbox_size(box)
    search_size = np.maximum(size * 2.3, np.array([20.0, 20.0]))
    sx1, sy1, sx2, sy2 = map(
        int,
        map(round, _clip_box(_box_from_center_size(center, search_size), width, height)),
    )
    roi = frame[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return box

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([3, 90, 55]), np.array([29, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expected_area = max(4.0, float(size[0] * size[1]))
    best_score = float("inf")
    best_box: list[float] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 3.0 or area > expected_area * 5.5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            continue
        contour_center = np.array([sx1 + x + w * 0.5, sy1 + y + h * 0.5])
        distance = float(np.linalg.norm(contour_center - center))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 0.0
            else 0.0
        )
        area_penalty = abs(log(max(area, 1.0) / expected_area))
        score = (
            distance / max(8.0, float(size.max()))
            + 0.34 * area_penalty
            + 0.42 * (1.0 - min(1.0, circularity))
        )
        if score < best_score:
            best_score = score
            pad = 1.5
            best_box = [
                sx1 + x - pad,
                sy1 + y - pad,
                sx1 + x + w + pad,
                sy1 + y + h + pad,
            ]

    if best_box is None or best_score > 2.4:
        return box
    return _clip_box(best_box, width, height)


@dataclass
class _Track:
    logical_id: int
    class_name: str
    class_id: int
    center: np.ndarray
    size: np.ndarray
    confidence: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    appearance: np.ndarray | None = None
    reference_appearance: np.ndarray | None = None
    age: int = 1
    hits: int = 1
    consecutive_hits: int = 1
    missed: int = 0
    confirmed: bool = False
    measured_center: np.ndarray | None = None
    previous_measured_center: np.ndarray | None = None
    last_template: dict[str, Any] = field(default_factory=dict)
    identity_guard_frames: int = 0

    @classmethod
    def from_detection(
        cls,
        logical_id: int,
        detection: dict[str, Any],
        appearance: np.ndarray | None,
    ) -> "_Track":
        box = list(map(float, detection["bbox_xyxy"]))
        center = _bbox_center(box)
        return cls(
            logical_id=logical_id,
            class_name=str(detection.get("class_name", "robot")),
            class_id=int(detection.get("class_id", -1)),
            center=center,
            size=_bbox_size(box),
            confidence=float(detection.get("confidence", 0.0)),
            appearance=appearance,
            reference_appearance=appearance.copy() if appearance is not None else None,
            measured_center=center.copy(),
            previous_measured_center=center.copy(),
            last_template=detection.copy(),
        )

    @property
    def box(self) -> list[float]:
        return _box_from_center_size(self.center, self.size)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def predict(self, dt: float) -> None:
        damping = 0.92 if self.missed < 4 else 0.72
        self.center = self.center + self.velocity * dt
        self.velocity *= damping
        self.age += 1
        self.missed += 1
        self.identity_guard_frames = max(0, self.identity_guard_frames - 1)
        # Keep the consecutive-hit count until the next measurement decides
        # whether the sequence was continuous.

    def update(
        self,
        detection: dict[str, Any],
        dt: float,
        appearance: np.ndarray | None,
        confirmation_hits: int,
        is_ball: bool = False,
        allow_appearance_update: bool = True,
    ) -> None:
        box = list(map(float, detection["bbox_xyxy"]))
        measured_center = _bbox_center(box)
        measured_size = _bbox_size(box)
        confidence = float(detection.get("confidence", 0.0))
        residual = measured_center - self.center
        normalized_residual = float(np.linalg.norm(residual)) / max(
            12.0, float(np.linalg.norm(self.size))
        )

        # Small residuals are smoothed strongly; actual fast movement follows
        # the measurement more quickly to avoid a visibly lagging box.
        if is_ball:
            alpha = min(0.88, 0.50 + 0.20 * normalized_residual + 0.12 * confidence)
            velocity_alpha = 0.34
            max_speed = 2800.0
        else:
            alpha = min(0.80, 0.34 + 0.24 * normalized_residual + 0.14 * confidence)
            velocity_alpha = 0.22
            max_speed = 1300.0

        self.center = self.center + alpha * residual

        if self.measured_center is not None and dt > 1e-6:
            measured_velocity = (measured_center - self.measured_center) / dt
            self.velocity = (
                (1.0 - velocity_alpha) * self.velocity
                + velocity_alpha * measured_velocity
            )
            speed = float(np.linalg.norm(self.velocity))
            if speed > max_speed:
                self.velocity *= max_speed / speed

        size_alpha = 0.44 if is_ball else 0.20 + 0.16 * confidence
        self.size = (1.0 - size_alpha) * self.size + size_alpha * measured_size

        if appearance is not None and allow_appearance_update:
            if self.appearance is None:
                self.appearance = appearance.copy()
            else:
                self.appearance = 0.82 * self.appearance + 0.18 * appearance
                total = float(self.appearance.sum())
                if total > 0.0:
                    self.appearance /= total

            # Only clean, high-confidence observations update the long-term
            # reference used for identity association.
            if confidence >= 0.75:
                if self.reference_appearance is None:
                    self.reference_appearance = appearance.copy()
                else:
                    self.reference_appearance = (
                        0.95 * self.reference_appearance + 0.05 * appearance
                    )
                    total = float(self.reference_appearance.sum())
                    if total > 0.0:
                        self.reference_appearance /= total

        self.confidence = confidence
        self.hits += 1
        self.consecutive_hits = self.consecutive_hits + 1 if self.missed <= 1 else 1
        self.missed = 0
        if self.consecutive_hits >= confirmation_hits or self.hits >= confirmation_hits + 1:
            self.confirmed = True
        self.previous_measured_center = self.measured_center
        self.measured_center = measured_center
        self.last_template = detection.copy()

    def to_detection(
        self,
        predicted: bool,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        result = self.last_template.copy()
        result["class_id"] = self.class_id
        result["class_name"] = self.class_name
        result["tracking_id"] = self.logical_id
        result["bbox_xyxy"] = [
            round(value, 2)
            for value in _clip_box(self.box, frame_width, frame_height)
        ]
        result["confidence"] = round(float(self.confidence * (0.82 ** self.missed)), 6)
        result["predicted"] = bool(predicted)
        result["measured"] = not predicted
        result["confirmed"] = bool(self.confirmed)
        result["tracking_status"] = "estimado" if predicted else "medido"
        result["track_age_frames"] = self.age
        result["track_hits"] = self.hits
        result["track_missed_frames"] = self.missed
        result["identity_check_required"] = bool(self.identity_guard_frames > 0)
        return result


class FutbotTemporalTracker:
    """Track four robots and one ball with a conservative lifecycle."""

    def __init__(
        self,
        fps: float,
        frame_width: int,
        frame_height: int,
        max_robots: int = 4,
        robot_confirmation_hits: int = 3,
        ball_confirmation_hits: int = 2,
        prediction_display_frames: int = 2,
        robot_memory_seconds: float = 0.75,
        ball_memory_seconds: float = 1.00,
        minimum_field_support: float = 0.055,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.dt = 1.0 / self.fps
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.frame_diagonal = hypot(self.frame_width, self.frame_height)
        self.max_robots = int(max_robots)
        self.robot_confirmation_hits = max(1, int(robot_confirmation_hits))
        self.ball_confirmation_hits = max(1, int(ball_confirmation_hits))
        self.prediction_display_frames = max(0, int(prediction_display_frames))
        self.robot_memory_frames = max(
            self.prediction_display_frames + 2,
            int(round(robot_memory_seconds * self.fps)),
        )
        self.ball_memory_frames = max(
            self.prediction_display_frames + 1,
            int(round(ball_memory_seconds * self.fps)),
        )
        self.minimum_field_support = float(minimum_field_support)

        self.robot_tracks: dict[int, _Track] = {}
        self.ball_track: _Track | None = None
        self.ball_recovery = AdaptiveBallRecovery(frame_width, frame_height)
        self.last_rejections: list[dict[str, Any]] = []
        self.last_field_candidates: list[dict[str, Any]] = []

    def _allocate_robot_id(self) -> int | None:
        for logical_id in range(self.max_robots):
            if logical_id not in self.robot_tracks:
                return logical_id
        return None

    def _robot_cost(
        self,
        track: _Track,
        detection: dict[str, Any],
        appearance: np.ndarray | None,
    ) -> float:
        box = list(map(float, detection["bbox_xyxy"]))
        det_center = _bbox_center(box)
        det_size = _bbox_size(box)
        distance = float(np.linalg.norm(det_center - track.center))
        scale = max(
            24.0,
            0.5 * (
                float(np.linalg.norm(track.size))
                + float(np.linalg.norm(det_size))
            ),
        )
        normalized_distance = distance / scale
        iou = _bbox_iou(track.box, box)
        size_difference = (
            abs(log(det_size[0] / track.size[0]))
            + abs(log(det_size[1] / track.size[1]))
        )
        reference = (
            track.reference_appearance
            if track.reference_appearance is not None
            else track.appearance
        )
        appearance_difference = _appearance_distance(reference, appearance)
        confidence = float(detection.get("confidence", 0.0))

        direction_penalty = 0.0
        if track.measured_center is not None and track.speed > 35.0:
            observed_motion = det_center - track.measured_center
            observed_speed = float(np.linalg.norm(observed_motion))
            if observed_speed > 3.0:
                cosine = float(
                    np.dot(observed_motion, track.velocity)
                    / max(observed_speed * track.speed, 1e-9)
                )
                # During a crossing, assigning the other robot commonly causes
                # an abrupt reversal. Penalize it without forbidding turns.
                direction_penalty = max(0.0, 0.35 - cosine)

        distance_gate = 2.6 + min(track.missed, 8) * 0.16
        absolute_gate = max(
            48.0,
            0.095 * self.frame_diagonal + track.speed * self.dt * 2.0,
        )
        if normalized_distance > distance_gate and distance > absolute_gate and iou < 0.01:
            return float("inf")
        if track.confirmed and appearance_difference > 0.88 and normalized_distance > 0.85:
            return float("inf")
        # A very different colour/appearance is not allowed to steal a
        # confirmed ID merely because both boxes overlap at a crossing.
        if track.confirmed and track.reference_appearance is not None and appearance_difference > 0.76:
            return float("inf")

        return (
            0.52 * normalized_distance
            + 0.22 * (1.0 - iou)
            + 0.09 * size_difference
            + 0.56 * appearance_difference
            + 0.20 * direction_penalty
            - 0.10 * confidence
        )

    @staticmethod
    def _optimal_assignment(
        costs: list[list[float]],
        unmatched_cost: float = 2.05,
    ) -> list[tuple[int, int]]:
        track_count = len(costs)
        detection_count = len(costs[0]) if track_count else 0

        @lru_cache(maxsize=None)
        def solve(
            track_index: int,
            used_mask: int,
        ) -> tuple[float, tuple[tuple[int, int], ...]]:
            if track_index >= track_count:
                return 0.0, ()

            best_cost, best_pairs = solve(track_index + 1, used_mask)
            best_cost += unmatched_cost
            for detection_index in range(detection_count):
                if used_mask & (1 << detection_index):
                    continue
                pair_cost = costs[track_index][detection_index]
                if not np.isfinite(pair_cost):
                    continue
                tail_cost, tail_pairs = solve(
                    track_index + 1,
                    used_mask | (1 << detection_index),
                )
                total = pair_cost + tail_cost
                if total < best_cost:
                    best_cost = total
                    best_pairs = ((track_index, detection_index),) + tail_pairs
            return best_cost, best_pairs

        return list(solve(0, 0)[1])

    def _filter_robot_candidates(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None,
        field_box: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        for detection in detections:
            item = detection.copy()
            score = _field_support_score(frame, list(map(float, item["bbox_xyxy"])))
            item["field_support_score"] = round(score, 5)
            confidence = float(item.get("confidence", 0.0))

            if field_box is not None:
                fx1, fy1, fx2, fy2 = map(float, field_box)
                field_width = max(1.0, fx2 - fx1)
                field_height = max(1.0, fy2 - fy1)
                bx1, by1, bx2, by2 = map(float, item["bbox_xyxy"])
                anchor_x = (bx1 + bx2) * 0.5
                anchor_y = by2
                inside_field = (
                    fx1 - 0.10 * field_width <= anchor_x <= fx2 + 0.10 * field_width
                    and fy1 - 0.12 * field_height <= anchor_y <= fy2 + 0.12 * field_height
                )
                item["inside_detected_field"] = bool(inside_field)
                if not inside_field and confidence < 0.94:
                    rejected = item.copy()
                    rejected["rejection_reason"] = "fuera_de_la_cancha_detectada"
                    self.last_rejections.append(rejected)
                    continue

            # Strong detections receive a small exception for robots touching a
            # dark wall. Low-confidence candidates must clearly sit on the field.
            required_support = self.minimum_field_support
            if confidence >= 0.88:
                required_support *= 0.45
            elif confidence >= 0.72:
                required_support *= 0.72

            if score < required_support:
                rejected = item.copy()
                rejected["rejection_reason"] = "sin_soporte_visual_de_cancha"
                rejected["required_field_support"] = round(required_support, 5)
                self.last_rejections.append(rejected)
                continue
            accepted.append(item)
        return accepted

    def _update_robots(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None,
        field_box: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        detections = self._filter_robot_candidates(detections, frame, field_box)
        detections = sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )[: max(self.max_robots * 2, self.max_robots)]

        for track in self.robot_tracks.values():
            track.predict(self.dt)

        descriptors = [
            _appearance_descriptor(frame, list(map(float, detection["bbox_xyxy"])))
            for detection in detections
        ]
        track_ids = sorted(self.robot_tracks)
        costs = [
            [
                self._robot_cost(self.robot_tracks[track_id], detection, descriptors[index])
                for index, detection in enumerate(detections)
            ]
            for track_id in track_ids
        ]
        assignments = self._optimal_assignment(costs) if costs and detections else []
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()

        for track_index, detection_index in assignments:
            track_id = track_ids[track_index]
            track = self.robot_tracks[track_id]
            row = [value for value in costs[track_index] if np.isfinite(value)]
            row.sort()
            assigned_cost = costs[track_index][detection_index]
            second_cost = next(
                (value for value in row if value > assigned_cost + 1e-9),
                float("inf"),
            )
            assignment_margin = second_cost - assigned_cost
            assigned_box = list(map(float, detections[detection_index]["bbox_xyxy"]))
            overlaps_other_detection = any(
                other_index != detection_index
                and _bbox_iou(assigned_box, list(map(float, other["bbox_xyxy"]))) > 0.10
                for other_index, other in enumerate(detections)
            )
            close_to_other_track = any(
                other_id != track_id
                and _bbox_iou(assigned_box, self.robot_tracks[other_id].box) > 0.08
                for other_id in track_ids
            )
            ambiguous_crossing = (
                assignment_margin < 0.16
                or overlaps_other_detection
                or close_to_other_track
            )
            item = detections[detection_index]
            item["association_ambiguous"] = bool(ambiguous_crossing)
            if ambiguous_crossing:
                track.identity_guard_frames = max(
                    track.identity_guard_frames,
                    max(6, int(round(0.45 * self.fps))),
                )
            item["association_margin"] = (
                round(float(assignment_margin), 5)
                if np.isfinite(assignment_margin)
                else None
            )
            track.update(
                item,
                self.dt,
                descriptors[detection_index],
                confirmation_hits=self.robot_confirmation_hits,
                is_ball=False,
                # Never learn the other robot's appearance while boxes touch.
                allow_appearance_update=not ambiguous_crossing,
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        # Tentative tracks disappear immediately after a short miss. Confirmed
        # tracks remain in memory for re-identification but are not drawn for long.
        for track_id in list(self.robot_tracks):
            track = self.robot_tracks[track_id]
            if not track.confirmed and track.missed > 1:
                del self.robot_tracks[track_id]
            elif track.confirmed and track.missed > self.robot_memory_frames:
                del self.robot_tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            logical_id = self._allocate_robot_id()
            if logical_id is None:
                rejected = detection.copy()
                rejected["rejection_reason"] = "sin_id_de_robot_disponible"
                self.last_rejections.append(rejected)
                continue
            self.robot_tracks[logical_id] = _Track.from_detection(
                logical_id,
                detection,
                descriptors[detection_index],
            )

        output: list[dict[str, Any]] = []
        for logical_id in sorted(self.robot_tracks):
            track = self.robot_tracks[logical_id]
            if not track.confirmed:
                continue
            predicted = logical_id not in matched_tracks and track.missed > 0
            if predicted and track.missed > self.prediction_display_frames:
                continue
            output.append(
                track.to_detection(
                    predicted=predicted,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                )
            )
        return output

    def _ball_cost(self, track: _Track, detection: dict[str, Any]) -> float:
        box = list(map(float, detection["bbox_xyxy"]))
        center = _bbox_center(box)
        distance = float(np.linalg.norm(center - track.center))
        scale = max(10.0, float(np.linalg.norm(track.size)) * 1.8)
        normalized_distance = distance / scale
        if distance > max(80.0, 0.11 * self.frame_diagonal + track.speed * self.dt * 2.5):
            return float("inf")
        return normalized_distance - 0.15 * float(detection.get("confidence", 0.0))

    def _update_ball(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        refined: list[dict[str, Any]] = []
        for detection in detections:
            item = detection.copy()
            item["raw_bbox_xyxy"] = list(item["bbox_xyxy"])
            item["bbox_xyxy"] = _refine_orange_ball(
                frame,
                list(map(float, item["bbox_xyxy"])),
            )
            refined.append(item)

        if self.ball_track is not None:
            self.ball_track.predict(self.dt)

        matched = False
        recovered = False
        selected: dict[str, Any] | None = None
        if refined:
            if self.ball_track is None:
                selected = max(
                    refined,
                    key=lambda item: float(item.get("confidence", 0.0)),
                )
                self.ball_track = _Track.from_detection(0, selected, None)
                matched = True
            else:
                candidates = [
                    (self._ball_cost(self.ball_track, detection), detection)
                    for detection in refined
                ]
                valid = [pair for pair in candidates if np.isfinite(pair[0])]
                if valid:
                    _, selected = min(valid, key=lambda pair: pair[0])

        # Si YOLO no produjo una medición compatible, busca únicamente cerca de
        # la posición predicha usando el perfil naranja aprendido.
        if selected is None and self.ball_track is not None:
            selected = self.ball_recovery.recover(
                frame=frame,
                predicted_box=self.ball_track.box,
                missed_frames=self.ball_track.missed,
                template=self.ball_track.last_template,
            )
            recovered = selected is not None

        if selected is not None and self.ball_track is not None:
            if self.ball_track.hits == 1 and matched:
                # El track acaba de crearse; no se actualiza dos veces con la
                # misma medición, pero sí alimenta el modelo de color.
                pass
            else:
                self.ball_track.update(
                    selected,
                    self.dt,
                    appearance=None,
                    confirmation_hits=self.ball_confirmation_hits,
                    is_ball=True,
                )
                matched = True
            if not recovered:
                self.ball_recovery.update_model(frame, selected)

        if self.ball_track is None:
            return []
        if not self.ball_track.confirmed and self.ball_track.missed > 1:
            self.ball_track = None
            return []
        if self.ball_track.confirmed and self.ball_track.missed > self.ball_memory_frames:
            self.ball_track = None
            return []
        if not self.ball_track.confirmed:
            return []

        predicted = not matched and self.ball_track.missed > 0
        if predicted and self.ball_track.missed > self.prediction_display_frames:
            return []
        result = self.ball_track.to_detection(
            predicted,
            self.frame_width,
            self.frame_height,
        )
        result["tracking_id"] = "ball"
        if recovered:
            result["predicted"] = False
            result["measured"] = False
            result["recovered_by_color"] = True
            result["tracking_status"] = "recuperado"
        return [result]

    def _update_field(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Selecciona la cancha principal sin alterar las cajas de YOLO.

        V5 favorece la caja con mayor cobertura razonable y utiliza la
        confianza solo como señal secundaria. No hay suavizado ni persistencia
        entre cuadros, por lo que una cámara móvil no arrastra cajas viejas.
        """
        selected, diagnostics = select_main_field(
            detections,
            self.frame_width,
            self.frame_height,
        )
        self.last_field_candidates = diagnostics
        return [selected] if selected is not None else []

    def _update_goals(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Conserva como máximo dos porterías y les asigna un lado en imagen.

        La orientación aliado/rival se resolverá más adelante con la homografía.
        Por ahora izquierda/derecha solo describe el cuadro de video.
        """
        if not detections:
            return []
        strongest = sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )[:2]
        strongest.sort(key=lambda item: float(_bbox_center(item["bbox_xyxy"])[0]))
        output: list[dict[str, Any]] = []
        for index, detection in enumerate(strongest):
            item = detection.copy()
            center_x = float(_bbox_center(item["bbox_xyxy"])[0])
            if len(strongest) == 2:
                side = "izquierda" if index == 0 else "derecha"
            else:
                side = "izquierda" if center_x < self.frame_width * 0.5 else "derecha"
            item["class_group"] = "goal"
            item["goal_side_image"] = side
            item["goal_id"] = f"goal_{side}"
            item["display_name"] = f"Portería {side}"
            output.append(item)
        return output

    def update(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        self.last_rejections = []
        robots: list[dict[str, Any]] = []
        balls: list[dict[str, Any]] = []
        fields: list[dict[str, Any]] = []
        goals: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []

        for detection in detections:
            if not detection.get("bbox_xyxy"):
                continue
            item = detection.copy()
            class_name = str(item.get("class_name", "")).strip().lower()
            if class_name in ROBOT_NAMES:
                item["class_group"] = "robot"
                robots.append(item)
            elif class_name in BALL_NAMES:
                item["class_group"] = "ball"
                balls.append(item)
            elif class_name in FIELD_NAMES:
                item["class_group"] = "field"
                fields.append(item)
            elif class_name in GOAL_NAMES:
                item["class_group"] = "goal"
                goals.append(item)
            else:
                other.append(item)

        tracked: list[dict[str, Any]] = []
        field_output = self._update_field(fields)
        field_box = field_output[0]["bbox_xyxy"] if field_output else None
        tracked.extend(field_output)
        tracked.extend(self._update_robots(robots, frame, field_box))
        tracked.extend(self._update_ball(balls, frame))
        tracked.extend(self._update_goals(goals))
        tracked.extend(other)
        return tracked
