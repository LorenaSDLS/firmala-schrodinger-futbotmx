"""Recuperación local del balón naranja cuando YOLO pierde algunos cuadros."""

from __future__ import annotations

from collections import deque
from math import log
from typing import Any

import cv2
import numpy as np


class AdaptiveBallRecovery:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.hue_samples: deque[float] = deque(maxlen=1800)
        self.saturation_samples: deque[float] = deque(maxlen=1800)
        self.value_samples: deque[float] = deque(maxlen=1800)
        self.area_samples: deque[float] = deque(maxlen=120)

    @staticmethod
    def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(float, box)
        ix1 = max(0, min(width - 1, int(np.floor(x1))))
        iy1 = max(0, min(height - 1, int(np.floor(y1))))
        ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
        iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))
        return ix1, iy1, ix2, iy2

    def update_model(self, frame: np.ndarray | None, detection: dict[str, Any]) -> None:
        if frame is None or frame.size == 0:
            return
        confidence = float(detection.get("confidence", 0.0))
        if confidence < 0.45:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_box(
            list(map(float, detection.get("bbox_xyxy", [0, 0, 0, 0]))),
            width,
            height,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        orange_like = (
            (hue >= 0)
            & (hue <= 38)
            & (saturation >= 75)
            & (value >= 55)
        )
        if int(orange_like.sum()) < 3:
            return
        h_values = hue[orange_like].reshape(-1)
        s_values = saturation[orange_like].reshape(-1)
        v_values = value[orange_like].reshape(-1)
        # Submuestreo para que una caja grande no domine el historial.
        step = max(1, len(h_values) // 80)
        self.hue_samples.extend(float(value) for value in h_values[::step])
        self.saturation_samples.extend(float(value) for value in s_values[::step])
        self.value_samples.extend(float(value) for value in v_values[::step])
        self.area_samples.append(float(max(1, (x2 - x1) * (y2 - y1))))

    def _thresholds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.hue_samples) < 20:
            return np.array([2, 80, 48]), np.array([32, 255, 255])
        hue = np.asarray(self.hue_samples, dtype=np.float32)
        saturation = np.asarray(self.saturation_samples, dtype=np.float32)
        value = np.asarray(self.value_samples, dtype=np.float32)
        h_med = float(np.median(hue))
        h_mad = float(np.median(np.abs(hue - h_med)))
        s_low = float(np.percentile(saturation, 8))
        v_low = float(np.percentile(value, 8))
        hue_radius = max(5.0, min(14.0, 3.0 * max(h_mad, 1.5)))
        lower = np.array(
            [max(0, int(h_med - hue_radius)), max(55, int(s_low - 22)), max(38, int(v_low - 28))],
            dtype=np.uint8,
        )
        upper = np.array(
            [min(42, int(h_med + hue_radius)), 255, 255],
            dtype=np.uint8,
        )
        return lower, upper

    def recover(
        self,
        frame: np.ndarray | None,
        predicted_box: list[float],
        missed_frames: int,
        template: dict[str, Any],
    ) -> dict[str, Any] | None:
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(float, predicted_box)
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
        box_width = max(3.0, x2 - x1)
        box_height = max(3.0, y2 - y1)
        growth = min(7.0, 2.8 + 0.24 * max(0, missed_frames))
        search_width = max(30.0, box_width * growth)
        search_height = max(30.0, box_height * growth)
        search_box = [
            center[0] - search_width * 0.5,
            center[1] - search_height * 0.5,
            center[0] + search_width * 0.5,
            center[1] + search_height * 0.5,
        ]
        sx1, sy1, sx2, sy2 = self._clip_box(search_box, width, height)
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower, upper = self._thresholds()
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        expected_area = (
            float(np.median(self.area_samples))
            if self.area_samples
            else max(6.0, box_width * box_height)
        )
        best: tuple[float, list[float], float] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(2.0, 0.12 * expected_area) or area > 4.2 * expected_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 2 or max(w, h) > 3.8 * max(box_width, box_height):
                continue
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > 2.6:
                continue
            candidate_center = np.array([sx1 + x + w * 0.5, sy1 + y + h * 0.5])
            distance = float(np.linalg.norm(candidate_center - center))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
                if perimeter > 1e-6
                else 0.0
            )
            area_penalty = abs(log(max(area, 1.0) / max(expected_area, 1.0)))
            score = (
                distance / max(12.0, 0.5 * (search_width + search_height))
                + 0.42 * area_penalty
                + 0.36 * (1.0 - min(1.0, circularity))
                + 0.10 * max(0.0, aspect - 1.0)
            )
            if best is None or score < best[0]:
                pad = 1.5
                candidate_box = [
                    max(0.0, sx1 + x - pad),
                    max(0.0, sy1 + y - pad),
                    min(float(width), sx1 + x + w + pad),
                    min(float(height), sy1 + y + h + pad),
                ]
                best = (score, candidate_box, circularity)

        if best is None or best[0] > 1.35:
            return None
        score, box, circularity = best
        confidence = float(np.clip(0.58 - 0.24 * score, 0.24, 0.56))
        result = template.copy()
        result.update(
            {
                "class_name": template.get("class_name", "orange ball"),
                "class_group": "ball",
                "bbox_xyxy": [round(float(value), 2) for value in box],
                "confidence": round(confidence, 6),
                "recovered_by_color": True,
                "recovery_score": round(float(score), 6),
                "recovery_circularity": round(float(circularity), 6),
            }
        )
        return result
