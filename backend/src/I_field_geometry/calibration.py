from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.I_field_geometry.feature_constraints import (
    CANONICAL_LINES_NORMALIZED,
    LONGITUDINAL_FEATURES,
    TRANSVERSE_FEATURES,
    has_global_line_support,
    homography_from_semantic_lines,
    line_from_segment,
    local_rectification_from_segments,
    score_manual_anchors,
)

LINE_ORDER = ("near", "far", "left", "right")
FEATURE_ORDER = ("near", "far", "left", "right", "center", "near_area", "far_area")
LINE_LABELS_ES = {
    "near": "Linea de gol CERCANA",
    "far": "Linea de gol LEJANA",
    "left": "Lateral IZQUIERDA (mirando de cercana a lejana)",
    "right": "Lateral DERECHA (mirando de cercana a lejana)",
    "center": "Linea CENTRAL",
    "near_area": "Frente del area CERCANA",
    "far_area": "Frente del area LEJANA",
}
LINE_COLORS = {
    "near": (0, 220, 255), "far": (255, 0, 210), "left": (70, 255, 70),
    "right": (255, 180, 40), "center": (255, 255, 40),
    "near_area": (70, 190, 255), "far_area": (255, 90, 210),
}


def _normalize_line(line: np.ndarray) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64).reshape(3)
    norm = float(np.hypot(line[0], line[1]))
    if norm < 1e-9:
        raise ValueError("No se puede construir una recta con puntos coincidentes.")
    return line / norm


def line_from_two_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return line_from_segment(np.asarray([first, second], dtype=np.float64))


def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point = np.cross(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))
    if abs(float(point[2])) < 1e-10:
        raise ValueError("Dos rectas de calibración son prácticamente paralelas.")
    point = point[:2] / point[2]
    if not np.isfinite(point).all():
        raise ValueError("La intersección de rectas no es finita.")
    return point.astype(np.float32)


def corners_from_semantic_lines(lines: dict[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in LINE_ORDER if name not in lines]
    if missing:
        raise ValueError(f"Faltan rectas de calibración: {', '.join(missing)}")
    return np.float32([
        intersect_lines(lines["far"], lines["left"]),
        intersect_lines(lines["far"], lines["right"]),
        intersect_lines(lines["near"], lines["right"]),
        intersect_lines(lines["near"], lines["left"]),
    ])


def calibration_homography(corners_image: np.ndarray, field_width: float, field_height: float) -> np.ndarray:
    canonical = np.float32([[field_width, 0.0], [field_width, field_height], [0.0, field_height], [0.0, 0.0]])
    homography = cv2.getPerspectiveTransform(np.asarray(corners_image, dtype=np.float32).reshape(4, 2), canonical)
    if not np.isfinite(homography).all():
        raise ValueError("La homografía calculada no es válida.")
    return homography


@dataclass(frozen=True)
class FieldCalibration:
    source_frame_index: int
    frame_width: int
    frame_height: int
    field_width: float
    field_height: float
    semantic_lines: dict[str, np.ndarray]
    semantic_segments: dict[str, np.ndarray] = field(default_factory=dict)
    corners_image: np.ndarray | None = None
    homography_image_to_field: np.ndarray | None = None
    local_homography_image_to_local: np.ndarray | None = None
    source: str = "calibracion_asistida_flexible_v8"

    @property
    def is_complete(self) -> bool:
        return self.homography_image_to_field is not None

    @property
    def has_global_registration(self) -> bool:
        return self.homography_image_to_field is not None

    @property
    def has_local_registration(self) -> bool:
        return self.local_homography_image_to_local is not None

    @property
    def feature_count(self) -> int:
        return len(self.semantic_lines)

    @property
    def transverse_count(self) -> int:
        return len(set(self.semantic_lines) & TRANSVERSE_FEATURES)

    @property
    def longitudinal_count(self) -> int:
        return len(set(self.semantic_lines) & LONGITUDINAL_FEATURES)

    def scaled_to(self, width: int, height: int) -> "FieldCalibration":
        width, height = int(width), int(height)
        if width == self.frame_width and height == self.frame_height:
            return self
        sx, sy = width / max(1.0, self.frame_width), height / max(1.0, self.frame_height)
        scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        inverse_scale = np.linalg.inv(scale)
        lines = {name: _normalize_line(inverse_scale.T @ line) for name, line in self.semantic_lines.items()}
        segments = {name: np.asarray(points, dtype=np.float32) * np.array([sx, sy], dtype=np.float32) for name, points in self.semantic_segments.items()}
        corners = None if self.corners_image is None else self.corners_image * np.array([sx, sy], dtype=np.float32)
        homography = None if self.homography_image_to_field is None else self.homography_image_to_field @ inverse_scale
        local = None if self.local_homography_image_to_local is None else self.local_homography_image_to_local @ inverse_scale
        return FieldCalibration(self.source_frame_index, width, height, self.field_width, self.field_height, lines, segments, corners, homography, local, self.source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3, "source": self.source, "source_frame_index": self.source_frame_index,
            "frame_width": self.frame_width, "frame_height": self.frame_height,
            "field_width": self.field_width, "field_height": self.field_height,
            "complete": self.is_complete, "global_registration": self.has_global_registration,
            "local_registration": self.has_local_registration,
            "corners_image": None if self.corners_image is None else self.corners_image.astype(float).tolist(),
            "homography_image_to_field": None if self.homography_image_to_field is None else self.homography_image_to_field.astype(float).tolist(),
            "local_homography_image_to_local": None if self.local_homography_image_to_local is None else self.local_homography_image_to_local.astype(float).tolist(),
            "semantic_lines": {name: line.astype(float).tolist() for name, line in self.semantic_lines.items()},
            "semantic_segments": {name: np.asarray(points, dtype=float).tolist() for name, points in self.semantic_segments.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FieldCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        lines = {str(name): _normalize_line(np.asarray(line, dtype=np.float64)) for name, line in payload.get("semantic_lines", {}).items()}
        segments = {str(name): np.asarray(points, dtype=np.float32).reshape(2, 2) for name, points in payload.get("semantic_segments", {}).items()}
        corners = payload.get("corners_image"); homography = payload.get("homography_image_to_field"); local = payload.get("local_homography_image_to_local")
        if local is None and segments:
            local, _, _ = local_rectification_from_segments(segments)
        return cls(
            int(payload.get("source_frame_index", 0)), int(payload["frame_width"]), int(payload["frame_height"]),
            float(payload.get("field_width", 100.0)), float(payload.get("field_height", 60.0)), lines, segments,
            None if corners is None else np.asarray(corners, dtype=np.float32).reshape(4, 2),
            None if homography is None else np.asarray(homography, dtype=np.float64).reshape(3, 3),
            None if local is None else np.asarray(local, dtype=np.float64).reshape(3, 3),
            str(payload.get("source", "calibracion_asistida_flexible_v8")),
        )


def create_calibration_from_points(
    points_by_line: dict[str, list[tuple[float, float]]], frame_width: int, frame_height: int,
    field_width: float = 100.0, field_height: float = 60.0, source_frame_index: int = 0,
) -> FieldCalibration:
    lines: dict[str, np.ndarray] = {}; segments: dict[str, np.ndarray] = {}
    for name, raw_points in points_by_line.items():
        if name not in CANONICAL_LINES_NORMALIZED or len(raw_points) != 2:
            continue
        segment = np.asarray(raw_points, dtype=np.float32).reshape(2, 2)
        if float(np.linalg.norm(segment[1] - segment[0])) < 8.0:
            raise ValueError(f"El segmento {name} es demasiado corto.")
        lines[name] = line_from_two_points(segment[0], segment[1]); segments[name] = segment
    if not lines:
        raise ValueError("Marca por lo menos una línea visible de la cancha.")

    local_h, local_source, _ = local_rectification_from_segments(segments)
    homography = None; corners = None; source = f"calibracion_asistida_{local_source}"
    solved = homography_from_semantic_lines(lines) if has_global_line_support(lines) else None
    if solved is not None:
        image_to_normalized, _field_to_image = solved
        scale = np.array([[field_width, 0, 0], [0, field_height, 0], [0, 0, 1]], dtype=np.float64)
        homography = scale @ image_to_normalized
        # Hard-check every manually labeled line. No average can hide one bad anchor.
        field_to_image = np.linalg.inv(image_to_normalized)
        scores = score_manual_anchors(segments, field_to_image, float(np.hypot(frame_width, frame_height)))
        if not scores or not all(item.hard_pass for item in scores) or min(item.score for item in scores) < 0.72:
            raise ValueError("Las líneas no forman una homografía consistente. Revisa solo la etiqueta o segmento dudoso.")
        canonical_corners = np.float32([[1, 0], [1, 1], [0, 1], [0, 0]])
        corners = cv2.perspectiveTransform(canonical_corners.reshape(1, 4, 2), field_to_image).reshape(4, 2)
        source = "calibracion_asistida_anclas_duras_v8"

    return FieldCalibration(
        int(source_frame_index), int(frame_width), int(frame_height), float(field_width), float(field_height),
        lines, segments, corners, homography, local_h, source,
    )
