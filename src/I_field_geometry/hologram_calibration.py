from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.I_field_geometry.field_spec import FieldSpec


def _as_homography(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("La homografía del holograma no es invertible.")
    return matrix / matrix[2, 2]


@dataclass(frozen=True)
class HologramKeyframe:
    frame_index: int
    corners_image: np.ndarray
    confidence: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_image, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(corners).all():
            raise ValueError("Las esquinas del holograma contienen valores inválidos.")
        if abs(float(cv2.contourArea(corners))) < 25.0:
            raise ValueError("El holograma es degenerado o demasiado pequeño.")
        object.__setattr__(self, "corners_image", corners)
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.05, 1.0)))

    def field_to_image(self, spec: FieldSpec) -> np.ndarray:
        field_corners = np.float32(
            [
                [0.0, 0.0],
                [spec.surface_length_cm, 0.0],
                [spec.surface_length_cm, spec.surface_width_cm],
                [0.0, spec.surface_width_cm],
            ]
        )
        return _as_homography(cv2.getPerspectiveTransform(field_corners, self.corners_image))

    def image_to_field(self, spec: FieldSpec) -> np.ndarray:
        return _as_homography(np.linalg.inv(self.field_to_image(spec)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "corners_image": self.corners_image.astype(float).tolist(),
            "confidence": round(float(self.confidence), 6),
            "note": str(self.note),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HologramKeyframe":
        return cls(
            frame_index=int(payload["frame_index"]),
            corners_image=np.asarray(payload["corners_image"], dtype=np.float32),
            confidence=float(payload.get("confidence", 1.0)),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class HologramCalibration:
    frame_width: int
    frame_height: int
    fps: float
    total_frames: int
    field_spec: FieldSpec = field(default_factory=FieldSpec)
    keyframes: tuple[HologramKeyframe, ...] = field(default_factory=tuple)
    source: str = "holograma_asistido_v11"
    version: int = 11

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("La resolución del video es inválida.")
        if not self.keyframes:
            raise ValueError("La calibración holográfica necesita al menos un keyframe.")
        ordered = tuple(sorted(self.keyframes, key=lambda item: item.frame_index))
        if len({item.frame_index for item in ordered}) != len(ordered):
            raise ValueError("No puede haber dos keyframes en el mismo cuadro.")
        object.__setattr__(self, "keyframes", ordered)

    @property
    def field_width(self) -> float:
        return float(self.field_spec.surface_length_cm)

    @property
    def field_height(self) -> float:
        return float(self.field_spec.surface_width_cm)

    def scaled_to(self, width: int, height: int) -> "HologramCalibration":
        width, height = int(width), int(height)
        if width == self.frame_width and height == self.frame_height:
            return self
        sx = width / max(1.0, float(self.frame_width))
        sy = height / max(1.0, float(self.frame_height))
        keyframes = tuple(
            HologramKeyframe(
                frame_index=item.frame_index,
                corners_image=item.corners_image * np.float32([sx, sy]),
                confidence=item.confidence,
                note=item.note,
            )
            for item in self.keyframes
        )
        return HologramCalibration(
            frame_width=width,
            frame_height=height,
            fps=self.fps,
            total_frames=self.total_frames,
            field_spec=self.field_spec,
            keyframes=keyframes,
            source=self.source,
            version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "source": self.source,
            "calibration_type": "assisted_hologram",
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "field_spec": self.field_spec.to_dict(),
            "keyframes": [item.to_dict() for item in self.keyframes],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "HologramCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("calibration_type") != "assisted_hologram" and not str(
            payload.get("source", "")
        ).startswith("holograma_asistido"):
            raise ValueError("El archivo no es una calibración holográfica V11.")
        return cls(
            frame_width=int(payload["frame_width"]),
            frame_height=int(payload["frame_height"]),
            fps=float(payload.get("fps", 30.0)),
            total_frames=int(payload.get("total_frames", 0)),
            field_spec=FieldSpec.from_dict(payload.get("field_spec", {})),
            keyframes=tuple(
                HologramKeyframe.from_dict(item)
                for item in payload.get("keyframes", [])
            ),
            source=str(payload.get("source", "holograma_asistido_v11")),
            version=int(payload.get("version", 11)),
        )


def is_hologram_calibration(path: str | Path | None) -> bool:
    if path is None:
        return False
    source = Path(path)
    if not source.exists():
        return False
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("calibration_type") == "assisted_hologram"
        or str(payload.get("source", "")).startswith("holograma_asistido")
    )
