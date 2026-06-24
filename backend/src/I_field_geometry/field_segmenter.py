from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FIELD_CLASS_NAMES = {
    "field",
    "field surface",
    "field_surface",
    "playing field",
    "cancha",
    "campo",
    "superficie",
    "superficie cancha",
    "superficie_cancha",
}


@dataclass
class FieldMaskResult:
    mask: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    bbox_xyxy: list[float]
    polygon: np.ndarray
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(float(self.confidence), 6),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "bbox_xyxy": [round(float(value), 3) for value in self.bbox_xyxy],
            "coverage": round(float(self.coverage), 6),
            "polygon_points": int(len(self.polygon)),
        }


class FieldSegmenter:
    """Thin Ultralytics segmentation wrapper for the playing surface.

    The segmentation model is deliberately independent from the detector used
    for robots, ball and goals. This keeps the already-good object detector
    unchanged while allowing the field to be represented by a polygon.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = max(0.01, float(confidence_threshold))
        self.image_size = max(128, int(image_size))
        self.last_candidates: list[dict[str, Any]] = []

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el segmentador de cancha en: {self.weights_path}"
            )

        print("Importando segmentador Ultralytics...")
        from ultralytics import YOLO

        print(f"Cargando segmentador de cancha desde: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names

        try:
            import torch

            self.device: int | str = 0 if torch.cuda.is_available() else "cpu"
            self.use_half = bool(torch.cuda.is_available())
            device_name = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            )
        except Exception:
            self.device = "cpu"
            self.use_half = False
            device_name = "CPU"
        print(f"Segmentador de cancha ejecutándose en: {device_name}")

    def _class_name(self, class_id: int) -> str:
        names = self.class_names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _mask_polygon(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 2), dtype=np.float32)
        contour = max(contours, key=cv2.contourArea)
        return contour.reshape(-1, 2).astype(np.float32)

    def segment_frame(self, frame: np.ndarray) -> FieldMaskResult | None:
        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            half=self.use_half,
            retina_masks=False,
            verbose=False,
        )
        self.last_candidates = []
        if not results:
            return None

        result = results[0]
        if result.boxes is None or result.masks is None:
            return None

        mask_data = result.masks.data
        if mask_data is None or len(mask_data) == 0:
            return None

        candidates: list[tuple[float, FieldMaskResult]] = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = self._class_name(class_id)
            normalized_name = class_name.strip().lower()
            # A one-class segmentation checkpoint is accepted even if the user
            # chose a custom class name in Roboflow.
            if len(self.class_names) > 1 and normalized_name not in FIELD_CLASS_NAMES:
                continue

            confidence = float(box.conf[0].detach().cpu().item())
            raw_mask = mask_data[index].detach().cpu().numpy()
            if raw_mask.shape != (height, width):
                raw_mask = cv2.resize(
                    raw_mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            binary = (raw_mask >= 0.50).astype(np.uint8) * 255
            coverage = float(np.count_nonzero(binary)) / max(1.0, float(width * height))
            if coverage < 0.015:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            polygon = self._mask_polygon(binary)
            candidate = FieldMaskResult(
                mask=binary,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                bbox_xyxy=xyxy,
                polygon=polygon,
                coverage=coverage,
            )
            # Coverage dominates because there should be only one physical field.
            score = 0.72 * coverage + 0.28 * confidence
            self.last_candidates.append({**candidate.to_dict(), "score": round(score, 6)})
            candidates.append((score, candidate))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]
