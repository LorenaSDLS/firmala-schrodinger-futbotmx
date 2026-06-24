from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.shared.paths import YOLO_WEIGHTS_PATH


ROBOT_NAMES = {"robot", "robots"}
BALL_NAMES = {"orange ball", "ball", "pelota", "balon", "balón"}
FIELD_NAMES = {"field", "playing field", "cancha", "campo"}
GOAL_NAMES = {"goal", "goals", "goal box", "goal_box", "goal mouth", "goal_mouth", "porteria", "portería", "arco"}

DEFAULT_CLASS_THRESHOLDS = {
    "robot": 0.55,
    "ball": 0.35,
    "field": 0.25,
    "goal": 0.35,
}


def normalize_class_group(class_name: str) -> str:
    name = str(class_name).strip().lower()
    if name in ROBOT_NAMES:
        return "robot"
    if name in BALL_NAMES:
        return "ball"
    if name in FIELD_NAMES:
        return "field"
    if name in GOAL_NAMES:
        return "goal"
    return "other"


class YOLODetector:
    """YOLO wrapper with class-specific confidence thresholds.

    A single low confidence threshold was allowing badges and clothing to enter
    the robot tracker. The detector still asks YOLO for low-confidence results
    so the small ball is not lost, then applies a stricter threshold per class.
    """

    def __init__(
        self,
        weights_path: str | Path = YOLO_WEIGHTS_PATH,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = max(0.01, float(confidence_threshold))
        self.image_size = int(image_size)
        self.class_thresholds = DEFAULT_CLASS_THRESHOLDS.copy()
        if class_thresholds:
            self.class_thresholds.update(
                {str(key): float(value) for key, value in class_thresholds.items()}
            )

        self.last_rejected_detections: list[dict[str, Any]] = []

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo YOLO en: {self.weights_path}"
            )

        print("Importando Ultralytics YOLO...")
        from ultralytics import YOLO

        print(f"Cargando YOLO desde: {self.weights_path}")
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
        print(f"Detector principal ejecutándose en: {device_name}")

    def _threshold_for(self, class_name: str) -> float:
        group = normalize_class_group(class_name)
        # Los umbrales conocidos son independientes. Así la cancha puede usar
        # 0.25 sin reducir robots (0.55) ni balón (0.35). El umbral global se
        # conserva únicamente como fallback para clases no configuradas.
        if group in self.class_thresholds:
            return max(0.01, float(self.class_thresholds[group]))
        return self.confidence_threshold

    def detect_frame(self, frame) -> list[dict[str, Any]]:
        # Ask YOLO for all candidates needed by the most permissive class.
        inference_threshold = min(
            [self.confidence_threshold, *self.class_thresholds.values()]
        )
        results = self.model.predict(
            source=frame,
            conf=max(0.01, inference_threshold),
            imgsz=self.image_size,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        self.last_rejected_detections = []

        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1.0, float(frame_width * frame_height))

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = str(self.class_names.get(class_id, str(class_id)))
            group = normalize_class_group(class_name)

            x1, y1, x2, y2 = map(float, xyxy)
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area_ratio = width * height / frame_area

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "class_group": group,
                "confidence": confidence,
                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
            }

            required_confidence = self._threshold_for(class_name)
            if confidence < required_confidence:
                rejected = detection.copy()
                rejected["rejection_reason"] = "confianza_insuficiente"
                rejected["required_confidence"] = required_confidence
                self.last_rejected_detections.append(rejected)
                continue

            # Conservative geometric checks. They only remove obviously broken
            # boxes; contextual field support is checked by the temporal tracker.
            if group == "robot" and (width < 18 or height < 18 or area_ratio < 0.00008):
                rejected = detection.copy()
                rejected["rejection_reason"] = "caja_de_robot_demasiado_pequena"
                self.last_rejected_detections.append(rejected)
                continue

            if group == "ball" and (width < 2 or height < 2 or area_ratio > 0.025):
                rejected = detection.copy()
                rejected["rejection_reason"] = "tamano_de_balon_invalido"
                self.last_rejected_detections.append(rejected)
                continue

            if group == "goal" and (width < 12 or height < 12 or area_ratio < 0.00015):
                rejected = detection.copy()
                rejected["rejection_reason"] = "caja_de_porteria_invalida"
                self.last_rejected_detections.append(rejected)
                continue

            detections.append(detection)

        return detections


DISPLAY_NAMES_ES = {
    "robot": "Robot",
    "ball": "Balon",
    "field": "Cancha",
    "goal": "Porteria",
}

TEAM_COLORS_BGR = {
    "aliado": (255, 170, 0),
    "rival": (255, 0, 220),
    "desconocido": (120, 120, 120),
}


def draw_yolo_detections(frame, detections: list[dict[str, Any]]):
    """Draw production-facing labels in Spanish.

    OpenCV's built-in font is ASCII-only on many installations, therefore the
    on-video label uses "Balon" instead of an accented glyph.
    """
    annotated_frame = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        class_name = str(detection.get("class_name", ""))
        class_group = detection.get("class_group") or normalize_class_group(class_name)
        confidence = float(detection.get("confidence", 0.0))

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        tracking_id = detection.get("tracking_id")
        predicted = bool(detection.get("predicted", False))

        if class_group == "robot":
            team = str(detection.get("team", "desconocido"))
            display_name = detection.get("display_name")
            if not display_name:
                display_name = (
                    f"Robot {int(tracking_id) + 1}"
                    if tracking_id is not None
                    else "Robot"
                )
            color = TEAM_COLORS_BGR.get(team, TEAM_COLORS_BGR["desconocido"])
            label = f"{display_name} {confidence:.2f}"
        elif class_group == "ball":
            color = (0, 120, 255)
            label = f"Balon {confidence:.2f}"
        elif class_group == "field":
            color = (235, 235, 235)
            label = f"Cancha {confidence:.2f}"
        elif class_group == "goal":
            color = (0, 215, 255)
            side = str(detection.get("goal_side_image", "")).strip()
            side_text = f" {side.capitalize()}" if side else ""
            label = f"Porteria{side_text} {confidence:.2f}"
        else:
            color = (0, 255, 0)
            label = f"{class_name} {confidence:.2f}"

        if bool(detection.get("recovered_by_color", False)):
            label += " RECUPERADO"
        elif predicted:
            label += " ESTIMADO"

        thickness = 1 if predicted else 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame
