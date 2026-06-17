from pathlib import Path
from typing import Any

import cv2

from src.shared.paths import YOLO_WEIGHTS_PATH


class YOLODetector:
    def __init__(
        self,
        weights_path: str | Path = YOLO_WEIGHTS_PATH,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo YOLO en: {self.weights_path}"
            )

        print("Importando Ultralytics YOLO...")
        from ultralytics import YOLO

        print(f"Cargando YOLO desde: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names

    def detect_frame(self, frame) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = self.class_names.get(class_id, str(class_id))

            x1, y1, x2, y2 = xyxy

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2), 2),
                        round(float(y2), 2),
                    ],
                }
            )

        return detections


def draw_yolo_detections(frame, detections: list[dict[str, Any]]):
    annotated_frame = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated_frame