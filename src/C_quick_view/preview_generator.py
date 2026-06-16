import json
from pathlib import Path
from typing import Any

import cv2

from src.C_quick_view.sam_segmenter import SAMSegmenter
from src.C_quick_view.yolo_detector import YOLODetector, draw_yolo_detections


def draw_sam_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])

        label = (
            f"SAM {detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    return frame

def generate_quick_preview(
    video_path: str | Path,
    output_directory: str | Path,
    confidence_threshold: float = 0.25,
    sam_mode: str | None = "LoHa",
    sam_confidence: float = 0.40,
    max_frames: int | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    preview_path = output_directory / "quick_preview.mp4"
    detections_path = output_directory / "quick_detections.jsonl"

    detector = YOLODetector(confidence_threshold=confidence_threshold)

    sam = None

    if sam_mode is not None:
        sam = SAMSegmenter(
            mode=sam_mode,
            confidence_threshold=sam_confidence,
            api_path="Analizador de video/API_sam3.json",
        )


    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(preview_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0
    processed_frames = 0
    total_detections = 0

    with detections_path.open("w", encoding="utf-8") as detections_file:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if max_frames is not None and processed_frames >= max_frames:
                break

            yolo_detections = detector.detect_frame(frame)
            sam_detections = []

            # SAM3 + LoHa inicializa los objetos en el primer frame.
            if frame_index == 0 and sam is not None:
                print(f"SAM3 + {sam_mode} analizando el primer frame...")

                for prompt in ["playing field", "orange ball", "robots"]:
                    prompt_detections = sam.detect(frame, prompt)
                    sam_detections.extend(prompt_detections)

                    print(
                        f"  {prompt}: "
                        f"{len(prompt_detections)} detecciones"
                    )

            detections = yolo_detections + sam_detections
            total_detections += len(detections)

            annotated_frame = draw_yolo_detections(
                frame,
                yolo_detections,
            )

            annotated_frame = draw_sam_detections(
                annotated_frame,
                sam_detections,
            )
            writer.write(annotated_frame)

            frame_record = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 4),
                "detections": detections,
            }

            detections_file.write(
                json.dumps(frame_record, ensure_ascii=False) + "\n"
            )

            processed_frames += 1
            frame_index += 1

            if processed_frames % 30 == 0:
                print(
                    f"Procesados {processed_frames}/{total_frames} frames..."
                )

    capture.release()
    writer.release()

    return {
        "preview_path": str(preview_path),
        "detections_path": str(detections_path),
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "sam_mode": sam_mode,
        "fps": fps,
        "resolution": {
            "width": width,
            "height": height,
        },
    }

