import json
from pathlib import Path
from typing import Any

import cv2

from src.C_quick_view.yolo_detector import (
    YOLODetector,
    draw_yolo_detections,
)


def generate_quick_preview(
    video_path: str | Path,
    output_directory: str | Path,
    confidence_threshold: float = 0.25,
    max_frames: int | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    preview_path = output_directory / "quick_preview.mp4"
    detections_path = output_directory / "quick_detections.jsonl"

    detector = YOLODetector(confidence_threshold=confidence_threshold)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
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

            detections = detector.detect_frame(frame)
            total_detections += len(detections)

            annotated_frame = draw_yolo_detections(frame, detections)
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
        "sam_mode": None,
        "fps": fps,
        "resolution": {
            "width": width,
            "height": height,
        },
    }