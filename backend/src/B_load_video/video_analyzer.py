from pathlib import Path

import cv2

from src.B_load_video.video_validator import (
    validate_video_file,
    validate_video_properties,
)
from src.shared.models import VideoMetadata
from src.shared.paths import clean_video_name


def decode_codec(codec_value: int) -> str:
    if codec_value <= 0:
        return "unknown"

    codec = "".join(
        chr((codec_value >> (8 * index)) & 0xFF)
        for index in range(4)
    )

    return codec.strip() or "unknown"


def create_invalid_metadata(
    video_path: Path,
    errors: list[str],
) -> VideoMetadata:
    return VideoMetadata(
        video_name=clean_video_name(video_path),
        original_filename=video_path.name,
        format=video_path.suffix.lower().lstrip(".") or "unknown",
        codec="unknown",
        duration_seconds=0.0,
        fps=0.0,
        total_frames=0,
        width=0,
        height=0,
        status="invalid",
        validation_errors=errors,
    )


def analyze_video(video_path: str | Path) -> VideoMetadata:
    path = Path(video_path).expanduser().resolve()
    errors = validate_video_file(path)

    if not path.exists() or not path.is_file():
        return create_invalid_metadata(path, errors)

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        errors.append("OpenCV no pudo abrir el video.")
        capture.release()
        return create_invalid_metadata(path, errors)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))

    first_frame_read, _ = capture.read()
    capture.release()

    errors.extend(
        validate_video_properties(
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            first_frame_read=first_frame_read,
        )
    )

    duration_seconds = (
        total_frames / fps
        if fps > 0 and total_frames > 0
        else 0.0
    )

    return VideoMetadata(
        video_name=clean_video_name(path),
        original_filename=path.name,
        format=path.suffix.lower().lstrip(".") or "unknown",
        codec=decode_codec(codec_value),
        duration_seconds=duration_seconds,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        status="valid" if not errors else "invalid",
        validation_errors=errors,
    )