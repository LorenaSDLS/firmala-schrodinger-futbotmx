from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.F_simulation.field_registration import FieldRegistration
from src.I_field_geometry.calibration import FieldCalibration, create_calibration_from_points


@dataclass(frozen=True)
class MultiframeLineObservation:
    """A semantic field segment observed in one video frame.

    ``segment_reference`` is expressed in the common camera-reference image so
    observations from different frames can be fitted together without pretending
    they were simultaneously visible.
    """

    name: str
    frame_index: int
    segment_frame: np.ndarray
    segment_reference: np.ndarray


def transform_segment(segment: np.ndarray, homography: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(1, 2, 2)
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    projected = cv2.perspectiveTransform(segment, matrix).reshape(2, 2)
    if not np.isfinite(projected).all():
        raise ValueError("La transformación de la línea no es finita.")
    return projected.astype(np.float32)


def fit_reference_segment(segments: Iterable[np.ndarray]) -> np.ndarray:
    """Fit one stable line to repeated segment observations.

    Endpoints are chosen from the robust 5–95 percentile span of all projected
    points. This averages click noise while preventing one accidental far-away
    endpoint from making the saved anchor arbitrarily long.
    """

    arrays = [np.asarray(segment, dtype=np.float64).reshape(2, 2) for segment in segments]
    if not arrays:
        raise ValueError("No hay observaciones para ajustar la línea.")
    points = np.vstack(arrays)
    center = np.median(points, axis=0)
    centered = points - center
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("No se pudo ajustar la línea multicuadro.") from error
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("Las observaciones de la línea son degeneradas.")
    direction = direction / norm
    projection = centered @ direction
    low, high = np.percentile(projection, [5.0, 95.0])
    if high - low < 8.0:
        low, high = float(np.min(projection)), float(np.max(projection))
    if high - low < 8.0:
        raise ValueError("La línea ajustada es demasiado corta.")
    return np.asarray([center + low * direction, center + high * direction], dtype=np.float32)


def observations_to_reference_segments(
    observations: Iterable[MultiframeLineObservation],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for observation in observations:
        grouped.setdefault(str(observation.name), []).append(
            np.asarray(observation.segment_reference, dtype=np.float32).reshape(2, 2)
        )
    return {name: fit_reference_segment(values) for name, values in grouped.items()}


def build_multiframe_calibration(
    observations: Iterable[MultiframeLineObservation],
    frame_width: int,
    frame_height: int,
    field_width: float = 100.0,
    field_height: float = 60.0,
    source_frame_index: int = 0,
) -> FieldCalibration:
    segments = observations_to_reference_segments(observations)
    points = {
        name: [tuple(map(float, segment[0])), tuple(map(float, segment[1]))]
        for name, segment in segments.items()
    }
    calibration = create_calibration_from_points(
        points,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        field_width=float(field_width),
        field_height=float(field_height),
        source_frame_index=int(source_frame_index),
    )
    return FieldCalibration(
        source_frame_index=calibration.source_frame_index,
        frame_width=calibration.frame_width,
        frame_height=calibration.frame_height,
        field_width=calibration.field_width,
        field_height=calibration.field_height,
        semantic_lines=calibration.semantic_lines,
        semantic_segments=calibration.semantic_segments,
        corners_image=calibration.corners_image,
        homography_image_to_field=calibration.homography_image_to_field,
        local_homography_image_to_local=calibration.local_homography_image_to_local,
        source="calibracion_asistida_multicuadro_v10",
    )


def _keyframe_gray_and_mask(frame: np.ndarray, maximum_width: int) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, float(maximum_width) / max(1.0, float(width)))
    if scale < 1.0:
        image = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image = frame
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 22, 22]), np.array([112, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([179, 95, 255]))
    mask = cv2.bitwise_or(green, white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, mask, scale


def _orb_keyframe_increment(
    previous_gray: np.ndarray,
    previous_mask: np.ndarray,
    current_gray: np.ndarray,
    current_mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray | None, int, float]:
    """Estimate current-keyframe -> previous-keyframe with bounded ORB work."""
    orb = cv2.ORB_create(
        nfeatures=650,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=15,
        fastThreshold=10,
    )
    previous_keypoints, previous_descriptors = orb.detectAndCompute(
        previous_gray, previous_mask
    )
    current_keypoints, current_descriptors = orb.detectAndCompute(
        current_gray, current_mask
    )
    if previous_descriptors is None or current_descriptors is None:
        return None, 0, float("inf")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(current_descriptors, previous_descriptors, k=2)
    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.78 * second.distance
    ]
    if len(good) < 8:
        return None, len(good), float("inf")
    good.sort(key=lambda item: (item.distance, item.queryIdx, item.trainIdx))
    good = good[:240]
    current_points = np.float32(
        [current_keypoints[item.queryIdx].pt for item in good]
    )
    previous_points = np.float32(
        [previous_keypoints[item.trainIdx].pt for item in good]
    )
    affine, inliers = FieldRegistration._deterministic_similarity(
        current_points, previous_points, threshold=3.4
    )
    if affine is None or inliers is None:
        return None, len(good), float("inf")
    small = np.eye(3, dtype=np.float64)
    small[:2, :] = affine
    projected = cv2.perspectiveTransform(
        current_points.reshape(1, -1, 2), small
    ).reshape(-1, 2)
    selector = inliers.reshape(-1).astype(bool)
    error = float(
        np.median(np.linalg.norm(projected[selector] - previous_points[selector], axis=1))
    ) if np.any(selector) else float("inf")
    to_small = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    to_full = np.linalg.inv(to_small)
    full = to_full @ small @ to_small
    if not np.isfinite(full).all():
        return None, len(good), error
    return full, int(np.count_nonzero(selector)), error


def precompute_video_registrations(
    video_path: str | Path,
    processing_max_width: int = 480,
    sample_stride: int = 10,
) -> tuple[list[int], list[np.ndarray], float, int, int]:
    """Build a fast keyframe registration cache for the interactive wizard.

    Only every ``sample_stride`` frame is decoded into the calibration timeline.
    ORB matching and a deterministic similarity fit keep the work bounded, and
    the UI intentionally navigates these keyframes instead of pretending that
    all 600 frames need a calibration pose before the user can click four lines.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    stride = max(1, int(sample_stride))
    frame_indices: list[int] = []
    matrices: list[np.ndarray] = []
    current_to_reference = np.eye(3, dtype=np.float64)
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    previous_scale = 1.0
    frame_index = 0
    cv2.setRNGSeed(10_010)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        is_keyframe = frame_index % stride == 0 or frame_index == total - 1
        if is_keyframe:
            gray, mask, scale = _keyframe_gray_and_mask(frame, int(processing_max_width))
            if previous_gray is not None and previous_mask is not None:
                increment, _inliers, error = _orb_keyframe_increment(
                    previous_gray,
                    previous_mask,
                    gray,
                    mask,
                    min(previous_scale, scale),
                )
                if increment is not None and error <= 5.0:
                    current_to_reference = current_to_reference @ increment
            frame_indices.append(frame_index)
            matrices.append(current_to_reference.copy())
            previous_gray, previous_mask, previous_scale = gray, mask, scale
            if len(frame_indices) % 12 == 0 or frame_index == total - 1:
                print(
                    f"Registro multicuadro: cuadro {frame_index}/{total - 1} "
                    f"({len(frame_indices)} keyframes)",
                    flush=True,
                )
        frame_index += 1
    capture.release()
    if not matrices:
        raise RuntimeError("El video no contiene cuadros legibles.")
    return frame_indices, matrices, fps, width, height

