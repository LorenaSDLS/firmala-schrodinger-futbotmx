from __future__ import annotations

import cv2
import numpy as np

from src.I_field_geometry.feature_constraints import CANONICAL_SEGMENTS_NORMALIZED
from src.I_field_geometry.multiframe_calibration import (
    MultiframeLineObservation,
    build_multiframe_calibration,
    fit_reference_segment,
    transform_segment,
)


def test_fit_reference_segment_averages_repeated_click_noise() -> None:
    base = np.float32([[100.0, 220.0], [700.0, 320.0]])
    segments = [
        base + np.float32([[0.0, 1.0], [0.0, -1.0]]),
        base + np.float32([[2.0, -2.0], [-2.0, 2.0]]),
        base + np.float32([[-1.0, 0.5], [1.0, -0.5]]),
    ]
    fitted = fit_reference_segment(segments)
    line = np.cross(
        np.array([*fitted[0], 1.0]),
        np.array([*fitted[1], 1.0]),
    )
    line /= np.hypot(line[0], line[1])
    distances = np.abs(base @ line[:2] + line[2])
    assert float(np.max(distances)) < 2.0
    assert float(np.linalg.norm(fitted[1] - fitted[0])) > 550.0


def test_multiframe_semantic_lines_recover_one_global_homography() -> None:
    field_to_reference = cv2.getPerspectiveTransform(
        np.float32([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        np.float32([[130.0, 150.0], [840.0, 230.0], [760.0, 930.0], [40.0, 820.0]]),
    )
    current_to_reference = [
        np.eye(3, dtype=np.float64),
        np.array([[1.0, -0.02, 18.0], [0.02, 1.0, -12.0], [0.0, 0.0, 1.0]]),
        np.array([[0.98, 0.03, -24.0], [-0.03, 0.98, 20.0], [0.0, 0.0, 1.0]]),
        np.array([[1.01, 0.01, 10.0], [-0.01, 1.01, 16.0], [0.0, 0.0, 1.0]]),
    ]
    observations: list[MultiframeLineObservation] = []
    for index, name in enumerate(("near", "far", "left", "right")):
        segment_reference = transform_segment(
            CANONICAL_SEGMENTS_NORMALIZED[name], field_to_reference
        )
        reference_to_current = np.linalg.inv(current_to_reference[index])
        segment_frame = transform_segment(segment_reference, reference_to_current)
        observations.append(
            MultiframeLineObservation(
                name=name,
                frame_index=index * 30,
                segment_frame=segment_frame,
                segment_reference=transform_segment(
                    segment_frame, current_to_reference[index]
                ),
            )
        )

    calibration = build_multiframe_calibration(
        observations,
        frame_width=900,
        frame_height=1000,
        field_width=100.0,
        field_height=60.0,
        source_frame_index=0,
    )
    assert calibration.is_complete
    assert calibration.source == "calibracion_asistida_multicuadro_v10"

    probe_field = np.float32([[[0.12, 0.18], [0.50, 0.50], [0.88, 0.76]]])
    probe_reference = cv2.perspectiveTransform(probe_field, field_to_reference)
    recovered = cv2.perspectiveTransform(
        probe_reference, calibration.homography_image_to_field
    )[0]
    expected = probe_field[0] * np.float32([100.0, 60.0])
    assert float(np.max(np.linalg.norm(recovered - expected, axis=1))) < 0.05
