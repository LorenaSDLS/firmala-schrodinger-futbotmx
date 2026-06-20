from __future__ import annotations

import cv2
import numpy as np

from src.I_field_geometry.calibration import create_calibration_from_points
from src.I_field_geometry.feature_constraints import (
    homography_from_semantic_lines,
    line_from_segment,
    score_manual_anchors,
)
from src.I_field_geometry.field_geometry import FieldGeometryEstimator
from src.I_field_geometry.field_segmenter import FieldMaskResult


def _project(segment: np.ndarray, h: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(segment, dtype=np.float32).reshape(1, 2, 2), h
    ).reshape(2, 2)


def test_partial_manual_features_create_only_local_orientation():
    calibration = create_calibration_from_points(
        {
            "near": [(80.0, 420.0), (520.0, 455.0)],
            "right": [(520.0, 455.0), (610.0, 120.0)],
        },
        frame_width=640,
        frame_height=480,
    )
    assert calibration.has_local_registration
    assert not calibration.has_global_registration
    assert calibration.homography_image_to_field is None


def test_two_by_two_semantic_grid_solves_global_without_outer_four():
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.float32([[90, 430], [230, 90], [590, 130], [555, 455]]),
    )
    canonical = {
        "near": np.float32([[0.0, 0.0], [0.0, 1.0]]),
        "center": np.float32([[0.5, 0.0], [0.5, 1.0]]),
        "left": np.float32([[0.0, 0.0], [1.0, 0.0]]),
        "right": np.float32([[0.0, 1.0], [1.0, 1.0]]),
    }
    points = {
        name: [tuple(point) for point in _project(segment, field_to_image)]
        for name, segment in canonical.items()
    }
    calibration = create_calibration_from_points(
        points,
        frame_width=640,
        frame_height=480,
        field_width=100.0,
        field_height=60.0,
    )
    assert calibration.has_global_registration
    assert calibration.source == "calibracion_asistida_anclas_duras_v8"
    expected = np.float32([[0.25, 0.35]])
    image_point = cv2.perspectiveTransform(expected.reshape(1, 1, 2), field_to_image)
    field_point = cv2.perspectiveTransform(image_point, calibration.homography_image_to_field)[0, 0]
    np.testing.assert_allclose(field_point, [25.0, 21.0], atol=0.2)


def test_hard_anchor_rejects_semantically_wrong_candidate():
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.float32([[80, 420], [220, 90], [590, 130], [555, 450]]),
    )
    near = _project(np.float32([[0, 0], [0, 1]]), field_to_image)
    # Deliberately label the near line as the center line.
    reports = score_manual_anchors(
        {"center": near},
        field_to_image,
        frame_diagonal=float(np.hypot(640, 480)),
    )
    assert len(reports) == 1
    assert not reports[0].hard_pass
    assert reports[0].canonical_error > 0.20


def test_semantic_line_solver_uses_feature_identity():
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
        np.float32([[70, 440], [210, 100], [600, 140], [550, 460]]),
    )
    semantic_segments = {
        "near": _project(np.float32([[0, 0], [0, 1]]), field_to_image),
        "far_area": _project(np.float32([[0.82, 0.22], [0.82, 0.78]]), field_to_image),
        "left": _project(np.float32([[0, 0], [1, 0]]), field_to_image),
        "right": _project(np.float32([[0, 1], [1, 1]]), field_to_image),
    }
    lines = {name: line_from_segment(segment) for name, segment in semantic_segments.items()}
    solved = homography_from_semantic_lines(lines)
    assert solved is not None
    image_to_field, recovered_field_to_image = solved
    test_points = np.float32([[0.1, 0.2], [0.7, 0.8]])
    image = cv2.perspectiveTransform(test_points.reshape(1, -1, 2), field_to_image)
    recovered = cv2.perspectiveTransform(image, image_to_field).reshape(-1, 2)
    np.testing.assert_allclose(recovered, test_points, atol=2e-3)
    np.testing.assert_allclose(
        recovered_field_to_image / recovered_field_to_image[2, 2],
        field_to_image / field_to_image[2, 2],
        atol=3e-2,
    )


def test_clipped_surface_stays_local_and_never_exports_coordinates():
    width, height = 640, 480
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    polygon = np.float32([[0, 105], [620, 150], [639, 479], [0, 479]])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    frame[mask > 0] = (70, 150, 90)
    cv2.line(frame, (30, 150), (610, 190), (245, 245, 245), 6)
    cv2.line(frame, (420, 150), (610, 470), (245, 245, 245), 6)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.98,
        class_id=0,
        class_name="field_surface",
        bbox_xyxy=[0, 105, 639, 479],
        polygon=polygon,
        coverage=float(np.count_nonzero(mask) / mask.size),
    )
    estimator = FieldGeometryEstimator(width, height)
    result = estimator.update(segmentation, np.eye(3), frame=frame, goal_detections=[])
    annotated = estimator.annotate_detection({"class_group": "ball", "bbox_xyxy": [300, 240, 315, 255]})
    assert result.geometry_state in {"local", "surface"}
    assert not result.valid
    assert result.homography_image_to_field is None
    assert not annotated["field_transform_valid"]
    assert "field_x_norm" not in annotated


def test_local_orientation_propagates_without_new_segmentation():
    width, height = 320, 240
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    polygon = np.float32([[0, 50], [300, 70], [319, 239], [0, 239]])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    frame[mask > 0] = (70, 150, 90)
    cv2.line(frame, (10, 65), (290, 85), (245, 245, 245), 5)
    cv2.line(frame, (220, 70), (300, 225), (245, 245, 245), 5)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.98,
        class_id=0,
        class_name="field_surface",
        bbox_xyxy=[0, 50, 319, 239],
        polygon=polygon,
        coverage=float(np.count_nonzero(mask) / mask.size),
    )
    estimator = FieldGeometryEstimator(width, height)
    first = estimator.update(segmentation, np.eye(3), frame=frame, goal_detections=[])
    assert first.geometry_state == "local"
    motion = np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    second = estimator.update(None, motion, frame=frame, goal_detections=[])
    assert second.geometry_state == "local"
    assert second.propagated
    assert second.local_homography_image_to_local is not None
