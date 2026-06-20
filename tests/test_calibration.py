from __future__ import annotations

import json

import cv2
import numpy as np

from src.I_field_geometry.calibration import (
    FieldCalibration,
    create_calibration_from_points,
)
from src.I_field_geometry.field_geometry import FieldGeometryEstimator


def test_line_calibration_infers_offscreen_corners_and_roundtrips(tmp_path):
    points = {
        "near": [(20.0, 370.0), (210.0, 520.0)],
        "far": [(330.0, 95.0), (580.0, 145.0)],
        "left": [(20.0, 370.0), (330.0, 95.0)],
        "right": [(210.0, 520.0), (580.0, 145.0)],
    }
    calibration = create_calibration_from_points(
        points,
        frame_width=640,
        frame_height=480,
        field_width=100.0,
        field_height=60.0,
        source_frame_index=12,
    )
    assert calibration.corners_image.shape == (4, 2)
    canonical = cv2.perspectiveTransform(
        calibration.corners_image.reshape(1, -1, 2),
        calibration.homography_image_to_field,
    ).reshape(-1, 2)
    np.testing.assert_allclose(
        canonical,
        np.array([[100, 0], [100, 60], [0, 60], [0, 0]], dtype=np.float32),
        atol=1e-3,
    )

    path = tmp_path / "field_calibration.json"
    calibration.save(path)
    loaded = FieldCalibration.load(path)
    np.testing.assert_allclose(
        loaded.homography_image_to_field,
        calibration.homography_image_to_field,
    )
    assert loaded.source_frame_index == 12


def test_assisted_calibration_activates_and_propagates(tmp_path):
    points = {
        "near": [(35.0, 390.0), (260.0, 510.0)],
        "far": [(360.0, 90.0), (590.0, 145.0)],
        "left": [(35.0, 390.0), (360.0, 90.0)],
        "right": [(260.0, 510.0), (590.0, 145.0)],
    }
    calibration = create_calibration_from_points(
        points,
        frame_width=640,
        frame_height=480,
        source_frame_index=2,
    )
    path = calibration.save(tmp_path / "calibration.json")
    estimator = FieldGeometryEstimator(
        640,
        480,
        calibration_path=path,
    )

    before = estimator.update(
        segmentation=None,
        current_to_reference=np.eye(3),
        frame_index=0,
    )
    assert not before.trusted

    current_to_reference = np.array(
        [[1.0, 0.0, -4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    activated = estimator.update(
        segmentation=None,
        current_to_reference=current_to_reference,
        frame_index=2,
    )
    assert activated.trusted
    assert activated.measured
    assert activated.source == "calibracion_asistida_anclas_duras_v8"

    next_transform = np.array(
        [[1.0, 0.0, -7.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    propagated = estimator.update(
        segmentation=None,
        current_to_reference=next_transform,
        frame_index=3,
    )
    assert propagated.trusted
    assert propagated.propagated
    assert propagated.source == "calibracion_asistida_propagada"
    assert propagated.homography_image_to_field is not None


def test_calibration_scales_to_new_resolution():
    points = {
        "near": [(20.0, 180.0), (120.0, 240.0)],
        "far": [(170.0, 40.0), (300.0, 70.0)],
        "left": [(20.0, 180.0), (170.0, 40.0)],
        "right": [(120.0, 240.0), (300.0, 70.0)],
    }
    calibration = create_calibration_from_points(
        points,
        frame_width=320,
        frame_height=240,
    )
    scaled = calibration.scaled_to(640, 480)
    np.testing.assert_allclose(scaled.corners_image, calibration.corners_image * 2.0)
    transformed = cv2.perspectiveTransform(
        scaled.corners_image.reshape(1, -1, 2),
        scaled.homography_image_to_field,
    ).reshape(-1, 2)
    np.testing.assert_allclose(
        transformed,
        np.array([[100, 0], [100, 60], [0, 60], [0, 0]], dtype=np.float32),
        atol=1e-3,
    )


def test_multiline_template_solver_recovers_clipped_projective_field():
    from src.I_field_geometry.field_template import build_template_points
    from src.I_field_geometry.template_registration import GoalAnchoredTemplateRegistrar

    width, height = 800, 600
    frame = np.full((height, width, 3), 42, dtype=np.uint8)
    # Registrar order: near-left, far-left, far-right, near-right.
    expected = np.float32([[70, 540], [340, 110], [710, 175], [650, 560]])
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), expected
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, expected.astype(np.int32), 255)
    frame[mask > 0] = (75, 155, 95)

    template = build_template_points(density=240)
    projected = cv2.perspectiveTransform(
        template.points.reshape(1, -1, 2), field_to_image
    ).reshape(-1, 2)
    for index in range(1, len(projected)):
        if template.groups[index] != template.groups[index - 1]:
            continue
        if np.linalg.norm(projected[index] - projected[index - 1]) > 80:
            continue
        cv2.line(
            frame,
            tuple(np.rint(projected[index - 1]).astype(int)),
            tuple(np.rint(projected[index]).astype(int)),
            (245, 245, 245),
            5,
            cv2.LINE_AA,
        )
    cv2.polylines(frame, [expected.astype(np.int32)], True, (25, 25, 25), 14)
    cv2.polylines(frame, [expected.astype(np.int32)], True, (245, 245, 245), 5)

    near_center = cv2.perspectiveTransform(
        np.float32([[[0.0, 0.5]]]), field_to_image
    )[0, 0]
    far_center = cv2.perspectiveTransform(
        np.float32([[[1.0, 0.5]]]), field_to_image
    )[0, 0]
    goals = []
    for center, size in ((near_center, (100, 130)), (far_center, (75, 75))):
        x, y = center
        box_width, box_height = size
        goals.append(
            {
                "class_group": "goal",
                "bbox_xyxy": [
                    x - box_width / 2,
                    y - box_height / 2,
                    x + box_width / 2,
                    y + box_height / 2,
                ],
                "confidence": 0.95,
            }
        )

    registrar = GoalAnchoredTemplateRegistrar(width, height, processing_max_dimension=800)
    result = registrar.register(frame, mask, goals, exclusion_boxes=[])
    assert result.valid
    assert result.trusted
    assert result.source.startswith("plantilla_semantica")
    assert result.corners_image is not None
    mean_error = float(
        np.mean(np.linalg.norm(result.corners_image - expected, axis=1))
    )
    assert mean_error < 18.0


def test_camera_registration_accepts_explicit_frame_index():
    from src.F_simulation.field_registration import FieldRegistration

    registration = FieldRegistration(64, 48, enabled=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    result = registration.update(frame, frame_index=17)

    assert result.valid
    assert registration.frame_index == 17
