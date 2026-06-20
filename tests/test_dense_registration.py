from __future__ import annotations

import cv2
import numpy as np


def _synthetic_field(width: int = 640, height: int = 360) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (80, 70), (560, 300), 255, cv2.FILLED)
    frame[mask > 0] = (65, 145, 82)
    return frame, mask


def test_adaptive_paint_detector_keeps_line_outside_strict_green_mask():
    from src.I_field_geometry.visual_evidence import AdaptiveFieldEvidenceExtractor

    frame, mask = _synthetic_field()
    # The white paint replaces green pixels, so a semantic segmenter can omit it
    # from the surface class. It remains close enough to be physical field paint.
    cv2.line(frame, (150, 60), (500, 60), (245, 245, 245), 6, cv2.LINE_AA)
    mask[54:68, 140:510] = 0

    evidence = AdaptiveFieldEvidenceExtractor().extract(frame, mask, [])

    assert np.count_nonzero(evidence.marking_mask[54:68, 140:510]) > 1000
    assert len(evidence.marking_lines) >= 1
    assert evidence.marking_pixel_fraction > 0.0


def test_temporal_evidence_combines_static_markings_from_multiple_frames():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    first, mask = _synthetic_field()
    second, _ = _synthetic_field()
    cv2.line(first, (200, 75), (200, 295), (240, 240, 240), 6, cv2.LINE_AA)
    cv2.line(second, (400, 75), (400, 295), (240, 240, 240), 6, cv2.LINE_AA)

    estimator = FieldGeometryEstimator(first.shape[1], first.shape[0])
    estimator._update_temporal_marking_evidence(
        first, mask, [], np.eye(3), registration_quality=1.0, registration_updated=True
    )
    combined = estimator._update_temporal_marking_evidence(
        second, mask, [], np.eye(3), registration_quality=1.0, registration_updated=True
    )

    assert estimator.temporal_evidence_frames == 2
    assert combined is not None
    assert np.count_nonzero(combined[:, 195:206]) > 500
    assert np.count_nonzero(combined[:, 395:406]) > 500


def test_unity_exporter_omits_points_without_trusted_global_homography():
    from src.E_events.unity_exporter import normalize_track_point, normalize_tracks

    invalid = {
        "x_px": 320.0,
        "y_px": 240.0,
        "field_transform_valid": False,
        "field_x_norm": 0.5,
        "field_y_norm": 0.5,
    }
    assert normalize_track_point(invalid, 640.0, 480.0) is None

    tracks = {
        "robots": {"blue_1": [invalid]},
        "ball": [invalid],
        "goals": {},
    }
    normalized = normalize_tracks(tracks, 640.0, 480.0)
    assert normalized["robots"]["blue_1"] == []
    assert normalized["ball"] == []


def test_unity_exporter_marks_only_trusted_v9_coordinates():
    from src.E_events.unity_exporter import normalize_track_point

    point = normalize_track_point(
        {
            "field_transform_valid": True,
            "field_x_norm": 0.31,
            "field_y_norm": 0.72,
            "visible": True,
        },
        640.0,
        480.0,
    )

    assert point is not None
    assert point["x_norm"] == 0.31
    assert point["y_norm"] == 0.72
    assert point["coordinate_source"] == "trusted_field_homography_v10"


def test_v9_evidence_debug_renderer_has_requested_video_shape():
    from src.I_field_geometry.field_geometry import (
        FieldGeometryEstimator,
        draw_field_evidence_debug,
    )

    frame, mask = _synthetic_field(320, 180)
    cv2.line(frame, (60, 50), (260, 50), (245, 245, 245), 4, cv2.LINE_AA)
    estimator = FieldGeometryEstimator(320, 180)
    evidence = estimator.template_registrar.evidence_extractor.extract(frame, mask, [])
    debug = {
        "marking_mask": evidence.marking_mask,
        "boundary_mask": evidence.boundary_mask,
        "marking_lines": [item.segment for item in evidence.marking_lines],
        "side_lines": [item.segment for item in evidence.boundary_lines],
        "candidate_corners": [],
        "source": "prueba_v9",
    }

    rendered = draw_field_evidence_debug(frame, estimator.last_result, debug)
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8


def test_strong_portrait_perspective_keeps_converging_sidelines():
    """Regression: valid field-line families may differ by more than 48°."""
    from src.I_field_geometry.field_template import build_template_points
    from src.I_field_geometry.template_registration import GoalAnchoredTemplateRegistrar

    width, height = 800, 1000
    frame = np.full((height, width, 3), 45, dtype=np.uint8)
    # Registrar order: near-left, far-left, far-right, near-right. One near
    # corner is far outside the image, matching a close portrait camera.
    expected = np.float32([[-1500, 950], [350, 250], [700, 300], [520, 950]])
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), expected
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(expected).astype(np.int32), 255)
    frame[mask > 0] = (70, 150, 90)

    template = build_template_points(density=300)
    projected = cv2.perspectiveTransform(
        template.points.reshape(1, -1, 2), field_to_image
    ).reshape(-1, 2)
    for index in range(1, len(projected)):
        if template.groups[index] != template.groups[index - 1]:
            continue
        if np.linalg.norm(projected[index] - projected[index - 1]) > 150:
            continue
        cv2.line(
            frame,
            tuple(np.rint(projected[index - 1]).astype(int)),
            tuple(np.rint(projected[index]).astype(int)),
            (245, 245, 245),
            5,
            cv2.LINE_AA,
        )
    cv2.polylines(frame, [np.rint(expected).astype(np.int32)], True, (25, 25, 25), 14)
    cv2.polylines(frame, [np.rint(expected).astype(np.int32)], True, (245, 245, 245), 5)

    far_center = cv2.perspectiveTransform(
        np.float32([[[1.0, 0.5]]]), field_to_image
    )[0, 0]
    x, y = far_center
    goals = [
        {
            "class_group": "goal",
            "bbox_xyxy": [x - 110, y - 35, x + 110, y + 35],
            "confidence": 0.95,
        }
    ]

    result = GoalAnchoredTemplateRegistrar(
        width, height, processing_max_dimension=800
    ).register(
        frame,
        mask,
        goals,
        exclusion_boxes=[],
        temporal_evidence_frames=4,
    )

    assert result.valid and result.trusted
    assert result.corners_image is not None
    mean_error = float(np.mean(np.linalg.norm(result.corners_image - expected, axis=1)))
    assert mean_error < 12.0
    near_width = float(np.linalg.norm(result.corners_image[3] - result.corners_image[0]))
    far_width = float(np.linalg.norm(result.corners_image[2] - result.corners_image[1]))
    assert near_width > far_width
