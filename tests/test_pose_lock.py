from __future__ import annotations

import cv2
import numpy as np


def _surface(width: int = 640, height: int = 360):
    from src.I_field_geometry.field_segmenter import FieldMaskResult

    frame = np.full((height, width, 3), 28, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (70, 55), (570, 310), 255, cv2.FILLED)
    frame[mask > 0] = (65, 145, 82)
    polygon = np.float32([[70, 55], [570, 55], [570, 310], [70, 310]])
    result = FieldMaskResult(
        mask=mask,
        confidence=0.95,
        class_id=0,
        class_name="field",
        bbox_xyxy=[70.0, 55.0, 570.0, 310.0],
        polygon=polygon,
        coverage=float(np.count_nonzero(mask)) / mask.size,
    )
    return frame, mask, result


def _template_result(corners: np.ndarray, *, trusted: bool, local=None):
    from src.I_field_geometry.template_registration import TemplateRegistrationResult

    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    field_to_image = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners
    )
    image_to_field = np.linalg.inv(field_to_image)
    return TemplateRegistrationResult(
        valid=trusted,
        trusted=trusted,
        confidence=0.86 if trusted else 0.34,
        corners_image=corners if trusted else None,
        homography_image_to_field_normalized=image_to_field if trusted else None,
        homography_field_to_image_normalized=field_to_image if trusted else None,
        source="synthetic_v10",
        template_score=0.72,
        mask_score=0.82,
        goal_score=0.48,
        rail_score=0.72,
        visible_template_fraction=0.55,
        registration_scope="full" if trusted else "local",
        geometry_state="global" if trusted else "local",
        local_homography_image_to_local=local,
        feature_match_score=0.78,
        feature_match_count=5,
        feature_matches={"near": 0.8, "far": 0.8, "left": 0.8, "right": 0.8},
        reverse_template_score=0.55,
        boundary_alignment_score=0.68,
        candidate_margin=0.05,
        temporal_evidence_frames=4,
        marking_pixel_fraction=0.01,
        candidate_count=12,
        physical_boundary_score=0.74,
        physical_boundary_count=2,
        physical_boundary_scores={"left_long": 0.75, "right_long": 0.73},
    )


def test_directional_boundary_requires_field_to_exterior_transition():
    from src.I_field_geometry.template_registration import GoalAnchoredTemplateRegistrar

    frame, mask, _ = _surface()
    true_quad = np.float32([[70, 55], [570, 55], [570, 310], [70, 310]])
    false_inner_quad = np.float32([[170, 110], [470, 110], [470, 260], [170, 260]])

    true_score, true_count, _ = GoalAnchoredTemplateRegistrar._directional_boundary_evidence(
        frame, mask, true_quad
    )
    false_score, false_count, _ = GoalAnchoredTemplateRegistrar._directional_boundary_evidence(
        frame, mask, false_inner_quad
    )

    assert true_count >= 2
    assert true_score >= 0.58
    assert false_count == 0
    assert false_score < 0.20


def test_global_pose_needs_three_consistent_measurements(monkeypatch):
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, _mask, segmentation = _surface()
    estimator = FieldGeometryEstimator(frame.shape[1], frame.shape[0])
    # Registrar order: near-left, far-left, far-right, near-right.
    corners = np.float32([[70, 310], [70, 55], [570, 55], [570, 310]])
    synthetic = _template_result(corners, trusted=True)
    monkeypatch.setattr(estimator.template_registrar, "register", lambda **_: synthetic)

    first = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=0,
    )
    second = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=1,
    )
    third = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=2,
    )

    assert not first.trusted and first.pose_candidate_streak == 1
    assert not second.trusted and second.pose_candidate_streak == 2
    assert third.trusted and third.geometry_state == "global"
    assert third.pose_admission_state == "global_acquired"


def test_local_pose_is_locked_against_new_inconsistent_measurements(monkeypatch):
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, _mask, segmentation = _surface()
    estimator = FieldGeometryEstimator(frame.shape[1], frame.shape[0])
    first_local = np.array(
        [[0.0, -1.0, 360.0], [1.0, 0.0, -80.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    second_local = np.array(
        [[1.0, 0.0, -250.0], [0.0, 1.0, 500.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    results = iter(
        [
            _template_result(np.zeros((4, 2)), trusted=False, local=first_local),
            _template_result(np.zeros((4, 2)), trusted=False, local=second_local),
        ]
    )
    monkeypatch.setattr(estimator.template_registrar, "register", lambda **_: next(results))

    first = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=0,
    )
    second = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=1,
    )

    assert first.geometry_state == "local"
    assert second.geometry_state == "local"
    assert np.allclose(first.local_homography_image_to_local, first_local)
    assert np.allclose(second.local_homography_image_to_local, first_local)
    assert not np.allclose(second.local_homography_image_to_local, second_local)
    assert second.local_lock_active


def test_temporal_consensus_forgets_one_frame_false_marking():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, mask, _ = _surface()
    false_frame = frame.copy()
    cv2.line(false_frame, (210, 70), (210, 295), (245, 245, 245), 7, cv2.LINE_AA)
    estimator = FieldGeometryEstimator(frame.shape[1], frame.shape[0])

    estimator._update_temporal_marking_evidence(
        false_frame, mask, [], np.eye(3), 1.0, True
    )
    one_miss = estimator._update_temporal_marking_evidence(
        frame, mask, [], np.eye(3), 1.0, True
    )
    estimator._update_temporal_marking_evidence(
        frame, mask, [], np.eye(3), 1.0, True
    )
    three_misses = estimator._update_temporal_marking_evidence(
        frame, mask, [], np.eye(3), 1.0, True
    )

    assert one_miss is not None and three_misses is not None
    assert np.count_nonzero(one_miss[80:290, 204:217]) > 100
    assert np.count_nonzero(three_misses[80:290, 204:217]) < 30


def test_global_propagation_keeps_global_scope_and_state():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    estimator = FieldGeometryEstimator(640, 360)
    corners = np.float32([[70, 55], [570, 55], [570, 310], [70, 310]])
    homography = cv2.getPerspectiveTransform(corners, estimator.canonical_corners)
    estimator.reference_to_field = homography.copy()
    estimator._trusted_result_from_corners(
        corners, homography, "seed", measured=True, confidence=0.9
    )
    result = estimator.update(
        segmentation=None,
        current_to_reference=np.eye(3),
        frame=None,
        registration_quality=0.9,
        registration_updated=True,
        frame_index=1,
    )

    assert result.trusted
    assert result.geometry_state == "global"
    assert result.registration_scope == "full"
    assert result.source == "global_propagada_v10"


def test_future_partial_calibration_does_not_create_a_pre_activation_jump(tmp_path, monkeypatch):
    from src.I_field_geometry.calibration import create_calibration_from_points
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, _mask, segmentation = _surface()
    calibration = create_calibration_from_points(
        {"near": [(120.0, 220.0), (520.0, 250.0)]},
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        source_frame_index=2,
    )
    path = calibration.save(tmp_path / "field_calibration.json")
    estimator = FieldGeometryEstimator(frame.shape[1], frame.shape[0], calibration_path=path)
    automatic_local = np.array(
        [[1.0, 0.0, 80.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    synthetic = _template_result(np.zeros((4, 2)), trusted=False, local=automatic_local)
    monkeypatch.setattr(estimator.template_registrar, "register", lambda **_: synthetic)

    before = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=0,
    )
    still_before = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=1,
    )
    activated = estimator.update(
        segmentation, np.eye(3), frame=frame, registration_quality=1.0,
        registration_updated=True, frame_index=2,
    )

    assert before.geometry_state == "surface"
    assert still_before.geometry_state == "surface"
    assert before.local_homography_image_to_local is None
    assert not before.local_lock_active
    assert activated.geometry_state == "local"
    assert activated.local_lock_active
    assert activated.source == "calibracion_asistida_local_v10"


def test_deterministic_similarity_rejects_large_flow_outliers():
    from src.F_simulation.field_registration import FieldRegistration

    rng = np.random.default_rng(10)
    source = rng.uniform([20.0, 20.0], [620.0, 340.0], size=(80, 2))
    angle = np.deg2rad(2.5)
    scale = 1.012
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    destination = scale * (source @ rotation.T) + np.array([7.0, -5.0])
    destination += rng.normal(0.0, 0.18, destination.shape)
    destination[-12:] += rng.uniform(40.0, 120.0, size=(12, 2))

    affine, inliers = FieldRegistration._deterministic_similarity(
        source.astype(np.float32), destination.astype(np.float32)
    )
    assert affine is not None and inliers is not None
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine
    projected = cv2.perspectiveTransform(
        source[:-12].astype(np.float32).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)
    error = np.linalg.norm(projected - destination[:-12], axis=1)
    assert float(np.median(error)) < 0.5
    assert int(np.count_nonzero(inliers)) >= 60
