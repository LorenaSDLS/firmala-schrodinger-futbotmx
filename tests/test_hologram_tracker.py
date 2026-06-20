from __future__ import annotations

import json

import cv2
import numpy as np

from src.I_field_geometry.assisted_hologram_tracker import AssistedHologramTrajectory
from src.I_field_geometry.field_spec import FieldSpec
from src.I_field_geometry.field_template import FieldTemplateConfig, build_template_points
from src.I_field_geometry.hologram_calibration import HologramCalibration, HologramKeyframe
from src.I_field_geometry.hologram_wizard import (
    HologramEditor,
    _ascii_ui,
    initial_hologram_corners,
)


def _calibration(keyframes: tuple[HologramKeyframe, ...], total: int = 6) -> HologramCalibration:
    return HologramCalibration(
        frame_width=640,
        frame_height=480,
        fps=30.0,
        total_frames=total,
        field_spec=FieldSpec(),
        keyframes=keyframes,
    )


def test_metric_template_matches_real_field_and_has_no_center_circle() -> None:
    spec = FieldSpec()
    config = FieldTemplateConfig.from_spec(spec)
    assert np.isclose(config.goal_area_depth_ratio, 25.0 / 243.0)
    assert np.isclose(config.goal_area_width_ratio, 80.0 / 182.0)
    assert not config.include_center_circle
    template = build_template_points(config, density=120)
    center = template.points[template.groups == 1]
    assert len(center) > 10
    assert np.allclose(center[:, 0], 0.5, atol=1e-6)


def test_hologram_calibration_roundtrip(tmp_path) -> None:
    corners = np.float32([[80, 60], [560, 70], [590, 430], [45, 420]])
    calibration = _calibration((HologramKeyframe(3, corners),), total=20)
    path = calibration.save(tmp_path / "hologram.json")
    loaded = HologramCalibration.load(path)
    assert loaded.field_width == 243.0
    assert loaded.field_height == 182.0
    assert loaded.keyframes[0].frame_index == 3
    assert np.allclose(loaded.keyframes[0].corners_image, corners)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["calibration_type"] == "assisted_hologram"


def test_single_anchor_is_propagated_by_camera_motion() -> None:
    corners = np.float32([[80, 60], [560, 60], [560, 420], [80, 420]])
    calibration = _calibration((HologramKeyframe(0, corners),), total=6)
    matrices = []
    for frame in range(6):
        # Current frame is shifted +8 px in x relative to the reference.
        matrices.append(np.array([[1.0, 0.0, -8.0 * frame], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    trajectory = AssistedHologramTrajectory(
        calibration,
        np.stack(matrices),
        np.full(6, 0.9, dtype=np.float32),
        np.ones(6, dtype=np.uint8),
    )
    pose = trajectory.pose(5)
    assert pose.trusted
    assert np.allclose(pose.corners_image[:, 0], corners[:, 0] + 40.0, atol=0.5)
    result = trajectory.geometry_result(5)
    assert result.geometry_state == "global"
    assert result.trusted


def test_two_manual_anchors_close_drift_smoothly() -> None:
    first = np.float32([[60, 60], [500, 70], [540, 410], [40, 400]])
    second = first + np.float32([100.0, 15.0])
    calibration = _calibration(
        (HologramKeyframe(0, first), HologramKeyframe(5, second)),
        total=6,
    )
    # Deliberately imperfect camera trajectory predicts only +80 px by frame 5.
    matrices = np.stack(
        [np.array([[1.0, 0.0, -16.0 * i], [0.0, 1.0, -2.0 * i], [0.0, 0.0, 1.0]]) for i in range(6)]
    )
    trajectory = AssistedHologramTrajectory(
        calibration,
        matrices,
        np.full(6, 0.85, dtype=np.float32),
        np.ones(6, dtype=np.uint8),
    )
    assert np.allclose(trajectory.pose(0).corners_image, first)
    assert np.allclose(trajectory.pose(5).corners_image, second)
    middle = trajectory.pose(3).corners_image
    assert middle is not None
    assert first[:, 0].mean() < middle[:, 0].mean() < second[:, 0].mean()


def test_long_occlusion_is_marked_lost_without_a_future_anchor() -> None:
    corners = np.float32([[80, 60], [560, 60], [560, 420], [80, 420]])
    calibration = _calibration((HologramKeyframe(0, corners),), total=120)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 120, axis=0)
    qualities = np.zeros(120, dtype=np.float32)
    qualities[0] = 1.0
    updated = np.zeros(120, dtype=np.uint8)
    updated[0] = 1
    trajectory = AssistedHologramTrajectory(calibration, matrices, qualities, updated)
    assert trajectory.pose(110).state == "lost"
    assert not trajectory.geometry_result(110).trusted


def test_initial_guess_uses_green_surface() -> None:
    frame = np.full((480, 640, 3), 230, dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.int32([[90, 90], [550, 110], [600, 430], [40, 420]]),
        (40, 160, 70),
    )
    corners = initial_hologram_corners(frame)
    assert corners.shape == (4, 2)
    assert abs(cv2.contourArea(corners.astype(np.float32))) > 0.2 * 640 * 480


def test_hologram_ui_converts_unsupported_unicode_to_ascii() -> None:
    assert _ascii_ui("botón, predicción, tamaño ± 2 cm") == "boton, prediccion, tamano +/- 2 cm"


def test_hologram_video_can_zoom_far_out() -> None:
    editor = HologramEditor.__new__(HologramEditor)
    editor.width = 1360
    editor.height = 1808
    editor.video_zoom = 0.78
    for _ in range(20):
        editor.zoom_video(1.0 / 1.10)
    transform = editor._canvas_transform()
    assert editor.video_zoom < 0.20
    assert transform.scale > 0.0
    assert editor.height * transform.scale < 0.25 * 1000
