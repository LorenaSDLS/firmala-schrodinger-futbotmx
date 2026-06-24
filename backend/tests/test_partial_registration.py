from __future__ import annotations

import json

import cv2
import numpy as np

from src.I_field_geometry.calibration import FieldCalibration, create_calibration_from_points
from src.I_field_geometry.field_geometry import FieldGeometryResult, render_rectified_debug
from src.I_field_geometry.field_segmenter import FieldMaskResult


def test_partial_calibration_accepts_only_visible_lines(tmp_path):
    calibration = create_calibration_from_points(
        {
            "near": [(80.0, 420.0), (520.0, 455.0)],
            "right": [(520.0, 455.0), (610.0, 120.0)],
            "near_area": [(125.0, 360.0), (490.0, 390.0)],
        },
        frame_width=640,
        frame_height=480,
        source_frame_index=30,
    )
    assert not calibration.is_complete
    assert calibration.homography_image_to_field is None
    assert calibration.corners_image is None
    assert calibration.feature_count == 3

    path = calibration.save(tmp_path / "partial.json")
    loaded = FieldCalibration.load(path)
    assert not loaded.is_complete
    assert set(loaded.semantic_lines) == {"near", "right", "near_area"}
    assert loaded.source_frame_index == 30
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 3
    assert payload["complete"] is False


def test_complete_calibration_remains_backward_compatible():
    calibration = create_calibration_from_points(
        {
            "near": [(70.0, 440.0), (540.0, 450.0)],
            "far": [(210.0, 110.0), (470.0, 120.0)],
            "left": [(70.0, 440.0), (210.0, 110.0)],
            "right": [(540.0, 450.0), (470.0, 120.0)],
        },
        frame_width=640,
        frame_height=480,
    )
    assert calibration.is_complete
    assert calibration.homography_image_to_field is not None
    assert calibration.corners_image is not None


def test_rectified_debug_uses_only_segmented_visible_patch():
    width, height = 640, 480
    frame = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    quad = np.float32([[80, 430], [220, 100], [590, 150], [560, 455]])
    cv2.fillConvexPoly(frame, quad.astype(np.int32), (70, 155, 95))
    cv2.line(frame, (100, 390), (545, 410), (245, 245, 245), 7)

    # Only the right/near part of the visible surface is supported, simulating
    # a crop where most of the canonical field lies outside the camera.
    mask = np.zeros((height, width), dtype=np.uint8)
    partial_polygon = np.float32([[270, 430], [350, 170], [590, 150], [560, 455]])
    cv2.fillConvexPoly(mask, partial_polygon.astype(np.int32), 255)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.98,
        class_id=0,
        class_name="field_surface",
        bbox_xyxy=[270, 150, 590, 455],
        polygon=partial_polygon,
        coverage=float(np.count_nonzero(mask) / mask.size),
    )

    canonical = np.float32([[100, 0], [100, 60], [0, 60], [0, 0]])
    image_to_field = cv2.getPerspectiveTransform(quad, canonical)
    field_to_image = np.linalg.inv(image_to_field)
    geometry = FieldGeometryResult(
        valid=True,
        trusted=True,
        measured=True,
        propagated=False,
        confidence=0.91,
        corners_image=quad,
        homography_image_to_field=image_to_field,
        homography_field_to_image=field_to_image,
        mask_coverage=segmentation.coverage,
        source="test_partial",
        line_support={},
        side_visible={"far": True, "right": True, "near": False, "left": False},
        registration_scope="partial",
        geometry_state="global",
        field_width=100.0,
        field_height=60.0,
    )

    output = render_rectified_debug(
        frame,
        geometry,
        detections=[],
        segmentation=segmentation,
        output_width=1000,
        output_height=600,
    )
    assert output.shape == (600, 1200, 3)
    # V7 should produce a schematic/visible crop, not an almost-black full warp.
    dark_fraction = float(np.mean(np.max(output, axis=2) < 18))
    assert dark_fraction < 0.10
    # Status bar should not make the rest of the image blank.
    assert float(np.mean(output[80:])) > 35.0
