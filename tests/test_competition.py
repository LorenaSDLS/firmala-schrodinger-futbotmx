from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _record(frame_index: int, detections: list[dict]) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 30.0,
        "detections": detections,
    }


def test_ball_field_coordinates_reach_mesa_export(tmp_path: Path):
    from src.E_events.event_detector import generate_events
    from src.E_events.unity_exporter import normalize_tracks

    path = tmp_path / "quick_detections.jsonl"
    ball = {
        "class_name": "ball",
        "class_group": "ball",
        "confidence": 0.91,
        "bbox_xyxy": [100, 120, 120, 140],
        "field_x": 120.0,
        "field_y": 90.0,
        "field_x_norm": 0.493827,
        "field_y_norm": 0.494505,
        "inside_surface": True,
        "field_transform_valid": True,
        "field_transform_confidence": 0.93,
        "field_transform_source": "holograma_v11_tracking",
    }
    path.write_text(json.dumps(_record(0, [ball])) + "\n", encoding="utf-8")
    _, _, tracks = generate_events(path)
    assert tracks["ball"][0]["field_transform_valid"] is True
    assert tracks["ball"][0]["field_x_norm"] == ball["field_x_norm"]
    normalized = normalize_tracks(tracks, 640, 480)
    assert len(normalized["ball"]) == 1
    assert normalized["ball"][0]["x_norm"] == ball["field_x_norm"]


def test_recovery_and_reactivation_events_are_emitted():
    from src.E_events.event_detector import EventDetector

    detector = EventDetector()
    robot = {
        "class_name": "robot",
        "class_group": "robot",
        "tracking_id": 0,
        "confidence": 0.9,
        "bbox_xyxy": [100, 100, 150, 160],
    }
    ball = {
        "class_name": "ball",
        "class_group": "ball",
        "confidence": 0.9,
        "bbox_xyxy": [200, 200, 215, 215],
    }
    detector.process_frame_record(_record(0, [robot, ball]))
    for frame in range(1, 18):
        detector.process_frame_record(_record(frame, []))
    detector.process_frame_record(_record(18, [robot, ball]))
    types = [event.event_type for event in detector.events]
    assert "robot_inactive_candidate" in types
    assert "robot_reactivated" in types
    assert "ball_missing_candidate" in types
    assert "ball_recovered" in types


def test_penalty_entry_uses_metric_hologram_coordinates():
    from src.E_events.event_detector import EventDetector

    detector = EventDetector()
    detection = {
        "class_name": "robot",
        "class_group": "robot",
        "tracking_id": 0,
        "confidence": 0.9,
        "bbox_xyxy": [100, 100, 150, 160],
        "field_x_norm": 0.04,
        "field_y_norm": 0.50,
        "field_transform_valid": True,
        "field_transform_confidence": 0.92,
    }
    for frame in range(3):
        detector.process_frame_record(_record(frame, [detection]))
    events = [event for event in detector.events if event.event_type == "robot_entered_penalty_area"]
    assert len(events) == 1
    assert events[0].data["penalty_side"] == "amarilla"


def test_crossing_rejects_confirmed_track_with_opposite_appearance():
    from src.C_quick_view.temporal_tracker import FutbotTemporalTracker, _Track

    tracker = FutbotTemporalTracker(30, 640, 480)
    reference = np.zeros(144, dtype=np.float64)
    reference[0] = 1.0
    opposite = np.zeros(144, dtype=np.float64)
    opposite[-1] = 1.0
    track = _Track.from_detection(
        0,
        {"class_name": "robot", "class_id": 0, "confidence": 0.9, "bbox_xyxy": [100, 100, 160, 160]},
        reference,
    )
    track.confirmed = True
    cost = tracker._robot_cost(
        track,
        {"class_name": "robot", "class_id": 0, "confidence": 0.9, "bbox_xyxy": [102, 100, 162, 160]},
        opposite,
    )
    assert np.isinf(cost)


def test_hologram_editor_has_clickable_fit_and_finish_buttons(tmp_path: Path):
    from src.I_field_geometry.field_spec import FieldSpec
    from src.I_field_geometry.hologram_wizard import HologramEditor

    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    writer.write(np.full((120, 160, 3), (60, 170, 100), dtype=np.uint8))
    writer.release()
    editor = HologramEditor(video, [0], [np.eye(3)], 10.0, 160, 120, FieldSpec())
    try:
        editor.render()
        assert {"fit", "zoom_out", "zoom_in", "save", "finish"}.issubset(editor.button_regions)
        editor._dispatch_button("fit")
        assert editor.video_zoom == 0.42
        editor._dispatch_button("save")
        editor._dispatch_button("finish")
        assert editor.finish_requested is True
    finally:
        editor.close()


def test_red_card_has_narration_text():
    from src.G_narration.event_editor import narration_text

    text = narration_text(
        {"event_type": "red_card_robot_removed", "data": {"robot_name": "Rival 1"}},
        0.95,
    )
    assert "Tarjeta roja" in text
    assert "Rival 1" in text
