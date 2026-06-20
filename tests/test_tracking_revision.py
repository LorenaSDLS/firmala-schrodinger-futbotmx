import json
from pathlib import Path

import cv2
import numpy as np

from src.C_quick_view.team_classifier import TeamClassifier
from src.C_quick_view.temporal_tracker import FutbotTemporalTracker
from src.F_simulation.mesa_replay_model import FutbotReplayModel


def _green_frame() -> np.ndarray:
    return np.full((480, 640, 3), (70, 170, 90), dtype=np.uint8)


def test_tracks_need_confirmation_and_predictions_are_short():
    frame = _green_frame()
    tracker = FutbotTemporalTracker(60, 640, 480)
    output = []
    for index in range(3):
        output = tracker.update(
            [
                {
                    "class_id": 0,
                    "class_name": "robot",
                    "class_group": "robot",
                    "confidence": 0.92,
                    "bbox_xyxy": [90 + 2 * index, 160, 150 + 2 * index, 240],
                }
            ],
            frame,
        )
    assert len(output) == 1
    assert output[0]["confirmed"] is True
    assert len(tracker.update([], frame)) == 1
    assert len(tracker.update([], frame)) == 1
    assert len(tracker.update([], frame)) == 0


def _team_frame() -> tuple[np.ndarray, list[dict]]:
    frame = _green_frame()
    # Robot 0: dos platillos blancos anchos.
    cv2.ellipse(frame, (120, 190), (48, 12), 0, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(frame, (120, 225), (45, 12), 0, 0, 360, (235, 235, 235), -1)
    cv2.rectangle(frame, (105, 190), (112, 240), (40, 40, 40), -1)
    cv2.rectangle(frame, (130, 190), (137, 240), (40, 40, 40), -1)

    # Robots 1 y 2: misma construcción de base + torre, colores distintos.
    for x, base_color in ((270, (30, 30, 210)), (430, (30, 30, 30))):
        cv2.rectangle(frame, (x, 215), (x + 90, 260), base_color, -1)
        cv2.rectangle(frame, (x + 18, 165), (x + 25, 235), (25, 25, 25), -1)
        cv2.rectangle(frame, (x + 65, 165), (x + 72, 235), (25, 25, 25), -1)
        cv2.line(frame, (x + 22, 170), (x + 68, 170), (25, 25, 25), 6)
        cv2.circle(frame, (x + 32, 242), 10, (220, 220, 220), -1)
        cv2.circle(frame, (x + 62, 242), 10, (220, 220, 220), -1)

    detections = [
        {
            "tracking_id": 0,
            "class_group": "robot",
            "confidence": 0.95,
            "predicted": False,
            "bbox_xyxy": [65, 165, 175, 260],
        },
        {
            "tracking_id": 1,
            "class_group": "robot",
            "confidence": 0.95,
            "predicted": False,
            "bbox_xyxy": [265, 155, 365, 265],
        },
        {
            "tracking_id": 2,
            "class_group": "robot",
            "confidence": 0.95,
            "predicted": False,
            "bbox_xyxy": [425, 155, 525, 265],
        },
    ]
    return frame, detections


def test_auto_team_classifier_pairs_similar_construction_not_color():
    frame, detections = _team_frame()
    classifier = TeamClassifier(mode="auto")
    output = []
    for _ in range(12):
        output = classifier.update(frame, detections)

    by_id = {int(item["tracking_id"]): item for item in output}
    assert classifier.locked is True
    assert by_id[1]["team"] == by_id[2]["team"]
    assert by_id[0]["team"] != by_id[1]["team"]
    assert by_id[1]["display_name"] != by_id[2]["display_name"]


def test_field_box_is_direct_yolo_measurement_and_not_stale():
    tracker = FutbotTemporalTracker(30, 640, 480)
    frame = _green_frame()
    first = tracker.update(
        [
            {
                "class_name": "field",
                "class_group": "field",
                "confidence": 0.9,
                "bbox_xyxy": [10, 20, 500, 450],
            }
        ],
        frame,
    )
    assert first[0]["bbox_xyxy"] == [10, 20, 500, 450]
    second = tracker.update(
        [
            {
                "class_name": "field",
                "class_group": "field",
                "confidence": 0.91,
                "bbox_xyxy": [80, 60, 620, 470],
            }
        ],
        frame,
    )
    assert second[0]["bbox_xyxy"] == [80, 60, 620, 470]
    assert tracker.update([], frame) == []


def test_replay_clamps_impossible_jump(tmp_path: Path):
    payload = {
        "video": {"fps": 60},
        "events": [],
        "tracks": {
            "robots": {
                "robot_0": [
                    {
                        "frame_index": 0,
                        "x_norm": 0.1,
                        "y_norm": 0.1,
                        "visible": True,
                        "team": "aliado",
                        "display_name": "Aliado 1",
                    },
                    {
                        "frame_index": 1,
                        "x_norm": 0.95,
                        "y_norm": 0.95,
                        "visible": True,
                        "team": "aliado",
                        "display_name": "Aliado 1",
                    },
                ]
            },
            "ball": [],
        },
    }
    json_path = tmp_path / "replay.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    model = FutbotReplayModel(json_path)
    points = model.tracks["robots"]["robot_0"]
    assert points[1]["replay_jump_clamped"] is True


def test_replay_interpolates_ball_gap(tmp_path: Path):
    payload = {
        "video": {"fps": 10},
        "events": [],
        "tracks": {
            "robots": {},
            "ball": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "x_norm": 0.2,
                    "y_norm": 0.3,
                    "confidence": 0.9,
                    "visible": True,
                },
                {
                    "frame_index": 5,
                    "timestamp_seconds": 0.5,
                    "x_norm": 0.7,
                    "y_norm": 0.5,
                    "confidence": 0.8,
                    "visible": True,
                },
            ],
        },
    }
    json_path = tmp_path / "ball_replay.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    model = FutbotReplayModel(json_path, ball_interpolation_max_seconds=1.0)
    points = model.tracks["ball"]
    assert [point["frame_index"] for point in points] == list(range(6))
    assert all(point.get("source") == "interpolado" for point in points[1:5])


def test_ball_is_recovered_from_adaptive_orange_model():
    tracker = FutbotTemporalTracker(30, 640, 480)
    last_output = []
    for frame_index in range(4):
        frame = _green_frame()
        center_x = 100 + 5 * frame_index
        cv2.circle(frame, (center_x, 200), 8, (0, 120, 255), -1)
        detections = (
            []
            if frame_index == 3
            else [
                {
                    "class_id": 1,
                    "class_name": "orange ball",
                    "class_group": "ball",
                    "confidence": 0.9,
                    "bbox_xyxy": [center_x - 9, 191, center_x + 9, 209],
                }
            ]
        )
        last_output = tracker.update(detections, frame)

    assert len(last_output) == 1
    assert last_output[0].get("recovered_by_color") is True
    assert last_output[0].get("tracking_status") == "recuperado"


def _draw_robot(frame: np.ndarray, kind: str, x: int, y: int) -> list[float]:
    if kind in {"rojo", "negro"}:
        color = (25, 25, 210) if kind == "rojo" else (25, 25, 25)
        cv2.rectangle(frame, (x, y + 35), (x + 72, y + 74), color, -1)
        cv2.rectangle(frame, (x + 12, y), (x + 19, y + 58), (35, 35, 35), -1)
        cv2.rectangle(frame, (x + 53, y), (x + 60, y + 58), (35, 35, 35), -1)
        cv2.line(frame, (x + 16, y + 4), (x + 56, y + 4), (35, 35, 35), 5)
    else:
        cv2.ellipse(frame, (x + 36, y + 18), (34, 9), 0, 0, 360, (235, 235, 235), -1)
        cv2.ellipse(frame, (x + 36, y + 53), (32, 9), 0, 0, 360, (235, 235, 235), -1)
        cv2.rectangle(frame, (x + 22, y + 18), (x + 28, y + 72), (40, 40, 40), -1)
        cv2.rectangle(frame, (x + 45, y + 18), (x + 51, y + 72), (40, 40, 40), -1)
    return [float(x), float(y), float(x + 72), float(y + 76)]


def test_v4_reconstructs_physical_ids_after_online_swap(tmp_path: Path):
    from src.C_quick_view.offline_identity import (
        OfflineIdentityConfig,
        reconstruct_physical_identities,
    )

    video_path = tmp_path / "swap.mp4"
    detections_path = tmp_path / "quick_detections.jsonl"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (640, 360),
    )
    records = []
    kinds = ["rojo", "negro", "blanco_a", "blanco_b"]
    positions = [(70, 150), (210, 150), (360, 145), (500, 145)]
    for frame_index in range(24):
        frame = np.full((360, 640, 3), (70, 170, 90), dtype=np.uint8)
        boxes = [_draw_robot(frame, kind, x, y) for kind, (x, y) in zip(kinds, positions)]
        writer.write(frame)
        online_ids = [0, 1, 2, 3] if frame_index < 12 else [2, 1, 0, 3]
        detections = [
            {
                "class_id": 0,
                "class_name": "robot",
                "class_group": "robot",
                "confidence": 0.96,
                "bbox_xyxy": box,
                "tracking_id": online_id,
                "predicted": False,
                "measured": True,
            }
            for box, online_id in zip(boxes, online_ids)
        ]
        detections.append(
            {
                "class_id": 2,
                "class_name": "field",
                "class_group": "field",
                "confidence": 0.99,
                "bbox_xyxy": [0.0, 80.0, 640.0, 350.0],
            }
        )
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / 20.0,
                "camera_registration": {
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "valid": True,
                },
                "detections": detections,
            }
        )
    writer.release()
    detections_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = reconstruct_physical_identities(
        video_path,
        detections_path,
        tmp_path,
        OfflineIdentityConfig(
            sample_stride=2,
            minimum_tracklet_measurements=3,
            render_corrected_preview=False,
        ),
    )
    assert result["identity_count"] == 4

    rewritten = [json.loads(line) for line in detections_path.read_text().splitlines()]
    before_ids = None
    for frame_index in (5, 18):
        robots = [
            detection
            for detection in rewritten[frame_index]["detections"]
            if detection.get("class_group") == "robot"
        ]
        by_x = sorted(robots, key=lambda item: item["bbox_xyxy"][0])
        assert by_x[0]["team"] == by_x[1]["team"]
        assert by_x[2]["team"] == by_x[3]["team"]
        assert by_x[0]["team"] != by_x[2]["team"]
        current_ids = [item["physical_robot_id"] for item in by_x]
        if before_ids is None:
            before_ids = current_ids
        else:
            assert current_ids == before_ids


def test_bidirectional_replay_smoothing_reduces_single_frame_jitter(tmp_path: Path):
    payload = {
        "video": {"fps": 30},
        "events": [],
        "tracks": {
            "robots": {
                "robot_0": [
                    {
                        "frame_index": index,
                        "x_norm": 0.2 + 0.01 * index + (0.12 if index == 5 else 0.0),
                        "y_norm": 0.4,
                        "visible": True,
                        "team": "aliado",
                        "display_name": "Aliado 1",
                        "confidence": 0.9,
                    }
                    for index in range(11)
                ]
            },
            "ball": [],
        },
    }
    json_path = tmp_path / "smooth.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    model = FutbotReplayModel(json_path)
    points = model.tracks["robots"]["robot_0"]
    assert points[5]["x_norm"] < points[5]["x_norm_raw"]
    assert all(point.get("replay_smoothed_bidirectional") for point in points)



def test_offline_identity_single_view_distance_does_not_crash(monkeypatch):
    import src.C_quick_view.offline_identity as offline_identity

    monkeypatch.setattr(
        offline_identity,
        "rotation_invariant_distance",
        lambda first, second, identity: 0.27,
    )

    distance = offline_identity._robust_views_distance(
        [[object()]],
        [[object()]],
        identity=True,
    )

    assert distance == 0.27


def test_online_team_pairing_single_sample_does_not_crash(monkeypatch):
    import src.C_quick_view.team_clustering as team_clustering

    monkeypatch.setattr(
        team_clustering,
        "team_feature_distance",
        lambda first, second: 0.19,
    )

    distance = team_clustering._robust_pair_distance(
        [object()],
        [object()],
    )

    assert distance == 0.19


def test_v5_field_selection_prefers_complete_field_over_small_high_confidence_box():
    from src.C_quick_view.field_selector import select_main_field

    candidates = [
        {
            "class_group": "field",
            "confidence": 0.94,
            "bbox_xyxy": [240.0, 160.0, 620.0, 460.0],
        },
        {
            "class_group": "field",
            "confidence": 0.72,
            "bbox_xyxy": [20.0, 40.0, 630.0, 470.0],
        },
    ]

    selected, diagnostics = select_main_field(candidates, 640, 480)

    assert selected is not None
    assert selected["bbox_xyxy"] == candidates[1]["bbox_xyxy"]
    assert selected["field_candidates_count"] == 2
    assert len(diagnostics) == 2


def test_v52_default_thresholds_include_goal_without_lowering_robots():
    from src.C_quick_view.yolo_detector import DEFAULT_CLASS_THRESHOLDS

    assert DEFAULT_CLASS_THRESHOLDS == {
        "robot": 0.55,
        "ball": 0.35,
        "field": 0.25,
        "goal": 0.35,
    }


def test_goal_detections_receive_stable_image_sides():
    tracker = FutbotTemporalTracker(30, 640, 480)
    output = tracker.update(
        [
            {
                "class_id": 2,
                "class_name": "goal",
                "class_group": "goal",
                "confidence": 0.91,
                "bbox_xyxy": [20, 150, 100, 300],
            },
            {
                "class_id": 2,
                "class_name": "goal",
                "class_group": "goal",
                "confidence": 0.88,
                "bbox_xyxy": [540, 145, 625, 305],
            },
        ],
        _green_frame(),
    )
    goals = [item for item in output if item.get("class_group") == "goal"]
    assert [item["goal_side_image"] for item in goals] == ["izquierda", "derecha"]
    assert [item["goal_id"] for item in goals] == ["goal_izquierda", "goal_derecha"]


def test_goal_event_requires_consecutive_measured_ball_frames():
    from src.E_events.event_detector import EventDetector

    detector = EventDetector()
    for frame_index in range(5):
        detector.process_frame_record(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / 30.0,
                "detections": [
                    {
                        "class_name": "goal",
                        "class_group": "goal",
                        "goal_id": "goal_izquierda",
                        "goal_side_image": "izquierda",
                        "confidence": 0.92,
                        "bbox_xyxy": [10, 100, 120, 260],
                    },
                    {
                        "class_name": "ball",
                        "class_group": "ball",
                        "confidence": 0.89,
                        "predicted": False,
                        "bbox_xyxy": [50, 170, 66, 186],
                    },
                ],
            }
        )
    goals = [event for event in detector.events if event.event_type == "goal"]
    assert len(goals) == 1
    assert goals[0].data["goal_side_image"] == "izquierda"
    assert goals[0].data["scoring_team"] == "desconocido"


def test_v5_online_pairing_does_not_lock_when_margin_is_ambiguous(monkeypatch):
    from collections import deque

    import src.C_quick_view.team_classifier as team_classifier_module
    from src.C_quick_view.team_clustering import PairingResult

    classifier = team_classifier_module.TeamClassifier(mode="auto")
    classifier.minimum_samples_per_robot = 1
    classifier.minimum_pairing_margin = 0.12
    classifier.samples_by_id = {
        0: deque([object()]),
        1: deque([object()]),
        2: deque([object()]),
        3: deque([object()]),
    }

    monkeypatch.setattr(
        team_classifier_module,
        "solve_two_team_pairing",
        lambda samples: PairingResult(
            pair_a=(0, 1),
            pair_b=(2, 3),
            cost=0.40,
            second_best_cost=0.42,
            margin=(0.42 - 0.40) / 0.42,
            confidence=0.25,
            pair_distances={},
        ),
    )

    classifier._try_lock_pairing()

    assert classifier.locked is False
    assert classifier.team_by_id == {}


def test_v5_offline_pairing_returns_unknown_when_evidence_is_ambiguous(monkeypatch):
    from types import SimpleNamespace

    import src.C_quick_view.offline_identity as offline_identity

    tracklets = {
        robot_id: SimpleNamespace(feature_views=[[[robot_id]]])
        for robot_id in range(4)
    }
    clusters = {robot_id: [robot_id] for robot_id in range(4)}
    distances = {
        (0, 1): 0.20,
        (2, 3): 0.20,
        (0, 2): 0.21,
        (1, 3): 0.21,
        (0, 3): 0.50,
        (1, 2): 0.50,
    }

    def fake_distance(samples_a, samples_b, *, identity):
        _ = identity
        first = int(samples_a[0][0][0])
        second = int(samples_b[0][0][0])
        return distances[tuple(sorted((first, second)))]

    monkeypatch.setattr(offline_identity, "_robust_views_distance", fake_distance)

    (
        teams,
        numbers,
        _,
        proposed,
        confirmed,
        margin,
    ) = offline_identity._physical_pairing(
        clusters,
        tracklets,
        False,
        minimum_pairing_margin=0.12,
        force_pairing=False,
    )

    assert proposed == [[0, 1], [2, 3]]
    assert confirmed is False
    assert margin < 0.12
    assert set(teams.values()) == {"desconocido"}
    assert numbers == {}


def test_field_geometry_builds_homography_from_segmented_trapezoid():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator
    from src.I_field_geometry.field_segmenter import FieldMaskResult

    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[150, 90], [530, 110], [620, 450], [20, 440]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.95,
        class_id=0,
        class_name="field",
        bbox_xyxy=[20.0, 90.0, 620.0, 450.0],
        polygon=polygon.astype(np.float32),
        coverage=float(np.count_nonzero(mask)) / float(mask.size),
    )
    estimator = FieldGeometryEstimator(width, height, 100.0, 60.0)
    result = estimator.update(segmentation, np.eye(3, dtype=np.float64))

    assert result.valid is True
    assert result.measured is True
    assert result.homography_image_to_field is not None
    near_left = estimator.transform_point(20, 440)
    far_left = estimator.transform_point(150, 90)
    assert near_left is not None and far_left is not None
    assert near_left[0] < 5.0
    assert far_left[0] > 95.0


def test_field_geometry_propagates_when_mask_disappears():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator
    from src.I_field_geometry.field_segmenter import FieldMaskResult

    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[160, 100], [520, 100], [610, 450], [30, 450]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.94,
        class_id=0,
        class_name="field",
        bbox_xyxy=[30.0, 100.0, 610.0, 450.0],
        polygon=polygon.astype(np.float32),
        coverage=float(np.count_nonzero(mask)) / float(mask.size),
    )
    estimator = FieldGeometryEstimator(width, height)
    measured = estimator.update(segmentation, np.eye(3, dtype=np.float64))
    propagated = estimator.update(
        None,
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]]),
    )

    assert measured.measured is True
    assert propagated.valid is True
    assert propagated.propagated is True
    assert propagated.source == "propagada_registro"


def test_unity_exporter_prefers_homography_and_keeps_outside_coordinates():
    from src.E_events.unity_exporter import normalize_track_point

    point = normalize_track_point(
        {
            "x_px": 320.0,
            "y_px": 240.0,
            "field_transform_valid": True,
            "field_x_norm": 1.08,
            "field_y_norm": 0.42,
        },
        video_width=640.0,
        video_height=480.0,
    )

    assert point["coordinate_source"] == "trusted_field_homography_v10"
    assert point["x_norm"] == 1.08
    assert point["y_norm"] == 0.42


def test_field_geometry_does_not_use_image_border_as_first_calibration():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator
    from src.I_field_geometry.field_segmenter import FieldMaskResult

    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    # The actual near side continues below the video; the visible mask is clipped
    # exactly by the bottom image border.
    polygon = np.array(
        [[140, 90], [520, 105], [640, 480], [0, 480]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    segmentation = FieldMaskResult(
        mask=mask,
        confidence=0.96,
        class_id=0,
        class_name="field",
        bbox_xyxy=[0.0, 90.0, 640.0, 480.0],
        polygon=polygon.astype(np.float32),
        coverage=float(np.count_nonzero(mask)) / float(mask.size),
    )
    estimator = FieldGeometryEstimator(width, height)
    result = estimator.update(segmentation, np.eye(3, dtype=np.float64))

    assert result.valid is False
    assert result.side_visible.get("near") is False


def _synthetic_clipped_field() -> tuple[np.ndarray, np.ndarray]:
    from src.I_field_geometry.field_segmenter import FieldMaskResult

    height, width = 480, 640
    frame = np.full((height, width, 3), (215, 215, 215), dtype=np.uint8)
    # Physical surface extends beyond the left and lower camera margins.
    polygon = np.array([[-90, 190], [515, 112], [735, 500], [-115, 565]], np.int32)
    cv2.fillConvexPoly(frame, polygon, (65, 170, 112))
    # Visible black rails: far and right.  The other two physical sides are
    # outside the camera and must not be replaced with x=0 / y=height.
    cv2.line(frame, (-90, 190), (515, 112), (28, 28, 28), 18, cv2.LINE_AA)
    cv2.line(frame, (515, 112), (735, 500), (28, 28, 28), 18, cv2.LINE_AA)
    cv2.line(frame, (70, 250), (610, 205), (245, 245, 245), 8, cv2.LINE_AA)
    cv2.line(frame, (310, 160), (470, 455), (245, 245, 245), 8, cv2.LINE_AA)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    result = FieldMaskResult(
        mask=mask,
        confidence=0.98,
        class_id=0,
        class_name="field_surface",
        bbox_xyxy=[0.0, 105.0, 639.0, 479.0],
        polygon=polygon.astype(np.float32),
        coverage=float(np.count_nonzero(mask)) / mask.size,
    )
    return frame, result


def test_field_geometry_rejects_camera_margins_and_extrapolates_missing_sides():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, segmentation = _synthetic_clipped_field()
    estimator = FieldGeometryEstimator(640, 480)
    result = estimator.update(
        segmentation=segmentation,
        current_to_reference=np.eye(3),
        frame=frame,
        goal_detections=[],
    )
    assert result.valid is False
    assert result.trusted is False
    assert result.geometry_state == "local"
    assert result.local_homography_image_to_local is not None
    assert "left" in result.rejected_frame_sides
    assert "near" in result.rejected_frame_sides
    # V8 keeps the physical-edge evidence but refuses to invent off-screen corners.
    assert result.corners_image is None
    assert result.homography_image_to_field is None


def test_performance_auto_is_cpu_friendly_without_explicit_overrides(monkeypatch):
    import src.shared.performance as performance

    monkeypatch.setattr(performance, "_cuda_available", lambda: False)
    settings = performance.resolve_performance_settings("auto")
    assert settings.resolved_profile == "cpu"
    assert settings.field_segmentation_image_size == 448
    assert settings.field_segmentation_stride == 6


def test_performance_explicit_values_win(monkeypatch):
    import src.shared.performance as performance

    monkeypatch.setattr(performance, "_cuda_available", lambda: False)
    settings = performance.resolve_performance_settings(
        "cpu",
        field_segmentation_image_size=512,
        field_segmentation_stride=4,
    )
    assert settings.field_segmentation_image_size == 512
    assert settings.field_segmentation_stride == 4


def test_clipped_two_side_bootstrap_is_provisional_and_not_exported_as_valid():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    frame, segmentation = _synthetic_clipped_field()
    estimator = FieldGeometryEstimator(640, 480)
    result = estimator.update(
        segmentation=segmentation,
        current_to_reference=np.eye(3),
        frame=frame,
        goal_detections=[],
    )
    annotated = estimator.annotate_detection(
        {
            "class_group": "ball",
            "bbox_xyxy": [300.0, 230.0, 315.0, 245.0],
        }
    )

    assert result.valid is False
    assert result.trusted is False
    assert result.geometry_state == "local"
    assert result.homography_image_to_field is None
    assert annotated["field_transform_valid"] is False
    assert annotated["field_transform_provisional"] is False


def test_goal_consistency_requires_opposite_ends_for_two_goals():
    from src.I_field_geometry.field_geometry import FieldGeometryEstimator

    estimator = FieldGeometryEstimator(640, 480, 100.0, 60.0)
    image_corners = np.float32(
        [[150, 80], [500, 95], [610, 450], [35, 445]]
    )
    good_h = cv2.getPerspectiveTransform(image_corners, estimator.canonical_corners)
    goals = [
        {"class_group": "goal", "bbox_xyxy": [430, 80, 510, 120]},
        {"class_group": "goal", "bbox_xyxy": [20, 390, 110, 460]},
    ]
    good_score = estimator._goal_consistency_score(good_h, goals)

    # This homography collapses longitudinal separation so both goals land near
    # the same canonical end.  One individually good goal must not hide it.
    bad_image_corners = np.float32(
        [[20, 390], [110, 460], [190, 470], [100, 400]]
    )
    bad_h = cv2.getPerspectiveTransform(bad_image_corners, estimator.canonical_corners)
    bad_score = estimator._goal_consistency_score(bad_h, goals)

    assert good_score > bad_score


def test_cpu_profile_samples_expensive_debug_rendering(monkeypatch):
    import src.shared.performance as performance

    monkeypatch.setattr(performance, "_cuda_available", lambda: False)
    settings = performance.resolve_performance_settings("auto")
    assert settings.field_debug_stride == 6


def test_offline_identity_preserves_unresolved_robot_without_physical_id(tmp_path: Path):
    from src.C_quick_view.offline_identity import (
        OfflineIdentityConfig,
        reconstruct_physical_identities,
    )

    video_path = tmp_path / "unresolved.mp4"
    detections_path = tmp_path / "quick_detections.jsonl"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    records = []
    for frame_index in range(3):
        frame = np.full((120, 160, 3), (70, 170, 90), dtype=np.uint8)
        detections = []
        if frame_index == 1:
            cv2.rectangle(frame, (40, 30), (80, 90), (30, 30, 30), -1)
            detections.append(
                {
                    "class_group": "robot",
                    "class_name": "robot",
                    "confidence": 0.9,
                    "bbox_xyxy": [40, 30, 80, 90],
                    "tracking_id": 7,
                    "predicted": False,
                    "measured": True,
                }
            )
        writer.write(frame)
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / 10.0,
                "camera_registration": {
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "valid": True,
                },
                "detections": detections,
            }
        )
    writer.release()
    detections_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = reconstruct_physical_identities(
        video_path,
        detections_path,
        tmp_path,
        OfflineIdentityConfig(
            sample_stride=1,
            minimum_tracklet_measurements=3,
            preserve_unresolved_detections=True,
            render_corrected_preview=False,
        ),
    )

    assert result["identity_count"] == 0
    rewritten = [json.loads(line) for line in detections_path.read_text().splitlines()]
    robot = rewritten[1]["detections"][0]
    assert robot["identity_resolved_offline"] is False
    assert robot["physical_robot_id"] is None
    assert robot["tracking_id"] == 7
