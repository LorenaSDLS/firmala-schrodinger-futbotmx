import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def get_video_resolution(metadata: dict[str, Any]) -> tuple[float, float]:
    resolution = metadata.get("resolution", {})
    width = float(resolution.get("width") or metadata.get("width") or 1)
    height = float(resolution.get("height") or metadata.get("height") or 1)
    return width, height


def normalize_track_point(
    point: dict[str, Any],
    video_width: float,
    video_height: float,
) -> dict[str, Any]:
    normalized = point.copy()

    raw_x_px = float(point.get("x_px", 0.0))
    raw_y_px = float(point.get("y_px", 0.0))
    field_box = point.get("field_bbox_xyxy")

    # La homografía de la máscara es la fuente preferida. A diferencia de una
    # caja, conserva perspectiva y permite coordenadas fuera de [0, 1].
    if (
        bool(point.get("field_transform_valid", False))
        and point.get("field_x_norm") is not None
        and point.get("field_y_norm") is not None
    ):
        x_norm = float(point["field_x_norm"])
        y_norm = float(point["field_y_norm"])
        source = "field_mask_homography"
    # YOLO detecta la cancha en cada cuadro. Esta caja queda como respaldo.
    elif field_box and len(field_box) == 4:
        field_x1, field_y1, field_x2, field_y2 = map(float, field_box)
        field_width = max(1.0, field_x2 - field_x1)
        field_height = max(1.0, field_y2 - field_y1)
        x_norm = (raw_x_px - field_x1) / field_width
        y_norm = (raw_y_px - field_y1) / field_height
        source = "field_bbox_relative"
    else:
        use_stabilized = (
            point.get("stabilized_x_px") is not None
            and point.get("stabilized_y_px") is not None
            and bool(point.get("registration_valid", False))
        )
        if use_stabilized:
            x_px = float(point["stabilized_x_px"])
            y_px = float(point["stabilized_y_px"])
            source = "camera_stabilized_pixels"
        else:
            x_px = raw_x_px
            y_px = raw_y_px
            source = "video_pixels"
        x_norm = x_px / max(video_width, 1.0)
        y_norm = y_px / max(video_height, 1.0)

    # Mantiene salidas moderadas fuera del campo para representar porterías y
    # balones fuera. Solo recorta valores absurdos producidos por una homografía mala.
    x_norm = min(1.75, max(-0.75, x_norm))
    y_norm = min(1.75, max(-0.75, y_norm))
    normalized["x_norm"] = round(x_norm, 6)
    normalized["y_norm"] = round(y_norm, 6)
    normalized["coordinate_source"] = source
    return normalized


def canonical_robot_id(raw_robot_id: Any) -> str | None:
    """
    Convierte IDs de V8/V1 a los nombres que Unity ya tiene:
    robot_0, robot_1, robot_2, robot_3.
    """
    if raw_robot_id is None:
        return None

    text = str(raw_robot_id).strip()

    if text.startswith("robot_"):
        suffix = text.replace("robot_", "", 1)
        if suffix.isdigit():
            index = int(suffix)
            if 0 <= index <= 3:
                return f"robot_{index}"

    if text.isdigit():
        index = int(text)
        if 0 <= index <= 3:
            return f"robot_{index}"

    return None


def robot_index_from_id(robot_id: str) -> int | None:
    if not robot_id.startswith("robot_"):
        return None

    suffix = robot_id.replace("robot_", "", 1)

    if not suffix.isdigit():
        return None

    return int(suffix)


def normalize_tracks(
    tracks: dict[str, Any],
    video_width: float,
    video_height: float,
) -> dict[str, Any]:
    # Unity ya tiene estos 4 GameObjects.
    normalized_tracks = {
        "robots": {
            "robot_0": [],
            "robot_1": [],
            "robot_2": [],
            "robot_3": [],
        },
        "ball": [],
        "goals": {},
    }

    for raw_robot_id, points in tracks.get("robots", {}).items():
        unity_robot_id = canonical_robot_id(raw_robot_id)

        if unity_robot_id is None:
            print(
                "Aviso: se omitió un robot con ID no compatible "
                f"con Unity: {raw_robot_id}"
            )
            continue

        unity_index = robot_index_from_id(unity_robot_id)

        normalized_points = []

        for point in points:
            normalized_point = normalize_track_point(
                point,
                video_width,
                video_height,
            )

            # Alias compatibles con Unity y con V8.
            normalized_point["id"] = unity_robot_id
            normalized_point["robot_id"] = unity_robot_id
            normalized_point["tracking_id"] = point.get(
                "tracking_id",
                unity_index,
            )
            normalized_point["physical_robot_id"] = point.get(
                "physical_robot_id",
                unity_index,
            )
            normalized_point["display_name"] = point.get(
                "display_name",
                unity_robot_id,
            )
            normalized_point["team"] = point.get(
                "team",
                "desconocido",
            )
            normalized_point["team_number"] = point.get("team_number")

            normalized_points.append(normalized_point)

        normalized_tracks["robots"][unity_robot_id] = normalized_points

    normalized_tracks["ball"] = [
        normalize_track_point(point, video_width, video_height)
        for point in tracks.get("ball", [])
    ]

    for goal_id, points in tracks.get("goals", {}).items():
        normalized_tracks["goals"][goal_id] = [
            normalize_track_point(point, video_width, video_height)
            for point in points
        ]

    return normalized_tracks


def count_events(events: list[dict[str, Any]]) -> dict[str, int]:
    event_counts: dict[str, int] = {}
    for event in events:
        event_type = event.get("event_type", "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    return event_counts


def export_unity_mesa_json(output_directory: str | Path) -> str:
    output_directory = Path(output_directory)
    metadata_path = output_directory / "video_metadata.json"
    summary_path = output_directory / "match_summary.json"
    tracks_path = output_directory / "match_tracks.json"
    events_with_referee_path = output_directory / "match_events_with_referee.json"
    events_path = output_directory / "match_events.json"
    events_source = (
        events_with_referee_path if events_with_referee_path.exists() else events_path
    )

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}
    events = load_json(events_source) if events_source.exists() else []
    tracks = load_json(tracks_path) if tracks_path.exists() else {"robots": {}, "ball": [], "goals": {}}

    video_width, video_height = get_video_resolution(metadata)
    normalized_tracks = normalize_tracks(tracks, video_width, video_height)
    summary["total_events"] = len(events)
    summary["event_counts"] = count_events(events)

    final_data = {
        "schema_version": "0.2",
        "language": "es",
        "coordinate_system": {
            "source": "field_mask_homography_when_available",
            "fallback": "field_bbox_then_camera_stabilized_pixels_then_video_pixels",
            "normalized": True,
            "origin": "top_left",
            "x_norm": "near_goal_to_far_goal; values may be outside 0..1",
            "y_norm": "cross_field; values may be outside 0..1",
        },
        "video": metadata,
        "summary": summary,
        "events": events,
        "tracks": normalized_tracks,
        "files": {
            "quick_preview": str(output_directory / "quick_preview.mp4"),
            "quick_detections": str(output_directory / "quick_detections.jsonl"),
            "tracking_debug": str(output_directory / "tracking_debug.jsonl"),
            "rejected_detections": str(output_directory / "rejected_detections.jsonl"),
            "match_tracks": str(tracks_path),
            "possession_chart": str(output_directory / "possession_chart.png"),
            "object_paths": str(output_directory / "object_paths.png"),
            "event_timeline": str(output_directory / "event_timeline.png"),
        },
    }

    export_path = output_directory / "futbot_unity_mesa.json"
    save_json(final_data, export_path)
    return str(export_path)
