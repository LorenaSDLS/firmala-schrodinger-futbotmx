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
) -> dict[str, Any] | None:
    """Convert a track point only when a trusted field transform exists.

    V8 silently fell back to a YOLO field bounding box or stabilized video
    pixels. Those values look smooth in Mesa Replay but are not coordinates on
    the physical field. V10 omits the point instead of fabricating a location.
    """
    if not (
        bool(point.get("field_transform_valid", False))
        and point.get("field_x_norm") is not None
        and point.get("field_y_norm") is not None
    ):
        return None

    normalized = point.copy()
    x_norm = float(point["field_x_norm"])
    y_norm = float(point["field_y_norm"])
    if not (-4.0 <= x_norm <= 5.0 and -4.0 <= y_norm <= 5.0):
        return None
    x_norm = min(1.75, max(-0.75, x_norm))
    y_norm = min(1.75, max(-0.75, y_norm))
    normalized["x_norm"] = round(x_norm, 6)
    normalized["y_norm"] = round(y_norm, 6)
    normalized["coordinate_source"] = "trusted_field_homography_v10"
    normalized["visible"] = bool(point.get("visible", True))
    return normalized


def normalize_tracks(
    tracks: dict[str, Any],
    video_width: float,
    video_height: float,
) -> dict[str, Any]:
    normalized_tracks = {"robots": {}, "ball": [], "goals": {}}

    def valid_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for point in points:
            normalized = normalize_track_point(point, video_width, video_height)
            if normalized is not None:
                output.append(normalized)
        return output

    for robot_id, points in tracks.get("robots", {}).items():
        normalized_tracks["robots"][robot_id] = valid_points(points)
    normalized_tracks["ball"] = valid_points(tracks.get("ball", []))
    for goal_id, points in tracks.get("goals", {}).items():
        normalized_tracks["goals"][goal_id] = valid_points(points)
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
            "source": "trusted_field_hologram_v11_2",
            "fallback": None,
            "normalized": True,
            "origin": "near_left_field_corner",
            "x_norm": "near_goal_to_far_goal; trusted global points only",
            "y_norm": "left_side_to_right_side; trusted global points only",
            "invalid_policy": "omit_points_without_trusted_global_registration",
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
