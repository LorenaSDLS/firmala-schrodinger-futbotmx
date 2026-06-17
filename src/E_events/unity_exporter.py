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

    x_px = float(point.get("x_px", 0.0))
    y_px = float(point.get("y_px", 0.0))

    normalized["x_norm"] = round(x_px / video_width, 6)
    normalized["y_norm"] = round(y_px / video_height, 6)

    return normalized


def normalize_tracks(
    tracks: dict[str, Any],
    video_width: float,
    video_height: float,
) -> dict[str, Any]:
    normalized_tracks = {
        "robots": {},
        "ball": [],
    }

    for robot_id, points in tracks.get("robots", {}).items():
        normalized_tracks["robots"][robot_id] = [
            normalize_track_point(
                point=point,
                video_width=video_width,
                video_height=video_height,
            )
            for point in points
        ]

    normalized_tracks["ball"] = [
        normalize_track_point(
            point=point,
            video_width=video_width,
            video_height=video_height,
        )
        for point in tracks.get("ball", [])
    ]

    return normalized_tracks


def count_events(events: list[dict[str, Any]]) -> dict[str, int]:
    event_counts = {}

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
        events_with_referee_path
        if events_with_referee_path.exists()
        else events_path
    )

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}
    events = load_json(events_source) if events_source.exists() else []
    tracks = load_json(tracks_path) if tracks_path.exists() else {
        "robots": {},
        "ball": [],
    }

    video_width, video_height = get_video_resolution(metadata)
    normalized_tracks = normalize_tracks(
        tracks=tracks,
        video_width=video_width,
        video_height=video_height,
    )

    summary["total_events"] = len(events)
    summary["event_counts"] = count_events(events)

    final_data = {
        "schema_version": "0.1",
        "coordinate_system": {
            "source": "video_pixels",
            "normalized": True,
            "origin": "top_left",
            "x_norm": "x_px / video_width",
            "y_norm": "y_px / video_height",
        },
        "video": metadata,
        "summary": summary,
        "events": events,
        "tracks": normalized_tracks,
        "files": {
            "quick_preview": str(output_directory / "quick_preview.mp4"),
            "quick_detections": str(output_directory / "quick_detections.jsonl"),
            "match_tracks": str(tracks_path),
            "possession_chart": str(output_directory / "possession_chart.png"),
            "object_paths": str(output_directory / "object_paths.png"),
            "event_timeline": str(output_directory / "event_timeline.png"),
        },
    }

    export_path = output_directory / "futbot_unity_mesa.json"
    save_json(final_data, export_path)

    return str(export_path)