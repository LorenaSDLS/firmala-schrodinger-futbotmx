import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def export_unity_mesa_json(output_directory: str | Path) -> str:
    output_directory = Path(output_directory)

    metadata_path = output_directory / "video_metadata.json"
    summary_path = output_directory / "match_summary.json"
    events_with_referee_path = output_directory / "match_events_with_referee.json"
    events_path = output_directory / "match_events.json"

    events_source = (
        events_with_referee_path
        if events_with_referee_path.exists()
        else events_path
    )

    final_data = {
        "video": load_json(metadata_path) if metadata_path.exists() else {},
        "summary": load_json(summary_path) if summary_path.exists() else {},
        "events": load_json(events_source) if events_source.exists() else [],
        "files": {
            "quick_preview": str(output_directory / "quick_preview.mp4"),
            "quick_detections": str(output_directory / "quick_detections.jsonl"),
            "possession_chart": str(output_directory / "possession_chart.png"),
            "object_paths": str(output_directory / "object_paths.png"),
            "event_timeline": str(output_directory / "event_timeline.png"),
        },
    }

    export_path = output_directory / "futbot_unity_mesa.json"

    with export_path.open("w", encoding="utf-8") as file:
        json.dump(final_data, file, indent=4, ensure_ascii=False)

    return str(export_path)