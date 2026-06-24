import argparse
import json
from pathlib import Path

from src.E_events.event_detector import generate_events



def run_step_03(output_directory: str | Path,
    generate_visual_reports: bool = True) -> dict:

    output_directory = Path(output_directory)
    detections_path = output_directory / "quick_detections.jsonl"

    if not detections_path.exists():
        raise FileNotFoundError(
            f"No se encontro quick_detections.jsonl en: {output_directory}"
        )

    events, summary, tracks = generate_events(detections_path)

    events_path = output_directory / "match_events.json"
    summary_path = output_directory / "match_summary.json"
    tracks_path = output_directory / "match_tracks.json"

    with events_path.open("w", encoding="utf-8") as file:
        json.dump(events, file, indent=4, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    with tracks_path.open("w", encoding="utf-8") as file:
        json.dump(tracks, file, indent=4, ensure_ascii=False)

    possession_chart_path = output_directory / "possession_chart.png"
    object_paths_path = output_directory / "object_paths.png"
    event_timeline_path = output_directory / "event_timeline.png"

    if generate_visual_reports:
        from src.E_events.visual_reports import (
            create_possession_chart,
            create_object_paths,
            create_event_timeline,
        )
        create_possession_chart(summary_path, possession_chart_path)
        create_object_paths(detections_path, object_paths_path)
        create_event_timeline(events_path, event_timeline_path)
    else:
        possession_chart_path = None
        object_paths_path = None
        event_timeline_path = None

    print("\n" + "=" * 55)
    print(" PASO 03 - EVENTOS Y VISUALES")
    print("=" * 55)
    print(f"Eventos JSON:       {events_path}")
    print(f"Resumen JSON:       {summary_path}")
    print(f"Grafica posesion:   {possession_chart_path}")
    print(f"Rutas objetos:      {object_paths_path}")
    print(f"Timeline eventos:   {event_timeline_path}")
    print(f"Eventos detectados: {summary['total_events']}")
    print(f"Tracks JSON:        {tracks_path}")
    print("=" * 55 + "\n")

    return {
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "tracks_path": str(tracks_path),
        "possession_chart_path": str(possession_chart_path) if possession_chart_path else None,
        "object_paths_path": str(object_paths_path) if object_paths_path else None,
        "event_timeline_path": str(event_timeline_path) if event_timeline_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae eventos y graficas desde quick_detections.jsonl."
    )
    parser.add_argument(
        "output_directory",
        help="Carpeta del analisis, por ejemplo outputs/IMG_9798.",
    )

    arguments = parser.parse_args()
    run_step_03(arguments.output_directory)


if __name__ == "__main__":
    main()