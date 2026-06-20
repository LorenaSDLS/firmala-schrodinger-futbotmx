from __future__ import annotations
import json
from pathlib import Path



import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def create_possession_chart(summary_path: str | Path, output_path: str | Path) -> None:
    summary = load_json(summary_path)
    possession = summary.get("possession_seconds", {})

    if not possession:
        possession = {"sin_posesion_clara": 0}

    labels = list(possession.keys())
    values = list(possession.values())

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title("Tiempo estimado de posesion")
    plt.xlabel("Robot")
    plt.ylabel("Segundos")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_event_timeline(events_path: str | Path, output_path: str | Path) -> None:
    events = load_json(events_path)

    if not events:
        events = [{
            "timestamp_seconds": 0,
            "event_type": "sin_eventos",
        }]

    event_types = sorted({event["event_type"] for event in events})
    event_type_to_y = {
        event_type: index
        for index, event_type in enumerate(event_types)
    }

    x_values = [event["timestamp_seconds"] for event in events]
    y_values = [event_type_to_y[event["event_type"]] for event in events]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, y_values)
    plt.yticks(range(len(event_types)), event_types)
    plt.title("Linea de tiempo de eventos detectados")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Tipo de evento")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_object_paths(detections_path: str | Path, output_path: str | Path) -> None:
    robot_points: dict[str, list[tuple[float, float]]] = {}
    ball_points: list[tuple[float, float]] = []

    with Path(detections_path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            for detection in record.get("detections", []):
                class_name = detection.get("class_name", "").lower()
                bbox = detection.get("bbox_xyxy") or detection.get("box")

                if not bbox:
                    continue

                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                if class_name == "robot":
                    robot_id = detection.get("tracking_id", "sin_id")
                    robot_points.setdefault(f"robot_{robot_id}", []).append(center)

                elif class_name in {"orange ball", "ball", "pelota"}:
                    ball_points.append(center)

    plt.figure(figsize=(10, 6))

    for robot_id, points in robot_points.items():
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, label=robot_id, linewidth=1.5)

    if ball_points:
        xs, ys = zip(*ball_points)
        plt.plot(xs, ys, label="pelota", linewidth=2.5, linestyle="--")

    plt.gca().invert_yaxis()
    plt.title("Rutas detectadas en pixeles")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()