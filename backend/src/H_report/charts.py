from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Rectangle

from src.H_report.report_data import normalize_team

MAGENTA = "#d21aa5"
BLUE = "#0986df"
CYAN = "#12d9ff"
DARK = "#08051d"
FREE = "#c8cad4"
LENGTH = 243.0
WIDTH = 182.0


def _coord(point: dict[str, Any]) -> tuple[float, float] | None:
    if not point.get("field_transform_valid", False):
        return None
    x_value, y_value = point.get("field_x"), point.get("field_y")
    if x_value is None or y_value is None:
        x_norm, y_norm = point.get("field_x_norm"), point.get("field_y_norm")
        if x_norm is None or y_norm is None:
            return None
        x_value = float(x_norm) * LENGTH
        y_value = float(y_norm) * WIDTH
    x_value, y_value = float(x_value), float(y_value)
    return (x_value, y_value) if -3 <= x_value <= LENGTH + 3 and -3 <= y_value <= WIDTH + 3 else None


def _track_team(points: list[dict[str, Any]]) -> str:
    votes = Counter(
        normalize_team(point.get("team"))
        for point in points
        if normalize_team(point.get("team")) != "desconocido" and point.get("visible", True)
    )
    return votes.most_common(1)[0][0] if votes else "desconocido"


def _field(axis: Any) -> None:
    axis.set_facecolor(DARK)
    axis.add_patch(Rectangle((0, 0), LENGTH, WIDTH, fill=False, edgecolor="#743fff", linewidth=2.2))
    axis.plot([LENGTH / 2, LENGTH / 2], [0, WIDTH], color=CYAN, linewidth=1.5, alpha=0.85)
    center_y = WIDTH / 2
    area_width = 80.0
    area_depth = 25.0
    y0 = center_y - area_width / 2
    y1 = center_y + area_width / 2
    axis.plot([0, area_depth], [y0, y0], color=MAGENTA, linewidth=1.35)
    axis.plot([0, area_depth], [y1, y1], color=MAGENTA, linewidth=1.35)
    axis.add_patch(Arc((area_depth, center_y), 2 * area_depth, area_width, theta1=90, theta2=270, color=MAGENTA, linewidth=1.35))
    axis.add_patch(Rectangle((-8, center_y - 30), 8, 60, fill=False, edgecolor=MAGENTA, linewidth=1.1))
    axis.plot([LENGTH - area_depth, LENGTH], [y0, y0], color=BLUE, linewidth=1.35)
    axis.plot([LENGTH - area_depth, LENGTH], [y1, y1], color=BLUE, linewidth=1.35)
    axis.add_patch(Arc((LENGTH - area_depth, center_y), 2 * area_depth, area_width, theta1=-90, theta2=90, color=BLUE, linewidth=1.35))
    axis.add_patch(Rectangle((LENGTH, center_y - 30), 8, 60, fill=False, edgecolor=BLUE, linewidth=1.1))
    axis.set_xlim(-10, LENGTH + 10)
    axis.set_ylim(WIDTH + 5, -5)
    axis.set_aspect("equal")
    axis.axis("off")


def _save(figure: Any, path: Path, width: float, height: float, facecolor: str = DARK) -> None:
    figure.set_size_inches(width, height)
    figure.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    figure.savefig(path, dpi=180, facecolor=facecolor, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _resolve_team_map(tracks: dict[str, Any], team_assignments: dict[str, str] | None) -> dict[str, str]:
    resolved = dict(team_assignments or {})
    for robot_id, points in (tracks.get("robots") or {}).items():
        resolved.setdefault(str(robot_id), _track_team(points))
    return resolved


def _smoothed_histogram(points: list[tuple[float, float]], bins: tuple[int, int] = (40, 30)) -> np.ndarray:
    if not points:
        return np.zeros((bins[1], bins[0]), dtype=np.float32)
    coordinates = np.asarray(points, dtype=np.float32)
    histogram, _, _ = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=bins,
        range=((0, LENGTH), (0, WIDTH)),
    )
    histogram = histogram.T.astype(np.float32)
    # Lightweight separable blur without adding a SciPy dependency.
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    for _ in range(2):
        histogram = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, histogram)
        histogram = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, histogram)
    if histogram.max() > 0:
        histogram /= histogram.max()
    return histogram


def _draw_heatmap(path: Path, points: list[tuple[float, float]], cmap: str, empty_message: str) -> None:
    figure, axis = plt.subplots()
    _field(axis)
    histogram = _smoothed_histogram(points)
    if histogram.max() > 0:
        alpha = np.clip(histogram * 0.88, 0.0, 0.88)
        axis.imshow(
            histogram,
            extent=(0, LENGTH, WIDTH, 0),
            cmap=cmap,
            alpha=alpha,
            interpolation="bilinear",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
    else:
        axis.text(
            LENGTH / 2,
            WIDTH / 2,
            empty_message,
            color="#d7d9e5",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=0.82,
        )
    _save(figure, path, 5.1, 3.9)


def generate_charts(
    tracks_path: str | Path,
    output_directory: str | Path,
    *,
    team_assignments: dict[str, str] | None = None,
) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    teams = _resolve_team_map(tracks, team_assignments)

    # Movement map.
    movement_path = output / "mapa_movimiento.png"
    figure, axis = plt.subplots()
    _field(axis)
    for raw_robot_id, points in (tracks.get("robots") or {}).items():
        robot_id = str(raw_robot_id)
        team = teams.get(robot_id, "desconocido")
        color = MAGENTA if team == "aliado" else BLUE if team == "rival" else "#d7d7df"
        coordinates = [coordinate for coordinate in (_coord(point) for point in points if point.get("visible", True)) if coordinate]
        if len(coordinates) <= 1:
            continue
        xy = np.asarray(coordinates)
        start = 0
        for index in range(1, len(xy) + 1):
            break_segment = index == len(xy) or np.linalg.norm(xy[index] - xy[index - 1]) > 45
            if break_segment:
                segment = xy[start:index]
                if len(segment) > 1:
                    axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.25, alpha=0.65)
                start = index
        axis.scatter(xy[0, 0], xy[0, 1], s=14, color=color, marker="o", edgecolors="white", linewidths=0.4, zorder=5)
        axis.scatter(xy[-1, 0], xy[-1, 1], s=18, color=color, marker="x", linewidths=1.2, zorder=5)
        axis.text(xy[-1, 0] + 2, xy[-1, 1] - 2, robot_id.replace("robot_", "R"), color=color, fontsize=7, fontweight="bold")
    ball_coordinates = [coordinate for coordinate in (_coord(point) for point in tracks.get("ball") or [] if point.get("visible", True)) if coordinate]
    if len(ball_coordinates) > 1:
        xy = np.asarray(ball_coordinates)
        axis.plot(xy[:, 0], xy[:, 1], color="#ff8b22", linewidth=0.9, alpha=0.65, linestyle="--")
    _save(figure, movement_path, 10.2, 4)

    # Team heatmaps use the corrected team assignment, not only point-level labels.
    team_points: dict[str, list[tuple[float, float]]] = {"aliado": [], "rival": []}
    for raw_robot_id, points in (tracks.get("robots") or {}).items():
        robot_id = str(raw_robot_id)
        team = teams.get(robot_id, "desconocido")
        if team not in team_points:
            continue
        team_points[team].extend(
            coordinate
            for coordinate in (_coord(point) for point in points if point.get("visible", True))
            if coordinate
        )
    ally_heatmap = output / "zona_control_magenta.png"
    rival_heatmap = output / "zona_control_azul.png"
    _draw_heatmap(ally_heatmap, team_points["aliado"], "magma", "SIN DATOS MAGENTA")
    _draw_heatmap(rival_heatmap, team_points["rival"], "viridis", "SIN DATOS AZUL")

    # Possession timeline: magenta, blue and free/unassigned are shown explicitly.
    possession_path = output / "grafica_posesion.png"
    ball_points = sorted(tracks.get("ball") or [], key=lambda point: float(point.get("timestamp_seconds", 0.0)))
    duration = max([float(point.get("timestamp_seconds", 0.0)) for point in ball_points], default=1.0)
    bin_count = max(40, min(160, int(duration * 6)))
    bins = np.linspace(0.0, max(duration, 1.0), bin_count + 1)
    ally = np.zeros(bin_count, dtype=np.float32)
    rival = np.zeros(bin_count, dtype=np.float32)
    free = np.zeros(bin_count, dtype=np.float32)
    for point in ball_points:
        timestamp = float(point.get("timestamp_seconds", 0.0))
        index = min(bin_count - 1, max(0, int(np.searchsorted(bins, timestamp, side="right") - 1)))
        owner = point.get("owner_robot_id")
        team = teams.get(str(owner), "desconocido") if owner is not None else "desconocido"
        if team == "aliado":
            ally[index] += 1
        elif team == "rival":
            rival[index] += 1
        else:
            free[index] += 1
    total = ally + rival + free
    empty = total <= 0
    total[empty] = 1.0
    ally_percent = 100.0 * ally / total
    rival_percent = 100.0 * rival / total
    free_percent = 100.0 * free / total
    if bin_count >= 7:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        ally_percent = np.convolve(ally_percent, kernel, mode="same")
        rival_percent = np.convolve(rival_percent, kernel, mode="same")
        free_percent = np.convolve(free_percent, kernel, mode="same")
        normalization = ally_percent + rival_percent + free_percent
        normalization[normalization <= 0] = 1.0
        ally_percent = 100.0 * ally_percent / normalization
        rival_percent = 100.0 * rival_percent / normalization
        free_percent = 100.0 * free_percent / normalization
    x_values = (bins[:-1] + bins[1:]) / 2.0
    figure, axis = plt.subplots(figsize=(8, 2.55))
    axis.stackplot(
        x_values,
        ally_percent,
        rival_percent,
        free_percent,
        colors=[MAGENTA, BLUE, FREE],
        alpha=[0.68, 0.68, 0.50],
        labels=["Magenta", "Azul", "Libre / sin asignar"],
    )
    axis.set_xlim(0, max(duration, 1.0))
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.16, linestyle="--")
    axis.set_ylabel("Estado del balón (%)", fontsize=8)
    axis.set_xlabel("Tiempo (s)", fontsize=8)
    axis.tick_params(labelsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=0.5)
    figure.savefig(possession_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    return {
        "movement": str(movement_path),
        "ally_heatmap": str(ally_heatmap),
        "rival_heatmap": str(rival_heatmap),
        "possession": str(possession_path),
    }
