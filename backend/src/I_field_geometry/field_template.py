from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldTemplateConfig:
    """Normalized geometric template used by registration and Mesa Replay.

    Exact competition dimensions can replace these ratios later without
    changing the registration pipeline.  The defaults match the markings seen
    in the current FutBot videos: outer rectangle, center line/circle and a
    rounded goal area at each longitudinal end.
    """

    goal_area_depth_ratio: float = 0.18
    goal_area_width_ratio: float = 0.56
    goal_area_radius_ratio: float = 0.10
    center_circle_radius_ratio: float = 0.12
    goal_width_ratio: float = 0.34


@dataclass(frozen=True)
class TemplatePointSet:
    points: np.ndarray
    weights: np.ndarray
    groups: np.ndarray


def _sample_segment(first: tuple[float, float], second: tuple[float, float], count: int) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    t = np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float64)[:, None]
    return (1.0 - t) * first_array + t * second_array


def _sample_arc(
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    start_degrees: float,
    end_degrees: float,
    count: int,
) -> np.ndarray:
    angles = np.deg2rad(np.linspace(start_degrees, end_degrees, max(3, int(count))))
    return np.column_stack(
        [
            center[0] + radius_x * np.cos(angles),
            center[1] + radius_y * np.sin(angles),
        ]
    )


def build_template_points(
    config: FieldTemplateConfig | None = None,
    density: int = 180,
) -> TemplatePointSet:
    """Return weighted line samples in normalized field coordinates.

    Groups:
      0 outer physical boundary
      1 center markings
      2 near goal area
      3 far goal area
    """

    cfg = config or FieldTemplateConfig()
    density = max(80, int(density))
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    def add(array: np.ndarray, weight: float, group: int) -> None:
        clipped = np.asarray(array, dtype=np.float64).reshape(-1, 2)
        points.append(clipped)
        weights.append(np.full(len(clipped), float(weight), dtype=np.float64))
        groups.append(np.full(len(clipped), int(group), dtype=np.int16))

    edge_samples = density
    add(_sample_segment((0.0, 0.0), (1.0, 0.0), edge_samples), 1.45, 0)
    add(_sample_segment((1.0, 0.0), (1.0, 1.0), edge_samples), 1.45, 0)
    add(_sample_segment((1.0, 1.0), (0.0, 1.0), edge_samples), 1.45, 0)
    add(_sample_segment((0.0, 1.0), (0.0, 0.0), edge_samples), 1.45, 0)

    add(_sample_segment((0.5, 0.0), (0.5, 1.0), int(0.9 * density)), 0.72, 1)
    circle = _sample_arc(
        (0.5, 0.5),
        cfg.center_circle_radius_ratio,
        cfg.center_circle_radius_ratio,
        0.0,
        360.0,
        int(1.25 * density),
    )
    add(circle, 0.68, 1)

    half_width = 0.5 * cfg.goal_area_width_ratio
    y_low = 0.5 - half_width
    y_high = 0.5 + half_width
    depth = cfg.goal_area_depth_ratio
    radius = min(cfg.goal_area_radius_ratio, half_width, depth)

    # Near goal area: two short horizontal segments and a rounded cap facing infield.
    add(_sample_segment((0.0, y_low), (depth - radius, y_low), density // 3), 1.18, 2)
    add(_sample_segment((0.0, y_high), (depth - radius, y_high), density // 3), 1.18, 2)
    add(
        _sample_arc(
            (depth - radius, 0.5),
            radius,
            half_width,
            -90.0,
            90.0,
            density,
        ),
        1.28,
        2,
    )

    # Far goal area mirrors the near side.
    add(_sample_segment((1.0, y_low), (1.0 - depth + radius, y_low), density // 3), 1.18, 3)
    add(_sample_segment((1.0, y_high), (1.0 - depth + radius, y_high), density // 3), 1.18, 3)
    add(
        _sample_arc(
            (1.0 - depth + radius, 0.5),
            radius,
            half_width,
            90.0,
            270.0,
            density,
        ),
        1.28,
        3,
    )

    return TemplatePointSet(
        points=np.vstack(points).astype(np.float32),
        weights=np.concatenate(weights).astype(np.float32),
        groups=np.concatenate(groups),
    )


def render_template(
    width: int,
    height: int,
    config: FieldTemplateConfig | None = None,
    thickness: int = 2,
) -> np.ndarray:
    canvas = np.zeros((int(height), int(width)), dtype=np.uint8)
    samples = build_template_points(config, density=max(width, height))
    pixels = np.round(
        np.column_stack(
            [samples.points[:, 0] * (width - 1), samples.points[:, 1] * (height - 1)]
        )
    ).astype(np.int32)
    for index in range(1, len(pixels)):
        if samples.groups[index] != samples.groups[index - 1]:
            continue
        first = tuple(pixels[index - 1])
        second = tuple(pixels[index])
        if np.linalg.norm(pixels[index] - pixels[index - 1]) > 0.08 * max(width, height):
            continue
        cv2.line(canvas, first, second, 255, max(1, int(thickness)), cv2.LINE_AA)
    return canvas
