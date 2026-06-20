from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.I_field_geometry.field_spec import FieldSpec


@dataclass(frozen=True)
class FieldTemplateConfig:
    """Normalized planar template for the real FutBotMX carpet.

    V11 removes the fictitious centre circle used by older versions and uses
    the measured 243 x 182 cm surface, 25 x 80 cm D-shaped areas and 60 cm
    goals.  The template remains normalized so the registration code can use
    it with any requested output scale.
    """

    goal_area_depth_ratio: float = 25.0 / 243.0
    goal_area_width_ratio: float = 80.0 / 182.0
    goal_area_radius_ratio: float = 25.0 / 243.0
    center_circle_radius_ratio: float = 0.0
    goal_width_ratio: float = 60.0 / 182.0
    center_line_ratio: float = 0.5
    include_center_circle: bool = False
    penalty_area_shape: str = "semiellipse"

    @classmethod
    def from_spec(cls, spec: FieldSpec) -> "FieldTemplateConfig":
        return cls(
            goal_area_depth_ratio=spec.penalty_depth_ratio,
            goal_area_width_ratio=spec.penalty_width_ratio,
            goal_area_radius_ratio=spec.penalty_depth_ratio,
            center_circle_radius_ratio=(
                0.5 * spec.center_circle_diameter_cm / spec.surface_length_cm
                if spec.center_circle_diameter_cm > 0
                else 0.0
            ),
            goal_width_ratio=spec.goal_width_ratio,
            center_line_ratio=spec.center_line_ratio,
            include_center_circle=spec.center_circle_diameter_cm > 0,
            penalty_area_shape=str(spec.penalty_area_shape),
        )


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
    """Return weighted marking samples in normalized carpet coordinates.

    Groups:
      0 outer carpet boundary
      1 centre line (and optional centre circle)
      2 near D-shaped penalty area
      3 far D-shaped penalty area
      4 goal mouths, used mainly by the hologram editor
    """

    cfg = config or FieldTemplateConfig()
    density = max(80, int(density))
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    def add(array: np.ndarray, weight: float, group: int) -> None:
        values = np.asarray(array, dtype=np.float64).reshape(-1, 2)
        points.append(values)
        weights.append(np.full(len(values), float(weight), dtype=np.float64))
        groups.append(np.full(len(values), int(group), dtype=np.int16))

    add(_sample_segment((0.0, 0.0), (1.0, 0.0), density), 1.55, 0)
    add(_sample_segment((1.0, 0.0), (1.0, 1.0), density), 1.55, 0)
    add(_sample_segment((1.0, 1.0), (0.0, 1.0), density), 1.55, 0)
    add(_sample_segment((0.0, 1.0), (0.0, 0.0), density), 1.55, 0)

    center_x = float(np.clip(cfg.center_line_ratio, 0.0, 1.0))
    add(_sample_segment((center_x, 0.0), (center_x, 1.0), int(0.9 * density)), 0.95, 1)
    if cfg.include_center_circle and cfg.center_circle_radius_ratio > 0:
        add(
            _sample_arc(
                (center_x, 0.5),
                cfg.center_circle_radius_ratio,
                cfg.center_circle_radius_ratio,
                0.0,
                360.0,
                int(1.25 * density),
            ),
            0.68,
            1,
        )

    half_width = 0.5 * float(cfg.goal_area_width_ratio)
    y_low = 0.5 - half_width
    y_high = 0.5 + half_width
    depth = float(cfg.goal_area_depth_ratio)

    if str(cfg.penalty_area_shape).lower() == "semiellipse":
        # The measured 25 x 80 cm marking is a D shape: its diameter lies on
        # the goal line and its furthest point is 25 cm into the field.
        add(
            _sample_arc((0.0, 0.5), depth, half_width, -90.0, 90.0, density),
            1.35,
            2,
        )
        add(
            _sample_arc((1.0, 0.5), depth, half_width, 90.0, 270.0, density),
            1.35,
            3,
        )
    else:
        radius = min(float(cfg.goal_area_radius_ratio), half_width, depth)
        add(_sample_segment((0.0, y_low), (depth - radius, y_low), density // 3), 1.18, 2)
        add(_sample_segment((0.0, y_high), (depth - radius, y_high), density // 3), 1.18, 2)
        add(_sample_arc((depth - radius, 0.5), radius, half_width, -90.0, 90.0, density), 1.28, 2)
        add(_sample_segment((1.0, y_low), (1.0 - depth + radius, y_low), density // 3), 1.18, 3)
        add(_sample_segment((1.0, y_high), (1.0 - depth + radius, y_high), density // 3), 1.18, 3)
        add(_sample_arc((1.0 - depth + radius, 0.5), radius, half_width, 90.0, 270.0, density), 1.28, 3)

    goal_half = 0.5 * float(cfg.goal_width_ratio)
    add(_sample_segment((0.0, 0.5 - goal_half), (0.0, 0.5 + goal_half), density // 3), 0.82, 4)
    add(_sample_segment((1.0, 0.5 - goal_half), (1.0, 0.5 + goal_half), density // 3), 0.82, 4)

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
        if np.linalg.norm(pixels[index] - pixels[index - 1]) > 0.12 * max(width, height):
            continue
        cv2.line(canvas, first, second, 255, max(1, int(thickness)), cv2.LINE_AA)
    return canvas
