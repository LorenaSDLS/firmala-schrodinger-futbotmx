from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_FIELD_SPEC_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "field_spec.json"
)


@dataclass(frozen=True)
class FieldSpec:
    """Metric description of the competition playing surface.

    All planar markings are expressed in centimetres on the carpet plane.
    Walls and goals are stored for validation, but their upper edges must not
    be mixed with the carpet homography because they are not coplanar.
    """

    surface_length_cm: float = 243.0
    surface_width_cm: float = 182.0
    line_width_cm: float = 2.0
    center_line_x_cm: float = 121.5
    center_circle_diameter_cm: float = 0.0
    penalty_area_depth_cm: float = 25.0
    penalty_area_width_cm: float = 80.0
    penalty_area_shape: str = "semiellipse"
    goal_width_cm: float = 60.0
    goal_depth_cm: float = 10.0
    goal_height_cm: float = 10.0
    wall_min_height_cm: float = 22.0
    ball_diameter_cm: float = 4.2
    robot_max_width_cm: float = 18.0
    robot_max_length_cm: float = 18.0
    name: str = "FutBotMX competition field"
    version: int = 1

    def __post_init__(self) -> None:
        if self.surface_length_cm <= 0 or self.surface_width_cm <= 0:
            raise ValueError("Las dimensiones de la alfombra deben ser positivas.")
        if not 0.0 <= self.center_line_x_cm <= self.surface_length_cm:
            raise ValueError("La línea central está fuera de la cancha.")
        if self.penalty_area_depth_cm <= 0 or self.penalty_area_width_cm <= 0:
            raise ValueError("Las dimensiones del área penal deben ser positivas.")
        if self.penalty_area_depth_cm >= 0.5 * self.surface_length_cm:
            raise ValueError("El área penal es demasiado profunda para esta cancha.")
        if self.penalty_area_width_cm > self.surface_width_cm:
            raise ValueError("El área penal no puede ser más ancha que la cancha.")

    @property
    def aspect_ratio(self) -> float:
        return float(self.surface_length_cm / self.surface_width_cm)

    @property
    def center_line_ratio(self) -> float:
        return float(self.center_line_x_cm / self.surface_length_cm)

    @property
    def penalty_depth_ratio(self) -> float:
        return float(self.penalty_area_depth_cm / self.surface_length_cm)

    @property
    def penalty_width_ratio(self) -> float:
        return float(self.penalty_area_width_cm / self.surface_width_cm)

    @property
    def goal_width_ratio(self) -> float:
        return float(self.goal_width_cm / self.surface_width_cm)

    @property
    def ball_radius_cm(self) -> float:
        return float(0.5 * self.ball_diameter_cm)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldSpec":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FieldSpec":
        source = Path(path) if path is not None else DEFAULT_FIELD_SPEC_PATH
        if not source.exists():
            return cls()
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
