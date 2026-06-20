from dataclasses import dataclass, field

from src.D_domain.geometry import BBox


@dataclass
class Robot:
    robot_id: str
    bbox: BBox
    confidence: float
    team: str = "desconocido"
    team_number: int | None = None
    display_name: str | None = None
    active: bool = True
    has_ball: bool = False
    frames_missing: int = 0
    predicted: bool = False
    last_observed_frame: int | None = None
    stabilized_x_px: float | None = None
    stabilized_y_px: float | None = None
    registration_valid: bool = False
    registration_quality: float = 0.0
    field_x: float | None = None
    field_y: float | None = None
    field_x_norm: float | None = None
    field_y_norm: float | None = None
    inside_surface: bool | None = None
    field_transform_valid: bool = False
    field_transform_confidence: float = 0.0
    field_transform_source: str = "sin_calibracion"
    position_history: list[tuple[int, float, float]] = field(default_factory=list)

    def update(
        self,
        frame_index: int,
        bbox: BBox,
        confidence: float,
        predicted: bool = False,
        missed_frames: int = 0,
        team: str | None = None,
        team_number: int | None = None,
        display_name: str | None = None,
        stabilized_x_px: float | None = None,
        stabilized_y_px: float | None = None,
        registration_valid: bool = False,
        registration_quality: float = 0.0,
        field_x: float | None = None,
        field_y: float | None = None,
        field_x_norm: float | None = None,
        field_y_norm: float | None = None,
        inside_surface: bool | None = None,
        field_transform_valid: bool = False,
        field_transform_confidence: float = 0.0,
        field_transform_source: str = "sin_calibracion",
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.active = True
        self.predicted = predicted
        self.frames_missing = max(0, int(missed_frames)) if predicted else 0
        if team:
            self.team = team
        if team_number is not None:
            self.team_number = int(team_number)
        if display_name:
            self.display_name = display_name
        self.stabilized_x_px = (
            float(stabilized_x_px) if stabilized_x_px is not None else None
        )
        self.stabilized_y_px = (
            float(stabilized_y_px) if stabilized_y_px is not None else None
        )
        self.registration_valid = bool(registration_valid)
        self.registration_quality = float(registration_quality)
        self.field_x = float(field_x) if field_x is not None else None
        self.field_y = float(field_y) if field_y is not None else None
        self.field_x_norm = float(field_x_norm) if field_x_norm is not None else None
        self.field_y_norm = float(field_y_norm) if field_y_norm is not None else None
        self.inside_surface = bool(inside_surface) if inside_surface is not None else None
        self.field_transform_valid = bool(field_transform_valid)
        self.field_transform_confidence = float(field_transform_confidence)
        self.field_transform_source = str(field_transform_source)

        if not predicted:
            self.last_observed_frame = frame_index

        cx, cy = bbox.bottom_center
        self.position_history.append((frame_index, cx, cy))

    def mark_missing(self) -> None:
        self.frames_missing += 1
        self.predicted = True
        if self.frames_missing > 15:
            self.active = False

    def distance_to_bbox(self, bbox: BBox) -> float:
        return self.bbox.distance_to(bbox)

    def is_near(self, bbox: BBox, max_distance: float = 80.0) -> bool:
        return self.distance_to_bbox(bbox) <= max_distance

    def speed_px_per_frame(self) -> float:
        if len(self.position_history) < 2:
            return 0.0
        _, x1, y1 = self.position_history[-2]
        _, x2, y2 = self.position_history[-1]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
