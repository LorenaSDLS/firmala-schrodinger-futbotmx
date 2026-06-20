from dataclasses import dataclass

from src.D_domain.geometry import BBox


@dataclass
class Goal:
    goal_id: str
    side_image: str
    bbox: BBox
    confidence: float
    visible: bool = True
    frames_missing: int = 0
    field_x: float | None = None
    field_y: float | None = None
    field_x_norm: float | None = None
    field_y_norm: float | None = None
    field_transform_valid: bool = False
    field_transform_confidence: float = 0.0
    field_transform_source: str = "sin_calibracion"
    field_polygon: list[list[float]] | None = None

    def update(
        self,
        bbox: BBox,
        confidence: float,
        side_image: str | None = None,
        field_x: float | None = None,
        field_y: float | None = None,
        field_x_norm: float | None = None,
        field_y_norm: float | None = None,
        field_transform_valid: bool = False,
        field_transform_confidence: float = 0.0,
        field_transform_source: str = "sin_calibracion",
        field_polygon: list[list[float]] | None = None,
    ) -> None:
        self.bbox = bbox
        self.confidence = float(confidence)
        if side_image:
            self.side_image = side_image
        self.visible = True
        self.frames_missing = 0
        self.field_x = float(field_x) if field_x is not None else None
        self.field_y = float(field_y) if field_y is not None else None
        self.field_x_norm = float(field_x_norm) if field_x_norm is not None else None
        self.field_y_norm = float(field_y_norm) if field_y_norm is not None else None
        self.field_transform_valid = bool(field_transform_valid)
        self.field_transform_confidence = float(field_transform_confidence)
        self.field_transform_source = str(field_transform_source)
        self.field_polygon = field_polygon

    def mark_missing(self) -> None:
        self.frames_missing += 1
        if self.frames_missing > 3:
            self.visible = False

    def contains_ball_center(self, ball_bbox: BBox, inset_ratio: float = 0.04) -> bool:
        cx, cy = ball_bbox.center
        inset_x = self.bbox.width * max(0.0, min(0.35, inset_ratio))
        inset_y = self.bbox.height * max(0.0, min(0.35, inset_ratio))
        return (
            self.bbox.x1 + inset_x <= cx <= self.bbox.x2 - inset_x
            and self.bbox.y1 + inset_y <= cy <= self.bbox.y2 - inset_y
        )
