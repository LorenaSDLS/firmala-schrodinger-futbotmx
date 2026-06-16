from dataclasses import dataclass, field

from src.D_domain.geometry import BBox


@dataclass
class Ball:
    bbox: BBox
    confidence: float
    visible: bool = True
    owner_robot_id: str | None = None
    frames_missing: int = 0
    position_history: list[tuple[int, float, float]] = field(default_factory=list)

    def update(self, frame_index: int, bbox: BBox, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.visible = True
        self.frames_missing = 0

        cx, cy = bbox.center
        self.position_history.append((frame_index, cx, cy))

    def mark_missing(self) -> None:
        self.frames_missing += 1

        if self.frames_missing > 10:
            self.visible = False

    def speed_px_per_frame(self) -> float:
        if len(self.position_history) < 2:
            return 0.0

        _, x1, y1 = self.position_history[-2]
        _, x2, y2 = self.position_history[-1]

        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5