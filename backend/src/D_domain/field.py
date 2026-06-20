from dataclasses import dataclass

from src.D_domain.geometry import BBox


@dataclass
class Field:
    bbox: BBox
    confidence: float

    def update(self, bbox: BBox, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence

    def contains_bbox_center(self, bbox: BBox) -> bool:
        cx, cy = bbox.center

        return (
            self.bbox.x1 <= cx <= self.bbox.x2
            and self.bbox.y1 <= cy <= self.bbox.y2
        )

    def contains_point(self, x: float, y: float) -> bool:
        return (
            self.bbox.x1 <= x <= self.bbox.x2
            and self.bbox.y1 <= y <= self.bbox.y2
        )