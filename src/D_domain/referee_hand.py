from dataclasses import dataclass

from src.D_domain.geometry import BBox


@dataclass
class RefereeHand:
    bbox: BBox
    confidence: float
    visible: bool = True

    def update(self, bbox: BBox, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.visible = True

    def is_near_bbox(self, bbox: BBox, max_distance: float = 100.0) -> bool:
        return self.bbox.distance_to(bbox) <= max_distance

    def overlaps_bbox(self, bbox: BBox, margin: float = 20.0) -> bool:
        return self.bbox.expanded(margin).intersects(bbox)