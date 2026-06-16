from dataclasses import dataclass
from math import sqrt


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, values: list[float]) -> "BBox":
        return cls(
            x1=float(values[0]),
            y1=float(values[1]),
            x2=float(values[2]),
            y2=float(values[3]),
        )

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )

    def distance_to(self, other: "BBox") -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def expanded(self, margin: float) -> "BBox":
        return BBox(
            self.x1 - margin,
            self.y1 - margin,
            self.x2 + margin,
            self.y2 + margin,
        )

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]