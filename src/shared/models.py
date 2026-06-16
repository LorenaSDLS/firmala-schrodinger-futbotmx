from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoMetadata:
    video_name: str
    original_filename: str
    format: str
    codec: str
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int
    status: str
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "original_filename": self.original_filename,
            "format": self.format,
            "codec": self.codec,
            "duration_seconds": round(self.duration_seconds, 3),
            "fps": round(self.fps, 3),
            "total_frames": self.total_frames,
            "resolution": {
                "width": self.width,
                "height": self.height,
                "formatted": f"{self.width}x{self.height}",
            },
            "status": self.status,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }