from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSettings:
    requested_profile: str
    resolved_profile: str
    cuda_available: bool
    field_segmentation_image_size: int
    field_segmentation_stride: int
    field_debug_stride: int


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_performance_settings(
    profile: str = "auto",
    field_segmentation_image_size: int | None = None,
    field_segmentation_stride: int | None = None,
    field_debug_stride: int | None = None,
) -> PerformanceSettings:
    """Resolve conservative defaults for GPU and CPU machines.

    Explicit CLI values always win.  ``auto`` uses a CPU-friendly profile on
    machines without CUDA and a balanced profile when CUDA is available.
    """

    requested = str(profile or "auto").strip().lower()
    if requested not in {"auto", "cpu", "balanced", "quality"}:
        raise ValueError(
            "performance_profile debe ser auto, cpu, balanced o quality"
        )

    cuda = _cuda_available()
    resolved = "balanced" if requested == "auto" and cuda else "cpu" if requested == "auto" else requested

    defaults = {
        # Debug rendering/encoding is intentionally sampled too.  The normal
        # quick preview remains full frame-rate; only the expensive geometry
        # diagnostics are held between refreshed debug frames.
        "cpu": (448, 6, 6),
        "balanced": (512, 3, 3),
        "quality": (640, 1, 1),
    }
    default_imgsz, default_stride, default_debug_stride = defaults[resolved]

    imgsz = int(field_segmentation_image_size or default_imgsz)
    stride = int(field_segmentation_stride or default_stride)
    debug_stride = int(field_debug_stride or default_debug_stride)

    return PerformanceSettings(
        requested_profile=requested,
        resolved_profile=resolved,
        cuda_available=cuda,
        field_segmentation_image_size=max(128, imgsz),
        field_segmentation_stride=max(1, stride),
        field_debug_stride=max(1, debug_stride),
    )
