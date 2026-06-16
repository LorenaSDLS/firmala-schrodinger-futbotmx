from pathlib import Path


SUPPORTED_FORMATS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
}


def validate_video_file(video_path: Path) -> list[str]:
    errors = []

    if not video_path.exists():
        errors.append("El archivo no existe.")
        return errors

    if not video_path.is_file():
        errors.append("La ruta no corresponde a un archivo.")
        return errors

    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        errors.append(
            f"Formato no soportado: {video_path.suffix or 'sin extension'}."
        )

    if video_path.stat().st_size == 0:
        errors.append("El archivo esta vacio.")

    return errors


def validate_video_properties(
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    first_frame_read: bool,
) -> list[str]:
    errors = []

    if not first_frame_read:
        errors.append("No fue posible leer el primer frame.")

    if fps <= 0:
        errors.append("El video no tiene un FPS valido.")

    if total_frames <= 0:
        errors.append("El video no contiene frames validos.")

    if width <= 0 or height <= 0:
        errors.append("El video no tiene una resolucion valida.")

    return errors