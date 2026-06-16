
#recibe el path del video

#detecta el nombre y crea una carpeta

#json con datos: nombre_video, original, formato, duración, fps, total frames, resolution, status

#en la terminal imprimir los datos 

import argparse
import json
from pathlib import Path

from src.B_load_video.video_analyzer import analyze_video
from src.shared.models import VideoMetadata
from src.shared.paths import get_video_outputdirec


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} s"


def save_metadata(
    metadata: VideoMetadata,
    output_directory: Path,
) -> Path:
    metadata_path = output_directory / "video_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata.to_dict(),
            file,
            indent=4,
            ensure_ascii=False,
        )

    return metadata_path


def print_video_summary(
    metadata: VideoMetadata,
    metadata_path: Path,
) -> None:
    status = "VALIDO" if metadata.is_valid else "INVALIDO"

    print("\n" + "=" * 55)
    print(" PASO 01 - CARGA Y ANALISIS DEL VIDEO")
    print("=" * 55)
    print(f"Nombre:      {metadata.original_filename}")
    print(f"Formato:     {metadata.format}")
    print(f"Codec:       {metadata.codec}")
    print(f"Duracion:    {format_duration(metadata.duration_seconds)}")
    print(f"FPS:         {metadata.fps:.2f}")
    print(f"Frames:      {metadata.total_frames}")
    print(f"Resolucion:  {metadata.width}x{metadata.height}")
    print(f"Estado:      {status}")
    print(f"Datos JSON:  {metadata_path}")

    if metadata.validation_errors:
        print("\nProblemas encontrados:")

        for error in metadata.validation_errors:
            print(f"- {error}")

    print("=" * 55 + "\n")


def run_step_01(video_path: str | Path) -> tuple[VideoMetadata, Path]:
    metadata = analyze_video(video_path)

    output_directory = get_video_outputdirec(video_path)
    metadata_path = save_metadata(metadata, output_directory)

    print_video_summary(metadata, metadata_path)

    return metadata, output_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carga, valida y analiza los datos basicos de un video."
    )
    parser.add_argument(
        "video_path",
        help="Ruta del video que se desea analizar.",
    )

    arguments = parser.parse_args()
    run_step_01(arguments.video_path)


if __name__ == "__main__":
    main()