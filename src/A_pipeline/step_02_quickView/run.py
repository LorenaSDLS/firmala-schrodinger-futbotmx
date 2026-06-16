import argparse
from pathlib import Path

from src.A_pipeline.step_01_loadVideo.run import run_step_01
from src.C_quick_view.preview_generator import generate_quick_preview


def print_quick_preview_summary(result: dict) -> None:
    print("\n" + "=" * 55)
    print(" PASO 02 - QUICK PREVIEW HIBRIDO")
    print("=" * 55)
    print(f"Video generado:       {result['preview_path']}")
    print(f"Detecciones JSONL:    {result['detections_path']}")
    print(f"Frames procesados:    {result['processed_frames']}")
    print(f"Detecciones totales:  {result['total_detections']}")
    print(f"FPS usado:            {result['fps']:.2f}")
    print(f"Modo SAM:             {result['sam_mode'] or 'Desactivado'}")
    print(
        "Resolucion:           "
        f"{result['resolution']['width']}x"
        f"{result['resolution']['height']}"
    )
    print("=" * 55 + "\n")


def run_step_02(
    video_path: str | Path,
    confidence_threshold: float = 0.25,
    sam_mode: str | None = "LoHa",
    sam_confidence: float = 0.40,
    max_frames: int | None = None,
) -> dict:
    metadata, output_directory = run_step_01(video_path)

    if not metadata.is_valid:
        raise RuntimeError(
            "El video no es valido. No se puede generar el quick preview."
        )

    result = generate_quick_preview(
        video_path=video_path,
        output_directory=output_directory,
        confidence_threshold=confidence_threshold,
        sam_mode=sam_mode,
        sam_confidence=sam_confidence,
        max_frames=max_frames,
    )

    print_quick_preview_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un preview hibrido con YOLO y SAM3."
    )

    parser.add_argument(
        "video_path",
        help="Ruta del video que se desea analizar.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza para YOLO.",
    )

    parser.add_argument(
        "--sam-mode",
        choices=["LoHa", "DoRa", "none"],
        default="LoHa",
        help="Adaptador de SAM3. LoHa es el recomendado.",
    )

    parser.add_argument(
        "--sam-conf",
        type=float,
        default=0.40,
        help="Umbral de confianza para SAM3.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximo de frames a procesar para pruebas rapidas.",
    )

    arguments = parser.parse_args()

    sam_mode = (
        None
        if arguments.sam_mode == "none"
        else arguments.sam_mode
    )

    run_step_02(
        video_path=arguments.video_path,
        confidence_threshold=arguments.conf,
        sam_mode=sam_mode,
        sam_confidence=arguments.sam_conf,
        max_frames=arguments.max_frames,
    )


if __name__ == "__main__":
    main()