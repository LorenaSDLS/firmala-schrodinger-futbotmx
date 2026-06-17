import argparse
from pathlib import Path

from src.A_pipeline.step_02_quickView.run import run_step_02
from src.A_pipeline.step_03_extract_events.run import run_step_03
from src.A_pipeline.step_04_refereeHand.run import run_step_04
#from src.shared.paths import get_video_outputdirec


def run_full_pipeline(
    video_path: str | Path,
    sam_mode: str = "LoHa",
    yolo_confidence: float = 0.25,
    sam_confidence: float = 0.18,
    frame_window: int = 20,
    max_frames: int | None = None,
) -> dict:
    video_path = Path(video_path)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETO - FUTBOTMX")
    print("=" * 60)

    print("\n[1/4] Carga del video + quick preview")
    step_02_result = run_step_02(
        video_path=video_path,
        confidence_threshold=yolo_confidence,
        sam_mode=None,  # YOLO rapido, sin SAM aqui
        sam_confidence=sam_confidence,
        max_frames=max_frames,
    )

    output_directory = Path(step_02_result["preview_path"]).parent

    print("\n[2/4] Extraccion de eventos y graficas")
    step_03_result = run_step_03(output_directory)

    print("\n[3/4] Revision de mano del arbitro")
    step_04_result = None

    if sam_mode is not None:
        print("\n[3/4] Revision de mano del arbitro")
        step_04_result = run_step_04(
            video_path=video_path,
            output_directory=output_directory,
            sam_mode=sam_mode,
            sam_confidence=sam_confidence,
            frame_window=frame_window,
        )
    else:
        print("\n[3/4] Revision de mano del arbitro omitida: SAM desactivado.")

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETO TERMINADO")
    print("=" * 60)
    print(f"Carpeta de salida: {output_directory}")
    print(f"Preview:           {step_02_result['preview_path']}")
    print(f"Eventos:           {step_03_result['events_path']}")
    print(f"Graficas:          {step_03_result['possession_chart_path']}")
    print(f"Mano referee:      {step_04_result['preview_path']}")
    print("=" * 60 + "\n")

    return {
        "output_directory": str(output_directory),
        "step_02": step_02_result,
        "step_03": step_03_result,
        "step_04": step_04_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de FutBotMX."
    )

    parser.add_argument(
        "video_path",
        help="Ruta del video que se desea analizar.",
    )

    parser.add_argument(
        "--sam-mode",
        choices=["LoHa", "DoRa", "none"],
        default="LoHa",
    )

    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--sam-conf",
        type=float,
        default=0.18,
    )

    parser.add_argument(
        "--frame-window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    sam_mode = None if arguments.sam_mode == "none" else arguments.sam_mode

    run_full_pipeline(
        video_path=arguments.video_path,
        sam_mode=sam_mode,
        yolo_confidence=arguments.yolo_conf,
        sam_confidence=arguments.sam_conf,
        frame_window=arguments.frame_window,
        max_frames=arguments.max_frames,
    )


if __name__ == "__main__":
    main()