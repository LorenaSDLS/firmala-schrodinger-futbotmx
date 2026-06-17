import argparse
from pathlib import Path


def run_full_pipeline(
    video_path: str | Path,
    sam_mode: str | None = "LoHa",
    yolo_confidence: float = 0.25,
    sam_confidence: float = 0.18,
    frame_window: int = 0,
    max_frames: int | None = None,
    replay_fps: int = 30,
    replay_frame_stride: int = 1,
) -> dict:
    video_path = Path(video_path)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETO - FUTBOTMX")
    print("=" * 60)

    print("\n[1/6] Carga del video + quick preview YOLO")
    from src.A_pipeline.step_02_quickView.run import run_step_02

    step_02_result = run_step_02(
        video_path=video_path,
        confidence_threshold=yolo_confidence,
        max_frames=max_frames,
    )

    output_directory = Path(step_02_result["preview_path"]).parent

    print("\n[2/6] Extraccion de eventos, graficas y tracks")
    from src.A_pipeline.step_03_extract_events.run import run_step_03

    step_03_result = run_step_03(output_directory)

    step_04_result = None

    if sam_mode is not None:
        print("\n[3/6] Revision de mano del arbitro con SAM")
        from src.A_pipeline.step_04_refereeHand.run import run_step_04

        step_04_result = run_step_04(
            video_path=video_path,
            output_directory=output_directory,
            sam_mode=sam_mode,
            sam_confidence=sam_confidence,
            frame_window=frame_window,
        )
    else:
        print("\n[3/6] Revision de mano del arbitro omitida")

    print("\n[4/6] Exportacion JSON para Mesa/Unity")
    from src.E_events.unity_exporter import export_unity_mesa_json

    unity_mesa_json_path = export_unity_mesa_json(output_directory)

    print("\n[5/6] Generacion de animacion Mesa")
    from src.F_simulation.mesa_replay_exporter import export_mesa_replay_video

    mesa_replay_path = export_mesa_replay_video(
        json_path=unity_mesa_json_path,
        output_path=output_directory / "mesa_replay_events.mp4",
        fps=replay_fps,
        frame_stride=replay_frame_stride,
    )

    print("\n[6/6] Resumen final")
    print("=" * 60)
    print(" PIPELINE COMPLETO TERMINADO")
    print("=" * 60)
    print(f"Carpeta de salida:   {output_directory}")
    print(f"Preview YOLO:        {step_02_result['preview_path']}")
    print(f"Detecciones JSONL:   {step_02_result['detections_path']}")
    print(f"Eventos:             {step_03_result['events_path']}")
    print(f"Tracks:              {step_03_result['tracks_path']}")
    print(f"JSON Mesa/Unity:     {unity_mesa_json_path}")
    print(f"Animacion Mesa:      {mesa_replay_path}")

    if step_04_result is not None:
        print(f"Eventos referee:     {step_04_result['updated_events_path']}")
    else:
        print("Eventos referee:     omitido")

    print("=" * 60 + "\n")

    return {
        "output_directory": str(output_directory),
        "step_02": step_02_result,
        "step_03": step_03_result,
        "step_04": step_04_result,
        "unity_mesa_json_path": unity_mesa_json_path,
        "mesa_replay_path": mesa_replay_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de FutBotMX."
    )

    parser.add_argument("video_path")

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
        default=0,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--replay-fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--replay-frame-stride",
        type=int,
        default=1,
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
        replay_fps=arguments.replay_fps,
        replay_frame_stride=arguments.replay_frame_stride,
    )


if __name__ == "__main__":
    main()