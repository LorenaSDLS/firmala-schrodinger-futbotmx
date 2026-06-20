"""
python -m src.main_supr \
  "inputs/video-1088_singular_display.mov" \
  --sam-mode none \
  --performance-profile cpu \
  --field-seg-stride 12 \
  --no-field-debug \
  --no-tracking-debug \
  --no-offline-identity-v5 \
  --team-mode none \
  --no-camera-stabilization \
  --max-frames 60 \
  --fast
"""

import argparse
from pathlib import Path
from typing import Callable
from src.shared.paths import resolve_yolo_weights


ProgressCallback = Callable[..., None]


DEFAULT_TEAM_CONFIG = Path(__file__).resolve().parent.parent / "config" / "equipos.json"


def run_full_pipeline(
    video_path: str | Path,
    sam_mode: str | None = None, # "LoHa",
    yolo_confidence: float = 0.25,
    yolo_model: str = "v2",
    yolo_weights_path: str | Path | None = None,
    yolo_image_size: int = 640,
    robot_confidence: float = 0.55,
    ball_confidence: float = 0.35,
    field_confidence: float = 0.25,
    goal_confidence: float = 0.35,
    field_segmentation_weights: str | Path | None = None,
    field_segmentation_confidence: float = 0.25,
    field_segmentation_image_size: int | None = None,
    field_segmentation_stride: int | None = None,
    field_debug_stride: int | None = None,
    performance_profile: str = "auto",
    field_geometry_enabled: bool = True,
    field_debug: bool = True,
    field_canonical_width: float = 100.0,
    field_canonical_height: float = 60.0,
    field_calibration_mode: str = "auto",
    field_calibration_path: str | Path | None = None,
    field_calibration_frame: int = 0,
    sam_confidence: float = 0.18,
    frame_window: int = 0,
    max_frames: int | None = None,
    replay_fps: float | None = None,
    replay_frame_stride: int = 1,
    ball_interpolation_max_seconds: float = 8.0,
    team_mode: str | None = None,
    ally_appearance: str | None = None,
    team_config_path: str | Path | None = DEFAULT_TEAM_CONFIG,
    camera_stabilization: bool = True,
    save_tracking_debug: bool = True,
    offline_identity_v5: bool = True,
    robot_interpolation_seconds: float = 0.42,
    generate_narration: bool = False,
    generate_pdf: bool = True,
    generate_sample_video: bool = False,
    narration_engine: str = "edge",
    narration_voice: str | None = "es-MX-JorgeNeural",
    narration_rate: str = "+0%",
    narration_volume: str = "+0%",
    narration_max_events: int = 12,
    narration_coverage_ratio: float = 0.45,
    report_max_events: int = 8,
    fast_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
    generate_visual_reports: bool = True,
) -> dict:
    video_path = Path(video_path)

    total_steps = 4

    def report_progress(
        step: int,
        phase: str,
        phase_progress: float,
        overall_progress: float,
        message: str,
        current_frame: int | None = None,
        total_frames: int | None = None,
    ) -> None:
        if progress_callback is None:
            return

        progress_data = {
            "step": step,
            "total_steps": total_steps,
            "phase": phase,
            "phase_progress": round(
                max(0.0, min(100.0, phase_progress)),
                1,
            ),
            "overall_progress": round(
                max(0.0, min(100.0, overall_progress)),
                1,
            ),
            "message": message,
        }

        if current_frame is not None:
            progress_data["current_frame"] = current_frame

        if total_frames is not None:
            progress_data["total_frames"] = total_frames

        progress_callback(**progress_data)

    # El modo rápido siempre omite estos productos.
    if fast_mode:
        generate_visual_reports = True
        generate_narration = False
        generate_pdf = True
        generate_sample_video = False

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETO - FUTBOTMX")
    print("=" * 60)

    print("\n[1/6] Carga del video + previsualización YOLO")
    from src.A_pipeline.step_02_quickView.run import run_step_02

    resolved_weights = resolve_yolo_weights(yolo_model, yolo_weights_path)
    print(f"Detector principal:      {yolo_model.upper()}")


    print(f"Pesos YOLO:              {resolved_weights}")

    report_progress(
        step=1,
        phase="Analizando video",
        phase_progress=0,
        overall_progress=3,
        message="Validando y preparando el video.",
    )

    def report_yolo(
        processed_frames: int,
        total_frames: int,
    ) -> None:
        if total_frames > 0:
            percentage = (
                processed_frames / total_frames
            ) * 100.0
        else:
            percentage = 0.0

        report_progress(
            step=1,
            phase="Analizando video",
            phase_progress=percentage,
            overall_progress=5.0 + percentage * 0.75,
            message=(
                f"Procesados {processed_frames}/"
                f"{total_frames} frames."
            ),
            current_frame=processed_frames,
            total_frames=total_frames,
        )



    step_02_result = run_step_02(
        video_path=video_path,
        confidence_threshold=yolo_confidence,
        weights_path=resolved_weights,
        image_size=yolo_image_size,
        max_frames=max_frames,
        robot_confidence=robot_confidence,
        ball_confidence=ball_confidence,
        field_confidence=field_confidence,
        goal_confidence=goal_confidence,
        field_segmentation_weights=field_segmentation_weights,
        field_segmentation_confidence=field_segmentation_confidence,
        field_segmentation_image_size=field_segmentation_image_size,
        field_segmentation_stride=field_segmentation_stride,
        field_debug_stride=field_debug_stride,
        performance_profile=performance_profile,
        field_geometry_enabled=field_geometry_enabled,
        field_debug=field_debug,
        field_canonical_width=field_canonical_width,
        field_canonical_height=field_canonical_height,
        field_calibration_mode=field_calibration_mode,
        field_calibration_path=field_calibration_path,
        field_calibration_frame=field_calibration_frame,
        team_mode=team_mode,
        ally_appearance=ally_appearance,
        team_config_path=team_config_path,
        camera_stabilization=camera_stabilization,
        save_tracking_debug=save_tracking_debug,
        offline_identity_v5=offline_identity_v5,
        robot_interpolation_seconds=robot_interpolation_seconds,
        progress_callback=report_yolo,
    )

    report_progress(
        step=1,
        phase="Analizando video",
        phase_progress=100,
        overall_progress=80,
        message="Previsualización y tracking terminados.",
    )

    output_directory = Path(step_02_result["preview_path"]).parent

    print("\n[2/6] Extracción de eventos, gráficas y trayectorias")
    from src.A_pipeline.step_03_extract_events.run import run_step_03

    report_progress(
        step=2,
        phase="Generando eventos y trayectorias",
        phase_progress=0,
        overall_progress=81,
        message="Analizando detecciones y tracks.",
    )

    step_03_result = run_step_03(
        output_directory,
        generate_visual_reports=generate_visual_reports,
    )

    report_progress(
        step=2,
        phase="Generando eventos y trayectorias",
        phase_progress=100,
        overall_progress=92,
        message="Eventos y trayectorias terminados.",
    )
    step_04_result = None

    if sam_mode is not None:
        print("\n[3/6] Revisión de mano del árbitro con SAM")
        from src.A_pipeline.step_04_refereeHand.run import run_step_04

        step_04_result = run_step_04(
            video_path=video_path,
            output_directory=output_directory,
            sam_mode=sam_mode,
            sam_confidence=sam_confidence,
            frame_window=frame_window,
        )
    else:
        print("\n[3/6] Revisión de mano del árbitro omitida")

    print("\n[4/6] Exportación JSON para Mesa/Unity")
    from src.E_events.unity_exporter import export_unity_mesa_json

    report_progress(
        step=3,
        phase="Preparando datos para Unity",
        phase_progress=0,
        overall_progress=93,
        message="Generando el JSON del partido.",
    )

    unity_mesa_json_path = export_unity_mesa_json(output_directory)

    report_progress(
        step=3,
        phase="Preparando datos para Unity",
        phase_progress=100,
        overall_progress=98,
        message="JSON para Unity generado.",
    )

    mesa_replay_path = None

    if not fast_mode:
        print("\n[5/6] Generación de repetición de Mesa")

        from src.F_simulation.mesa_replay_exporter import (
            export_mesa_replay_video,
        )

        mesa_replay_path = export_mesa_replay_video(
            json_path=unity_mesa_json_path,
            output_path=output_directory / "mesa_replay_events.mp4",
            fps=replay_fps,
            frame_stride=replay_frame_stride,
            ball_interpolation_max_seconds=(
                ball_interpolation_max_seconds
            ),
        )
    else:
        print("\n[5/6] Repetición de Mesa omitida por fast_mode")
        mesa_replay_path = "omitido_por_modo_rapido"

    events_for_outputs = Path(
        step_04_result["updated_events_path"]
        if step_04_result is not None
        else step_03_result["events_path"]
    )

    narration_result = None
    if generate_narration or generate_sample_video:
        print("\n[6/8] Generación de narración WAV")
        from src.G_narration.run import run_narration
        narration_result = run_narration(
            events_path=events_for_outputs,
            video_path=video_path,
            output_directory=output_directory,
            preview_video_path=step_02_result["preview_path"],
            engine=narration_engine,
            voice=narration_voice,
            rate=narration_rate,
            volume=narration_volume,
            max_events=narration_max_events,
            maximum_coverage_ratio=narration_coverage_ratio,
            generate_sample_video=generate_sample_video,
        )

    report_result = None
    if generate_pdf:
        print("\n[7/8] Generación de infográfico PDF")
        from src.H_report.run import run_report
        report_result = run_report(
            output_directory=output_directory,
            events_path=events_for_outputs,
            summary_path=step_03_result["summary_path"],
            tracks_path=step_03_result["tracks_path"],
            max_featured_events=report_max_events,
        )

    print("\n[8/8] Resumen final")
    print("=" * 60)
    print(" PIPELINE COMPLETO TERMINADO")
    print("=" * 60)
    print(f"Carpeta de salida:       {output_directory}")
    print(f"Previsualización V5:     {step_02_result['preview_path']}")
    if step_02_result.get("online_preview_path"):
        print(f"Preview online original:{step_02_result['online_preview_path']}")
    if step_02_result.get("identity_summary_path"):
        print(f"Diagnóstico identidad:  {step_02_result['identity_summary_path']}")
    print(f"Detecciones JSONL:       {step_02_result['detections_path']}")
    print(f"Debug de tracking:       {step_02_result.get('tracking_debug_path')}")
    print(f"Detecciones rechazadas:  {step_02_result.get('rejected_detections_path')}")
    if step_02_result.get("field_geometry_debug_path"):
        print(f"Debug geometría cancha: {step_02_result['field_geometry_debug_path']}")
    if step_02_result.get("field_rectified_debug_path"):
        print(f"Vista cenital debug:     {step_02_result['field_rectified_debug_path']}")
    if step_02_result.get("field_homography_path"):
        print(f"Homografías:             {step_02_result['field_homography_path']}")
    print(f"Eventos:                 {step_03_result['events_path']}")
    print(f"Trayectorias:             {step_03_result['tracks_path']}")
    print(f"JSON Mesa/Unity:         {unity_mesa_json_path}")
    print(f"Repetición de mesa:      {mesa_replay_path}")
    if narration_result is not None:
        print(f"Narración WAV:           {narration_result['complete_wav_path']}")
        if narration_result.get("sample_video_path"):
            print(f"Video narrado:           {narration_result['sample_video_path']}")
    if report_result is not None:
        print(f"Infográfico PDF:         {report_result['pdf_path']}")
    if step_04_result is not None:
        print(f"Eventos de árbitro:      {step_04_result['updated_events_path']}")
    else:
        print("Eventos de árbitro:      omitido")
    print("=" * 60 + "\n")

    report_progress(
        step=4,
        phase="Finalizando análisis",
        phase_progress=100,
        overall_progress=100,
        message="Pipeline terminado correctamente.",
    )

    return {
        "output_directory": str(output_directory),
        "step_02": step_02_result,
        "step_03": step_03_result,
        "step_04": step_04_result,
        "unity_mesa_json_path": unity_mesa_json_path,
        "mesa_replay_path": mesa_replay_path,
        "events_for_outputs": str(events_for_outputs),
        "narration": narration_result,
        "report": report_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo de FutBotMX."
    )
    parser.add_argument("video_path")
    parser.add_argument("--sam-mode", choices=["LoHa", "DoRa", "none"], default="none")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-model", choices=["v2", "legacy"], default="v2")
    parser.add_argument("--yolo-weights", default=None, help="Ruta opcional a pesos personalizados.")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--robot-conf", type=float, default=0.55)
    parser.add_argument("--ball-conf", type=float, default=0.35)
    parser.add_argument("--field-conf", type=float, default=0.25)
    parser.add_argument("--goal-conf", type=float, default=0.35)
    parser.add_argument("--field-seg-weights", default=None)
    parser.add_argument("--field-seg-conf", type=float, default=0.25)
    parser.add_argument("--field-seg-imgsz", type=int, default=None)
    parser.add_argument("--field-seg-stride", type=int, default=None)
    parser.add_argument("--field-debug-stride", type=int, default=None)
    parser.add_argument(
        "--performance-profile",
        choices=["auto", "cpu", "balanced", "quality"],
        default="auto",
    )
    parser.add_argument("--no-field-geometry", action="store_true")
    parser.add_argument("--no-field-debug", action="store_true")
    parser.add_argument("--field-width", type=float, default=100.0)
    parser.add_argument("--field-height", type=float, default=60.0)
    parser.add_argument(
        "--field-calibration",
        choices=["auto", "assisted", "file"],
        default="auto",
        help="auto: solver multiseñal; assisted: etiqueta solo líneas visibles; file: reutiliza JSON.",
    )
    parser.add_argument("--field-calibration-file", default=None)
    parser.add_argument("--field-calibration-frame", type=int, default=0)
    parser.add_argument("--sam-conf", type=float, default=0.18)
    parser.add_argument("--frame-window", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--replay-fps", type=float, default=None)
    parser.add_argument("--replay-frame-stride", type=int, default=1)
    parser.add_argument(
        "--ball-interpolation-max-seconds",
        "--interpolacion-balon-max-segundos",
        dest="ball_interpolation_max_seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument("--team-mode", choices=["auto", "id", "none"], default=None)
    parser.add_argument(
        "--ally-appearance",
        "--apariencia-aliada",
        dest="ally_appearance",
        choices=["claro", "oscuro"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--team-config",
        "--config-equipos",
        dest="team_config",
        default=str(DEFAULT_TEAM_CONFIG),
    )
    parser.add_argument("--no-camera-stabilization", action="store_true")
    parser.add_argument("--no-tracking-debug", action="store_true")
    parser.add_argument(
        "--no-offline-identity-v5",
        "--no-offline-identity-v4",
        dest="no_offline_identity_v5",
        action="store_true",
        help="Desactiva la reconstrucción global conservadora de robots físicos.",
    )
    parser.add_argument("--narration", "--narracion", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--sample-video", "--video-muestra", action="store_true")
    parser.add_argument("--narration-engine", choices=["edge", "gtts", "windows"], default="edge")
    parser.add_argument("--narration-voice", default="es-MX-JorgeNeural")
    parser.add_argument("--narration-rate", default="+0%")
    parser.add_argument("--narration-volume", default="+0%")
    parser.add_argument("--narration-max-events", type=int, default=12)
    parser.add_argument("--narration-coverage-ratio", type=float, default=0.45)
    parser.add_argument("--report-max-events", type=int, default=8)
    parser.add_argument(
        "--robot-interpolation-seconds",
        "--interpolacion-robot-segundos",
        dest="robot_interpolation_seconds",
        type=float,
        default=0.42,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Omite Mesa, narración, video narrado y PDF.",
    )
    arguments = parser.parse_args()

    sam_mode = None if arguments.sam_mode == "none" else arguments.sam_mode
    team_config = arguments.team_config if arguments.team_config else None

    run_full_pipeline(
        video_path=arguments.video_path,
        sam_mode=sam_mode,
        yolo_confidence=arguments.yolo_conf,
        yolo_model=arguments.yolo_model,
        yolo_weights_path=arguments.yolo_weights,
        yolo_image_size=arguments.yolo_imgsz,
        robot_confidence=arguments.robot_conf,
        ball_confidence=arguments.ball_conf,
        field_confidence=arguments.field_conf,
        goal_confidence=arguments.goal_conf,
        field_segmentation_weights=arguments.field_seg_weights,
        field_segmentation_confidence=arguments.field_seg_conf,
        field_segmentation_image_size=arguments.field_seg_imgsz,
        field_segmentation_stride=arguments.field_seg_stride,
        field_debug_stride=arguments.field_debug_stride,
        performance_profile=arguments.performance_profile,
        field_geometry_enabled=not arguments.no_field_geometry,
        field_debug=not arguments.no_field_debug,
        field_canonical_width=arguments.field_width,
        field_canonical_height=arguments.field_height,
        field_calibration_mode=arguments.field_calibration,
        field_calibration_path=arguments.field_calibration_file,
        field_calibration_frame=arguments.field_calibration_frame,
        sam_confidence=arguments.sam_conf,
        frame_window=arguments.frame_window,
        max_frames=arguments.max_frames,
        replay_fps=arguments.replay_fps,
        replay_frame_stride=arguments.replay_frame_stride,
        ball_interpolation_max_seconds=arguments.ball_interpolation_max_seconds,
        team_mode=arguments.team_mode,
        ally_appearance=arguments.ally_appearance,
        team_config_path=team_config,
        camera_stabilization=not arguments.no_camera_stabilization,
        save_tracking_debug=not arguments.no_tracking_debug,
        offline_identity_v5=not arguments.no_offline_identity_v5,
        robot_interpolation_seconds=arguments.robot_interpolation_seconds,
        generate_narration=arguments.narration,
        generate_pdf=arguments.pdf,
        generate_sample_video=arguments.sample_video,
        narration_engine=arguments.narration_engine,
        narration_voice=arguments.narration_voice,
        narration_rate=arguments.narration_rate,
        narration_volume=arguments.narration_volume,
        narration_max_events=arguments.narration_max_events,
        narration_coverage_ratio=arguments.narration_coverage_ratio,
        report_max_events=arguments.report_max_events,
        fast_mode=arguments.fast,
    )

    try:
        import winsound

        winsound.Beep(2500, 1000)
    except (ImportError, RuntimeError):
        pass


if __name__ == "__main__":
    main()
