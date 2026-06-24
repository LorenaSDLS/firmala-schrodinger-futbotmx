import argparse
from pathlib import Path
from typing import Callable

from src.A_pipeline.step_01_loadVideo.run import run_step_01

FrameProgressCallback = Callable[[int, int], None]


def print_quick_preview_summary(result: dict) -> None:
    print("\n" + "=" * 55)
    print(" PASO 02 - PREVISUALIZACIÓN YOLO")
    print("=" * 55)
    print(f"Video generado:          {result['preview_path']}")
    print(f"Detecciones JSONL:       {result['detections_path']}")
    print(f"Debug de tracking:       {result.get('tracking_debug_path')}")
    print(f"Detecciones rechazadas:  {result.get('rejected_detections_path')}")
    print(f"Cuadros procesados:      {result['processed_frames']}")
    print(f"Detecciones aceptadas:   {result['total_detections']}")
    print(f"Detecciones rechazadas:  {result.get('total_rejected_detections', 0)}")
    print(f"FPS usado:               {result['fps']:.2f}")
    print(f"Clasificación de equipos: {result.get('team_mode', 'none')}")
    print(f"Parejas bloqueadas:       {result.get('team_locked', False)}")
    print(f"Resumen de equipos:       {result.get('team_summary_path')}")
    print(f"Identidad física V5:      {result.get('identity_v5') is not None}")
    if result.get("identity_summary_path"):
        print(f"Diagnóstico V5:           {result['identity_summary_path']}")
    if result.get("online_preview_path"):
        print(f"Preview online original:  {result['online_preview_path']}")
    print(f"Estabilización de cámara:{result.get('camera_stabilization', False)}")
    print(f"Segmentación de cancha:  {result.get('field_segmentation_enabled', False)}")
    if result.get('field_geometry_debug_path'):
        print(f"Debug geometría:          {result['field_geometry_debug_path']}")
    if result.get('field_rectified_debug_path'):
        print(f"Vista cenital debug:      {result['field_rectified_debug_path']}")
    if result.get('field_homography_path'):
        print(f"Homografías JSONL:        {result['field_homography_path']}")
    if result.get('field_calibration_path'):
        print(f"Calibración de cancha:    {result['field_calibration_path']}")
    print(
        "Resolución:              "
        f"{result['resolution']['width']}x{result['resolution']['height']}"
    )
    print("=" * 55 + "\n")


def run_step_02(
    video_path: str | Path,
    confidence_threshold: float = 0.25,
    weights_path: str | Path | None = None,
    image_size: int = 640,
    max_frames: int | None = None,
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
    team_mode: str | None = None,
    ally_appearance: str | None = None,
    team_config_path: str | Path | None = None,
    camera_stabilization: bool = True,
    save_tracking_debug: bool = True,
    offline_identity_v5=False,
    #offline_identity_v5: bool = True,
    robot_interpolation_seconds: float = 0.42,
    progress_callback: FrameProgressCallback | None = None,
) -> dict:
    metadata, output_directory = run_step_01(video_path)
    if not metadata.is_valid:
        raise RuntimeError(
            "El video no es válido. No se puede generar la previsualización."
        )

    calibration_path = field_calibration_path
    mode = str(field_calibration_mode or "auto").lower()
    if mode == "assisted":
        from src.I_field_geometry.calibration_wizard import calibrate_video_interactively

        calibration_path = output_directory / "field_calibration.json"
        print("Abriendo calibración V8: etiqueta únicamente características visibles...")
        calibrate_video_interactively(
            video_path=video_path,
            output_path=calibration_path,
            frame_index=field_calibration_frame,
            field_width=field_canonical_width,
            field_height=field_canonical_height,
        )
    elif mode == "file" and not calibration_path:
        raise ValueError(
            "--field-calibration file requiere --field-calibration-file."
        )

    print("Importando generador de previsualización YOLO...")
    from src.C_quick_view.preview_generator import generate_quick_preview

    result = generate_quick_preview(
        video_path=video_path,
        output_directory=output_directory,
        confidence_threshold=confidence_threshold,
        weights_path=weights_path,
        image_size=image_size,
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
        field_calibration_path=calibration_path,
        team_mode=team_mode,
        ally_appearance=ally_appearance,
        team_config_path=team_config_path,
        camera_stabilization=camera_stabilization,
        save_tracking_debug=save_tracking_debug,
        offline_identity_v5=offline_identity_v5,
        robot_interpolation_seconds=robot_interpolation_seconds,
        progress_callback=progress_callback,
    )
    print_quick_preview_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una previsualización con YOLO y tracking temporal."
    )
    parser.add_argument("video_path", help="Ruta del video que se desea analizar.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=None)
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
    )
    parser.add_argument("--field-calibration-file", default=None)
    parser.add_argument("--field-calibration-frame", type=int, default=0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--team-mode", choices=["auto", "id", "none"], default=None)
    parser.add_argument(
        "--ally-appearance",
        choices=["claro", "oscuro"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--team-config", default=None)
    parser.add_argument("--no-camera-stabilization", action="store_true")
    parser.add_argument("--no-tracking-debug", action="store_true")
    parser.add_argument(
        "--no-offline-identity-v5",
        "--no-offline-identity-v4",
        dest="no_offline_identity_v5",
        action="store_true",
    )
    parser.add_argument("--robot-interpolation-seconds", type=float, default=0.42)
    arguments = parser.parse_args()

    run_step_02(
        video_path=arguments.video_path,
        confidence_threshold=arguments.conf,
        weights_path=arguments.weights,
        image_size=arguments.imgsz,
        max_frames=arguments.max_frames,
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
        team_mode=arguments.team_mode,
        ally_appearance=arguments.ally_appearance,
        team_config_path=arguments.team_config,
        camera_stabilization=not arguments.no_camera_stabilization,
        save_tracking_debug=not arguments.no_tracking_debug,
        offline_identity_v5=not arguments.no_offline_identity_v5,
        robot_interpolation_seconds=arguments.robot_interpolation_seconds,
    )


if __name__ == "__main__":
    main()
