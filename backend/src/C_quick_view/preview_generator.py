from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np
from tqdm.auto import tqdm


FrameProgressCallback = Callable[[int, int], None]

def _camera_motion_metrics(
    reference_matrix: np.ndarray | None,
    current_matrix: np.ndarray | None,
    frame_width: int,
    frame_height: int,
) -> dict[str, float]:
    """
    Mide cuánto cambió la cámara desde la última geometría aceptada.
    Usa traslación, rotación y zoom aproximado.
    """
    if reference_matrix is None or current_matrix is None:
        return {
            "translation_fraction": 999.0,
            "rotation_degrees": 999.0,
            "zoom_delta": 999.0,
        }

    try:
        reference = np.asarray(reference_matrix, dtype=np.float64)
        current = np.asarray(current_matrix, dtype=np.float64)
        relative = np.linalg.inv(reference) @ current
    except Exception:
        return {
            "translation_fraction": 999.0,
            "rotation_degrees": 999.0,
            "zoom_delta": 999.0,
        }

    linear = relative[:2, :2]
    determinant = float(np.linalg.det(linear))

    if determinant <= 1e-9:
        scale = 1.0
    else:
        scale = float(np.sqrt(determinant))

    rotation = float(
        np.degrees(
            np.arctan2(
                linear[1, 0],
                linear[0, 0],
            )
        )
    )

    diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
    translation = float(np.linalg.norm(relative[:2, 2])) / diagonal

    return {
        "translation_fraction": translation,
        "rotation_degrees": rotation,
        "zoom_delta": abs(scale - 1.0),
    }


def _camera_changed_a_lot(
    metrics: dict[str, float],
) -> bool:
    """
    Umbrales agresivos: solo recalibra si la cámara cambió bastante.
    """
    return (
        abs(metrics["rotation_degrees"]) >= 4.0
        or metrics["zoom_delta"] >= 0.08
        or metrics["translation_fraction"] >= 0.12
    )


def _select_field_box_for_motion(
    detections: list[dict[str, Any]],
) -> list[float] | None:
    fields = [
        detection
        for detection in detections
        if str(detection.get("class_group", "")).lower() == "field"
        and detection.get("bbox_xyxy")
    ]

    if not fields:
        return None

    selected = max(
        fields,
        key=lambda item: float(item.get("confidence", 0.0)),
    )

    return list(map(float, selected["bbox_xyxy"]))


def _field_box_changed_a_lot(
    previous_box: list[float] | None,
    current_box: list[float] | None,
    frame_width: int,
    frame_height: int,
) -> bool:
    """
    Respaldo barato: si la caja YOLO de la cancha cambió mucho,
    probablemente cambió perspectiva/zoom/encuadre.
    """
    if previous_box is None or current_box is None:
        return False

    px1, py1, px2, py2 = previous_box
    cx1, cy1, cx2, cy2 = current_box

    previous_center = np.array(
        [(px1 + px2) * 0.5, (py1 + py2) * 0.5],
        dtype=np.float64,
    )
    current_center = np.array(
        [(cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5],
        dtype=np.float64,
    )

    diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
    center_delta = float(np.linalg.norm(current_center - previous_center)) / diagonal

    previous_width = max(1.0, px2 - px1)
    previous_height = max(1.0, py2 - py1)
    current_width = max(1.0, cx2 - cx1)
    current_height = max(1.0, cy2 - cy1)

    previous_area = previous_width * previous_height
    current_area = current_width * current_height

    area_delta = abs(np.log(current_area / max(previous_area, 1.0)))

    previous_aspect = previous_width / previous_height
    current_aspect = current_width / current_height
    aspect_delta = abs(np.log(current_aspect / max(previous_aspect, 1e-6)))

    return (
        center_delta >= 0.10
        or area_delta >= 0.18
        or aspect_delta >= 0.16
    )

from src.C_quick_view.team_classifier import TeamClassifier
from src.C_quick_view.offline_identity import (
    OfflineIdentityConfig,
    reconstruct_physical_identities,
)
from src.C_quick_view.temporal_tracker import FutbotTemporalTracker
from src.C_quick_view.yolo_detector import YOLODetector, draw_yolo_detections
from src.F_simulation.field_registration import FieldRegistration
from src.shared.paths import FIELD_SEGMENTATION_WEIGHTS_PATH
from src.shared.performance import resolve_performance_settings
from src.I_field_geometry.field_segmenter import FieldSegmenter
from src.I_field_geometry.field_geometry import (
    FieldGeometryEstimator,
    draw_field_geometry_overlay,
    render_rectified_debug,
)


def generate_quick_preview(
    video_path: str | Path,
    output_directory: str | Path,
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
    field_calibration_path: str | Path | None = None,
    team_mode: str | None = None,
    ally_appearance: str | None = None,
    team_config_path: str | Path | None = None,
    camera_stabilization: bool = True,
    save_tracking_debug: bool = True,
    offline_identity_v5: bool = True,
    robot_interpolation_seconds: float = 0.42,
    progress_callback: FrameProgressCallback | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    preview_path = output_directory / "quick_preview.mp4"
    detections_path = output_directory / "quick_detections.jsonl"
    rejected_path = output_directory / "rejected_detections.jsonl"
    tracking_debug_path = output_directory / "tracking_debug.jsonl"
    team_summary_path = output_directory / "team_clustering.json"
    field_geometry_debug_path = output_directory / "field_geometry_debug.mp4"
    field_rectified_debug_path = output_directory / "field_rectified_debug.mp4"
    field_homography_path = output_directory / "field_homography.jsonl"

    performance = resolve_performance_settings(
        profile=performance_profile,
        field_segmentation_image_size=field_segmentation_image_size,
        field_segmentation_stride=field_segmentation_stride,
        field_debug_stride=field_debug_stride,
    )
    print(
        "Perfil de rendimiento: "
        f"{performance.resolved_profile.upper()} | "
        f"segmentador {performance.field_segmentation_image_size}px cada "
        f"{performance.field_segmentation_stride} frame(s)"
    )

    detector_kwargs = {
        "confidence_threshold": confidence_threshold,
        "image_size": image_size,
        "class_thresholds": {
            "robot": robot_confidence,
            "ball": ball_confidence,
            "field": field_confidence,
            "goal": goal_confidence,
        },
    }
    if weights_path is not None:
        detector_kwargs["weights_path"] = weights_path
    detector = YOLODetector(**detector_kwargs)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    tracker = FutbotTemporalTracker(
        fps=fps,
        frame_width=width,
        frame_height=height,
        max_robots=4,
    )
    team_classifier = TeamClassifier(
        mode=team_mode,
        ally_appearance=ally_appearance,
        config_path=team_config_path,
    )
    registration = FieldRegistration(
        frame_width=width,
        frame_height=height,
        enabled=camera_stabilization,
    )

    segmenter = None
    field_geometry = FieldGeometryEstimator(
        frame_width=width,
        frame_height=height,
        field_width=field_canonical_width,
        field_height=field_canonical_height,
        calibration_path=field_calibration_path,
    )
    if field_geometry_enabled:
        segmentation_weights = Path(
            field_segmentation_weights or FIELD_SEGMENTATION_WEIGHTS_PATH
        ).expanduser().resolve()
        if segmentation_weights.exists():
            segmenter = FieldSegmenter(
                weights_path=segmentation_weights,
                confidence_threshold=field_segmentation_confidence,
                image_size=performance.field_segmentation_image_size,
            )
        else:
            print(
                "Aviso: no se encontró el segmentador de cancha en "
                f"{segmentation_weights}. Se omite la homografía por máscara."
            )

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"No se pudo crear el video de salida: {preview_path}")


    geometry_writer = None
    rectified_writer = None
    rectified_size = (1200, 600)
    if field_debug and segmenter is not None:
        geometry_writer = cv2.VideoWriter(
            str(field_geometry_debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        rectified_writer = cv2.VideoWriter(
            str(field_rectified_debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            rectified_size,
        )

    limit = total_frames
    
    if max_frames is not None:
        limit = (
            min(total_frames, int(max_frames)) 
            if total_frames > 0 
            else int(max_frames))

    if progress_callback is not None:
        progress_callback(0, limit)

    frame_index = 0
    processed_frames = 0
    total_detections = 0
    total_rejected = 0
    last_segmentation_result = None
    frames_since_segmentation = 10_000

    geometry_attempted_once = False
    frames_since_geometry_attempt = 10_000
    frames_since_accepted_geometry = 10_000
    last_geometry_registration_matrix = None
    last_geometry_field_box = None
    geometry_recalculation_calls = 0


# Umbrales. Puedes subirlos si quieres recalibrar todavía menos.
    minimum_geometry_gap_frames = max(
        24,
        int(performance.field_segmentation_stride),
    )
    geometry_retry_frames = max(
        60,
        int(round(fps * 1.5)),
    )
    geometry_safety_refresh_frames = max(
        300,
        int(round(fps * 8.0)),

    )

    last_geometry_debug_frame = None
    last_rectified_debug_frame = None
    segmentation_calls = 0
    debug_refreshes = 0
    stage_seconds = {
        "detector": 0.0,
        "segmenter": 0.0,
        "geometry": 0.0,
        "tracking": 0.0,
        "render_and_output": 0.0,
        "total": 0.0,
    }

    debug_file = (
        tracking_debug_path.open("w", encoding="utf-8")
        if save_tracking_debug
        else None
    )
    rejected_file = (
        rejected_path.open("w", encoding="utf-8")
        if save_tracking_debug
        else None
    )
    homography_file = (
        field_homography_path.open("w", encoding="utf-8")
        if segmenter is not None
        else None
    )

    progress = tqdm(
        total=limit if limit > 0 else None,
        desc="Analizando video",
        unit="frame",
    )

    try:
        with detections_path.open("w", encoding="utf-8") as detections_file:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if max_frames is not None and processed_frames >= max_frames:
                    break

                frame_started = perf_counter()
                # The detector runs every frame because the ball and robots move
                # quickly.  The large field mask is sampled less often and its
                # homography is propagated between measurements.
                stage_started = perf_counter()
                raw_detections = detector.detect_frame(frame)
                stage_seconds["detector"] += perf_counter() - stage_started
                dynamic_boxes = [
                    detection.get("bbox_xyxy", [])
                    for detection in raw_detections
                    if str(detection.get("class_group", "")).lower()
                    in {"robot", "ball"}
                ]

                frames_since_segmentation += 1
                frames_since_geometry_attempt += 1
                frames_since_accepted_geometry += 1
                segmentation_result = None
                geometry_recalculation_reason = None
                camera_motion_metrics = {
                    "translation_fraction": 0.0,
                    "rotation_degrees": 0.0,
                    "zoom_delta": 0.0,
                }          

                stage_started = perf_counter()

                # Registro ligero de cámara. Esto corre cada frame, pero es mucho más barato
                # que recalcular toda la geometría.
                registration_result = registration.update(
                    frame,
                    semantic_mask=None,
                    exclusion_boxes=dynamic_boxes,
                    frame_index=frame_index,
                    )
                current_field_box = _select_field_box_for_motion(raw_detections)

                if (
                    field_geometry_enabled
                    and segmenter is not None
            ):
                    can_attempt_geometry = (
                        not geometry_attempted_once
                        or frames_since_geometry_attempt >= minimum_geometry_gap_frames
                    )

                    first_geometry_attempt = not geometry_attempted_once

                    camera_changed_lot = False
                    if (
                        last_geometry_registration_matrix is not None
                        and bool(registration_result.valid)
                        and bool(registration_result.updated)
                    ):
                        camera_motion_metrics = _camera_motion_metrics(
                            reference_matrix=last_geometry_registration_matrix,
                            current_matrix=registration_result.matrix,
                            frame_width=width,
                            frame_height=height,
                        )
                        camera_changed_lot = _camera_changed_a_lot(camera_motion_metrics)

                    field_box_changed_lot = _field_box_changed_a_lot(
                        previous_box=last_geometry_field_box,
                        current_box=current_field_box,
                        frame_width=width,
                        frame_height=height,
                    )

                    retry_due = (
                        field_geometry.needs_reacquisition
                        and frames_since_geometry_attempt >= geometry_retry_frames
                    )

                    safety_refresh_due = (
                        frames_since_accepted_geometry >= geometry_safety_refresh_frames
                    )

                    if first_geometry_attempt:
                        geometry_recalculation_reason = "primer_frame"
                    elif camera_changed_lot:
                        geometry_recalculation_reason = "camara_movida_mucho"
                    elif field_box_changed_lot:
                        geometry_recalculation_reason = "bbox_cancha_cambio_mucho"
                    elif retry_due:
                        geometry_recalculation_reason = "reintento_sin_geometria_valida"
                    elif safety_refresh_due:
                        geometry_recalculation_reason = "refresco_seguridad"

                    should_recalculate_geometry = (
                        can_attempt_geometry
                        and geometry_recalculation_reason is not None
                    )

                    if should_recalculate_geometry:
                        segment_started = perf_counter()
                        segmentation_result = segmenter.segment_frame(frame)
                        stage_seconds["segmenter"] += perf_counter() - segment_started

                        segmentation_calls += 1
                        geometry_recalculation_calls += 1
                        frames_since_segmentation = 0
                        frames_since_geometry_attempt = 0
                        geometry_attempted_once = True

                        if segmentation_result is not None:
                            last_segmentation_result = segmentation_result
                    else:
                        should_recalculate_geometry = False

                    # Lo importante:
                    # - Si estamos recalculando, sí mandamos frame/máscara/líneas.
                    # - Si NO estamos recalculando, mandamos frame=None para evitar análisis pesado.
                    if field_geometry_enabled:
                        geometry_result = field_geometry.update(
                            segmentation=segmentation_result,
                            current_to_reference=registration_result.matrix,
                            frame=frame if segmentation_result is not None else None,
                            goal_detections=[
                                detection
                                for detection in raw_detections
                                if str(detection.get("class_group", "")).lower() == "goal"
                            ] if segmentation_result is not None else [],
                            exclusion_boxes=dynamic_boxes if segmentation_result is not None else [],
                            frame_index=frame_index,
                        )
                    else:
                        geometry_result = field_geometry.last_result

# Si la geometría recalculada fue válida, este frame se vuelve la nueva
# referencia para medir futuros cambios fuertes de cámara.
                    if should_recalculate_geometry:
                        if bool(getattr(geometry_result, "valid", False)):
                            last_geometry_registration_matrix = np.asarray(
                                registration_result.matrix,
                                dtype=np.float64,
                            ).copy()
                            last_geometry_field_box = current_field_box
                            frames_since_accepted_geometry = 0

                    stage_seconds["geometry"] += perf_counter() - stage_started

                stage_started = perf_counter()
                tracked_detections = tracker.update(raw_detections, frame=frame)
                tracked_detections = team_classifier.update(frame, tracked_detections)
                detections = []
                for detection in tracked_detections:
                    annotated = registration.annotate_detection(detection)
                    annotated = field_geometry.annotate_detection(annotated)
                    detections.append(annotated)
                total_detections += len(detections)
                stage_seconds["tracking"] += perf_counter() - stage_started

                stage_started = perf_counter()
                annotated_frame = draw_yolo_detections(frame, detections)
                writer.write(annotated_frame)

                refresh_field_debug = (
                    processed_frames % performance.field_debug_stride == 0
                )
                if refresh_field_debug:
                    debug_refreshes += 1
                if geometry_writer is not None:
                    if refresh_field_debug or last_geometry_debug_frame is None:
                        last_geometry_debug_frame = draw_field_geometry_overlay(
                            annotated_frame,
                            segmentation_result or last_segmentation_result,
                            geometry_result,
                        )
                    geometry_writer.write(last_geometry_debug_frame)
                if rectified_writer is not None:
                    if refresh_field_debug or last_rectified_debug_frame is None:
                        last_rectified_debug_frame = render_rectified_debug(
                            frame,
                            geometry_result,
                            detections,
                            segmentation=segmentation_result or last_segmentation_result,
                            output_width=1000,
                            output_height=600,
                        )
                    rectified_writer.write(last_rectified_debug_frame)

                frame_record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "camera_registration": registration_result.to_dict(),
                    "field_segmentation": (
                        segmentation_result.to_dict()
                        if segmentation_result is not None
                        else None
                    ),
                    "field_geometry": geometry_result.to_dict(),
                    "detections": detections,
                }
                detections_file.write(
                    json.dumps(frame_record, ensure_ascii=False) + "\n"
                )
                if homography_file is not None:
                    homography_file.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "timestamp_seconds": round(frame_index / fps, 4),
                                "segmentation": (
                                    segmentation_result.to_dict()
                                    if segmentation_result is not None
                                    else None
                                ),
                                "geometry": geometry_result.to_dict(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                rejected = [
                    *detector.last_rejected_detections,
                    *tracker.last_rejections,
                ]
                total_rejected += len(rejected)
                if rejected_file is not None and rejected:
                    rejected_file.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "timestamp_seconds": round(frame_index / fps, 4),
                                "rejected": rejected,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if debug_file is not None:
                    debug_file.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "robots_in_memory": {
                                    str(track_id): {
                                        "confirmed": track.confirmed,
                                        "hits": track.hits,
                                        "missed": track.missed,
                                        "speed_px_s": round(track.speed, 3),
                                    }
                                    for track_id, track in tracker.robot_tracks.items()
                                },
                                "ball_in_memory": (
                                    {
                                        "confirmed": tracker.ball_track.confirmed,
                                        "hits": tracker.ball_track.hits,
                                        "missed": tracker.ball_track.missed,
                                        "speed_px_s": round(tracker.ball_track.speed, 3),
                                    }
                                    if tracker.ball_track is not None
                                    else None
                                ),
                                "camera_registration": registration_result.to_dict(),
                                "field_segmentation": (
                                    segmentation_result.to_dict()
                                    if segmentation_result is not None
                                    else None
                                ),
                                "field_geometry": geometry_result.to_dict(),
                                "field_candidates": tracker.last_field_candidates,
                                "team_clustering": team_classifier.get_debug_state(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                stage_seconds["render_and_output"] += perf_counter() - stage_started
                stage_seconds["total"] += perf_counter() - frame_started
                processed_frames += 1
                frame_index += 1
                progress.update(1)

                if progress_callback is not None:
                    progress_callback(processed_frames, limit)
               
                
                
    finally:
        progress.close()
        capture.release()
        writer.release()
        if geometry_writer is not None:
            geometry_writer.release()
        if rectified_writer is not None:
            rectified_writer.release()
        if debug_file is not None:
            debug_file.close()
        if rejected_file is not None:
            rejected_file.close()
        if homography_file is not None:
            homography_file.close()

    # Reporta los últimos frames cuando el total no es múltiplo de 30.
    if progress_callback is not None:
        final_total = limit if limit > 0 else processed_frames

        progress_callback(
            processed_frames,
            final_total,
        )

    if processed_frames > 0:
        average_ms = {
            key: 1000.0 * value / processed_frames
            for key, value in stage_seconds.items()
        }
        print(
            "\nPerfil por frame: "
            f"YOLO {average_ms['detector']:.1f} ms | "
            f"segmentador {average_ms['segmenter']:.1f} ms promedio "
            f"({segmentation_calls} llamadas) | "
            f"geometria {average_ms['geometry']:.1f} ms | "
            f"tracking {average_ms['tracking']:.1f} ms | "
            f"salida {average_ms['render_and_output']:.1f} ms | "
            f"total {average_ms['total']:.1f} ms"
        )

    online_team_summary = team_classifier.get_debug_state()
    (output_directory / "team_clustering_online.json").write_text(
        json.dumps(online_team_summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    identity_result = None
    if offline_identity_v5:
        print("\nReconstruyendo robots físicos y equipos con información del video completo...")
        offline_config = OfflineIdentityConfig.from_json(
            team_config_path,
            swap_team_labels=team_classifier.swap_team_labels,
        )
        offline_config.robot_interpolation_seconds = max(
            0.0,
            float(robot_interpolation_seconds),
        )
        identity_result = reconstruct_physical_identities(
            video_path=video_path,
            detections_path=detections_path,
            output_directory=output_directory,
            config=offline_config,
        )
        preview_path = Path(identity_result["preview_path"])
        team_summary_path = Path(identity_result["summary_path"])
    else:
        # Compatibilidad con el comportamiento online anterior.
        team_classifier.backfill_jsonl(detections_path)
        team_summary_path.write_text(
            json.dumps(online_team_summary, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "preview_path": str(preview_path),
        "detections_path": str(detections_path),
        "rejected_detections_path": str(rejected_path) if save_tracking_debug else None,
        "tracking_debug_path": str(tracking_debug_path) if save_tracking_debug else None,
        "team_summary_path": str(team_summary_path),
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "total_rejected_detections": total_rejected,
        "sam_mode": None,
        "fps": fps,
        "camera_stabilization": camera_stabilization,
        "team_mode": team_classifier.mode,
        "team_locked": (
            bool(identity_result.get("team_pairing_confirmed", False))
            if identity_result
            else team_classifier.locked
        ),
        "team_assignment": (
            identity_result.get("team_by_physical", {})
            if identity_result
            else team_classifier.get_debug_state().get("team_by_id", {})
        ),
        "identity_v5": identity_result,
        "identity_summary_path": (
            identity_result.get("summary_path") if identity_result else None
        ),
        "online_preview_path": (
            identity_result.get("online_preview_path") if identity_result else None
        ),
        "field_geometry_debug_path": (
            str(field_geometry_debug_path) if geometry_writer is not None else None
        ),
        "field_rectified_debug_path": (
            str(field_rectified_debug_path) if rectified_writer is not None else None
        ),
        "field_homography_path": (
            str(field_homography_path) if segmenter is not None else None
        ),
        "field_segmentation_enabled": segmenter is not None,
        "field_calibration_path": (
            str(Path(field_calibration_path).expanduser().resolve())
            if field_calibration_path is not None
            else None
        ),
        "performance": {
            "profile": performance.resolved_profile,
            "cuda_available": performance.cuda_available,
            "field_segmentation_image_size": performance.field_segmentation_image_size,
            "field_segmentation_stride": performance.field_segmentation_stride,
            "field_debug_stride": performance.field_debug_stride,
            "segmentation_calls": segmentation_calls,
            "debug_refreshes": debug_refreshes,
            "average_ms_per_frame": {
                key: round(1000.0 * value / max(1, processed_frames), 3)
                for key, value in stage_seconds.items()
            },
        },
        "resolution": {"width": width, "height": height},
    }
