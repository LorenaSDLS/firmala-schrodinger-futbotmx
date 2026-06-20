from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.C_quick_view.team_classifier import TeamClassifier
from src.C_quick_view.offline_identity import (
    OfflineIdentityConfig,
    reconstruct_physical_identities,
)
from src.C_quick_view.temporal_tracker import FutbotTemporalTracker
from src.C_quick_view.yolo_detector import YOLODetector, draw_yolo_detections
from src.F_simulation.field_registration import FieldRegistration, RegistrationResult
from src.shared.paths import FIELD_SEGMENTATION_WEIGHTS_PATH
from src.shared.performance import resolve_performance_settings
from src.I_field_geometry.field_segmenter import FieldSegmenter
from src.I_field_geometry.hologram_calibration import HologramCalibration, is_hologram_calibration
from src.I_field_geometry.assisted_hologram_tracker import AssistedHologramTrajectory
from src.I_field_geometry.field_geometry import (
    FieldGeometryEstimator,
    draw_field_geometry_overlay,
    draw_field_evidence_debug,
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
    field_evidence_debug_path = output_directory / "field_evidence_debug.mp4"
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
    hologram_calibration = None
    hologram_trajectory = None
    geometry_calibration_path = field_calibration_path
    if is_hologram_calibration(field_calibration_path):
        hologram_calibration = HologramCalibration.load(field_calibration_path).scaled_to(width, height)
        field_canonical_width = hologram_calibration.field_width
        field_canonical_height = hologram_calibration.field_height
        geometry_calibration_path = None
        print(
            f"V11 holograma asistido: {len(hologram_calibration.keyframes)} ancla(s), "
            f"cancha {field_canonical_width:.1f} x {field_canonical_height:.1f} cm"
        )
        hologram_trajectory = AssistedHologramTrajectory.from_video(
            video_path=video_path,
            calibration=hologram_calibration,
            cache_path=output_directory / "field_hologram_trajectory.npz",
            processing_max_width=420 if performance.resolved_profile != "cpu" else 320,
        )

    field_geometry = FieldGeometryEstimator(
        frame_width=width,
        frame_height=height,
        field_width=field_canonical_width,
        field_height=field_canonical_height,
        calibration_path=geometry_calibration_path,
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
    evidence_writer = None
    rectified_size = (1200, 600)
    if field_debug and (segmenter is not None or hologram_trajectory is not None):
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
        evidence_writer = cv2.VideoWriter(
            str(field_evidence_debug_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    limit = total_frames
    if max_frames is not None:
        limit = min(total_frames, int(max_frames)) if total_frames > 0 else int(max_frames)

    frame_index = 0
    processed_frames = 0
    total_detections = 0
    total_rejected = 0
    last_segmentation_result = None
    frames_since_segmentation = 10_000
    last_geometry_debug_frame = None
    last_rectified_debug_frame = None
    last_evidence_debug_frame = None
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
        if segmenter is not None or hologram_trajectory is not None
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
                segmentation_result = None
                if segmenter is not None:
                    stride = performance.field_segmentation_stride
                    warmup_reacquisition = (
                        field_geometry.needs_reacquisition
                        and (processed_frames < 12 or frames_since_segmentation >= max(1, stride // 2))
                    )
                    should_segment = (
                        processed_frames == 0
                        or frames_since_segmentation >= stride
                        or warmup_reacquisition
                    )
                    if should_segment:
                        stage_started = perf_counter()
                        segmentation_result = segmenter.segment_frame(frame)
                        stage_seconds["segmenter"] += perf_counter() - stage_started
                        segmentation_calls += 1
                        frames_since_segmentation = 0
                        if segmentation_result is not None:
                            last_segmentation_result = segmentation_result

                stage_started = perf_counter()
                if hologram_trajectory is not None:
                    pose = hologram_trajectory.pose(frame_index)
                    registration.current_to_reference = hologram_trajectory.matrices[
                        min(frame_index, len(hologram_trajectory.matrices) - 1)
                    ].copy()
                    registration_result = RegistrationResult(
                        matrix=registration.current_to_reference.copy(),
                        valid=True,
                        updated=pose.registration_updated,
                        quality=pose.registration_quality,
                        tracked_points=0,
                        inlier_ratio=pose.registration_quality,
                        model_type="hologram_cache",
                    )
                    registration.last_result = registration_result
                    support_segmentation = segmentation_result or last_segmentation_result
                    support_mask = (
                        support_segmentation.mask
                        if support_segmentation is not None
                        else None
                    )
                    geometry_result = hologram_trajectory.geometry_result(
                        frame_index,
                        surface_mask=support_mask,
                    )
                    field_geometry.last_result = geometry_result
                    if support_mask is not None:
                        field_geometry.last_surface_mask_image = (support_mask > 0).astype("uint8") * 255
                    elif geometry_result.corners_image is not None:
                        projected_surface = np.zeros((height, width), dtype="uint8")
                        cv2.fillConvexPoly(
                            projected_surface,
                            np.rint(geometry_result.corners_image).astype("int32"),
                            255,
                        )
                        field_geometry.last_surface_mask_image = projected_surface
                    else:
                        field_geometry.last_surface_mask_image = None
                else:
                    registration_result = registration.update(
                        frame,
                        semantic_mask=(
                            segmentation_result.mask
                            if segmentation_result is not None
                            else None
                        ),
                        exclusion_boxes=dynamic_boxes,
                        frame_index=frame_index,
                    )
                    geometry_result = field_geometry.update(
                        segmentation=segmentation_result,
                        current_to_reference=registration_result.matrix,
                        frame=frame,
                        goal_detections=[
                            detection
                            for detection in raw_detections
                            if str(detection.get("class_group", "")).lower() == "goal"
                        ],
                        exclusion_boxes=dynamic_boxes,
                        frame_index=frame_index,
                        registration_quality=registration_result.quality,
                        registration_updated=registration_result.updated,
                    )

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
                if evidence_writer is not None:
                    if refresh_field_debug or last_evidence_debug_frame is None:
                        last_evidence_debug_frame = draw_field_evidence_debug(
                            frame,
                            geometry_result,
                            field_geometry.template_registrar.last_debug,
                        )
                    evidence_writer.write(last_evidence_debug_frame)
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
    finally:
        progress.close()
        capture.release()
        writer.release()
        if geometry_writer is not None:
            geometry_writer.release()
        if rectified_writer is not None:
            rectified_writer.release()
        if evidence_writer is not None:
            evidence_writer.release()
        if debug_file is not None:
            debug_file.close()
        if rejected_file is not None:
            rejected_file.close()
        if homography_file is not None:
            homography_file.close()

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
        "field_evidence_debug_path": (
            str(field_evidence_debug_path) if evidence_writer is not None else None
        ),
        "field_homography_path": (
            str(field_homography_path)
            if segmenter is not None or hologram_trajectory is not None else None
        ),
        "field_segmentation_enabled": segmenter is not None,
        "field_hologram_enabled": bool(hologram_trajectory is not None),
        "field_hologram_trajectory_path": (
            str(output_directory / "field_hologram_trajectory.npz")
            if hologram_trajectory is not None else None
        ),
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
