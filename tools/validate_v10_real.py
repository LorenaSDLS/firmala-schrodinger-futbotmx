#!/usr/bin/env python3
"""Replay only the geometry stack on a recorded run.

The script reuses detections from a previous JSONL run so camera registration,
field segmentation and geometry can be changed and measured without running the
object detector again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.F_simulation.field_registration import FieldRegistration
from src.I_field_geometry.field_geometry import (
    FieldGeometryEstimator,
    draw_field_evidence_debug,
    draw_field_geometry_overlay,
    render_rectified_debug,
)
from src.I_field_geometry.field_segmenter import FieldSegmenter


def read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["frame_index"])] = row
    return rows


def writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    value = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), size)
    if not value.isOpened():
        raise RuntimeError(f"No se pudo crear {path}")
    return value


def transformed_probe(homography: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    if homography is None:
        return None
    points = np.float32([[[0.15 * width, 0.20 * height], [0.50 * width, 0.50 * height], [0.85 * width, 0.80 * height]]])
    try:
        projected = cv2.perspectiveTransform(points, np.asarray(homography, np.float64))[0]
    except cv2.error:
        return None
    return projected if np.isfinite(projected).all() else None


def main() -> None:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(10_010)
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("src/FIELD_SEGMENTATION/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--segmentation-stride", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.detections)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segmenter = FieldSegmenter(args.weights, image_size=args.image_size)
    registration = FieldRegistration(width, height, processing_max_width=720)
    geometry = FieldGeometryEstimator(width, height, calibration_path=args.calibration)

    geometry_writer = None if args.no_videos else writer(args.output / "field_geometry_v10_validation.mp4", fps, (680, 904))
    rectified_writer = None if args.no_videos else writer(args.output / "field_rectified_v10_validation.mp4", fps, (1120, 600))
    evidence_writer = None if args.no_videos else writer(args.output / "field_evidence_v10_validation.mp4", fps / max(1, args.segmentation_stride), (1280, 720))
    jsonl = (args.output / "field_homography_v10.jsonl").open("w", encoding="utf-8")

    previous_probe: np.ndarray | None = None
    local_jumps: list[float] = []
    metrics: dict[str, Any] = {
        "frames": 0,
        "segmentation_measurements": 0,
        "global_frames": 0,
        "global_measured_frames": 0,
        "trusted_frames": 0,
        "local_frames": 0,
        "surface_frames": 0,
        "global_suspended_frames": 0,
        "registration_invalid_frames": 0,
        "registration_projective_frames": 0,
        "candidate_states": {},
        "sources": {},
        "physical_boundary_histogram": {},
        "global_frame_indices": [],
        "global_measured_indices": [],
    }

    frame_index = 0
    last_segmentation = None
    while frame_index < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        row = rows.get(frame_index, {})
        detections = list(row.get("detections") or [])
        dynamic_boxes = [
            list(map(float, d.get("bbox_xyxy", [])))
            for d in detections
            if d.get("class_group") in {"robot", "ball"} and len(d.get("bbox_xyxy", [])) == 4
        ]
        goal_detections = [d for d in detections if d.get("class_group") == "goal"]

        segmentation = None
        if frame_index % max(1, args.segmentation_stride) == 0:
            segmentation = segmenter.segment_frame(frame)
            last_segmentation = segmentation
            metrics["segmentation_measurements"] += 1

        reg = registration.update(
            frame,
            semantic_mask=None if segmentation is None else segmentation.mask,
            exclusion_boxes=dynamic_boxes,
            frame_index=frame_index,
        )
        result = geometry.update(
            segmentation=segmentation,
            current_to_reference=reg.matrix,
            frame=frame,
            goal_detections=goal_detections,
            exclusion_boxes=dynamic_boxes,
            frame_index=frame_index,
            registration_quality=reg.quality,
            registration_updated=reg.updated,
        )

        state = str(result.geometry_state)
        metrics[f"{state}_frames"] = int(metrics.get(f"{state}_frames", 0)) + 1
        metrics["trusted_frames"] += int(result.trusted)
        metrics["registration_invalid_frames"] += int(not reg.valid)
        metrics["registration_projective_frames"] += int(reg.model_type == "homography")
        source = str(result.source)
        metrics["sources"][source] = int(metrics["sources"].get(source, 0)) + 1
        admission = str(result.pose_admission_state)
        metrics["candidate_states"][admission] = int(metrics["candidate_states"].get(admission, 0)) + 1
        boundary_count = str(int(result.physical_boundary_count))
        metrics["physical_boundary_histogram"][boundary_count] = int(metrics["physical_boundary_histogram"].get(boundary_count, 0)) + 1
        if state == "global":
            metrics["global_frame_indices"].append(frame_index)
            metrics["global_measured_frames"] += int(result.measured)
            if result.measured:
                metrics["global_measured_indices"].append(frame_index)

        probe = transformed_probe(result.local_homography_image_to_local, width, height)
        if probe is not None and previous_probe is not None:
            local_jumps.append(float(np.median(np.linalg.norm(probe - previous_probe, axis=1))))
        if probe is not None:
            previous_probe = probe

        debug = geometry.template_registrar.last_debug or {}
        payload = {
            "frame_index": frame_index,
            "camera_registration": reg.to_dict(),
            "field_segmentation": None if segmentation is None else segmentation.to_dict(),
            "field_geometry": result.to_dict(),
            "template_debug": {
                "source": debug.get("source"),
                "best_corners_work": (
                    None
                    if debug.get("best_corners") is None
                    else np.asarray(debug["best_corners"], dtype=float).round(4).tolist()
                ),
                "candidate_margin": debug.get("candidate_margin"),
                "candidate_count": debug.get("candidate_count"),
                "physical_boundary_scores": debug.get("physical_boundary_scores", {}),
            },
        }
        jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if not args.no_videos:
            geometry_frame = draw_field_geometry_overlay(frame, segmentation or last_segmentation, result)
            assert geometry_writer is not None
            geometry_writer.write(cv2.resize(geometry_frame, (680, 904), interpolation=cv2.INTER_AREA))
            rectified_frame = render_rectified_debug(
                frame, result, detections, segmentation=segmentation or last_segmentation,
                output_width=1000, output_height=600,
            )
            if rectified_frame.shape[1] != 1120 or rectified_frame.shape[0] != 600:
                rectified_frame = cv2.resize(rectified_frame, (1120, 600), interpolation=cv2.INTER_AREA)
            assert rectified_writer is not None
            rectified_writer.write(rectified_frame)

            if segmentation is not None:
                evidence = draw_field_evidence_debug(frame, result, geometry.template_registrar.last_debug)
                assert evidence_writer is not None
                evidence_writer.write(cv2.resize(evidence, (1280, 720), interpolation=cv2.INTER_AREA))
                if result.pose_candidate_streak > 0 or result.geometry_state == "global":
                    cv2.imwrite(str(args.output / f"candidate_{frame_index:04d}.jpg"), evidence)

        metrics["frames"] += 1
        frame_index += 1
        if frame_index % 60 == 0:
            jsonl.flush()
            print(
                f"frame={frame_index} state={state} source={source} "
                f"boundary={result.physical_boundary_count} streak={result.pose_candidate_streak}",
                flush=True,
            )

    cap.release()
    if geometry_writer is not None:
        geometry_writer.release()
    if rectified_writer is not None:
        rectified_writer.release()
    if evidence_writer is not None:
        evidence_writer.release()
    jsonl.close()

    if local_jumps:
        values = np.asarray(local_jumps, dtype=np.float64)
        metrics["local_motion_median"] = float(np.median(values))
        metrics["local_motion_p95"] = float(np.percentile(values, 95))
        metrics["local_motion_max"] = float(np.max(values))
        metrics["local_jumps_gt_50"] = int(np.count_nonzero(values > 50.0))
        metrics["local_jumps_gt_200"] = int(np.count_nonzero(values > 200.0))
    else:
        metrics.update(local_motion_median=None, local_motion_p95=None, local_motion_max=None, local_jumps_gt_50=0, local_jumps_gt_200=0)

    summary_path = args.output / "validation_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    # Ultralytics/PyTorch may leave CPU worker threads alive after all outputs
    # are closed. The validation harness is a one-shot process, so terminate
    # explicitly after flushing the regression artifacts.
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
