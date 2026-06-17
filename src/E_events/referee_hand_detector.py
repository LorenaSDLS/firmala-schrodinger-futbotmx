import json
from pathlib import Path
from typing import Any

import cv2

#from src.C_quick_view.sam_segmenter import SAMSegmenter


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_video_frames(video_path: str | Path, frame_indices: list[int]) -> dict[int, Any]:
    requested_frames = sorted(set(frame_indices))
    frames = {}

    if not requested_frames:
        return frames

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    requested_set = set(requested_frames)
    max_requested = max(requested_frames)
    frame_index = 0

    while frame_index <= max_requested:
        success, frame = capture.read()

        if not success:
            break

        if frame_index in requested_set:
            frames[frame_index] = frame.copy()

        frame_index += 1

    capture.release()
    return frames


def draw_hand_detections(frame, detections):
    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
        confidence = detection["confidence"]

        label = f"referee hand {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    return annotated


def get_candidate_events(events: list[dict]) -> list[dict]:
    candidate_types = {
        "robot_inactive_candidate",
        "ball_missing_candidate",
        "ball_out_of_field",
    }

    return [
        event
        for event in events
        if event.get("event_type") in candidate_types
        and event.get("data", {}).get("last_known_bbox")
    ]


def analyze_referee_hand_candidates(
    video_path: str | Path,
    output_directory: str | Path,
    sam_mode: str = "LoHa",
    sam_confidence: float = 0.25,
    frame_window: int = 5,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    events_path = output_directory / "match_events.json"

    if not events_path.exists():
        raise FileNotFoundError(
            f"No se encontro match_events.json en: {output_directory}"
        )

    events = load_json(events_path)
    candidate_events = get_candidate_events(events)

    if max_candidates is not None:
        candidate_events = candidate_events[:max_candidates]

    frames_needed = []
    for event in candidate_events:
        base_frame = int(event["frame_index"])
            
        frames_needed.extend([
            max(0, base_frame - frame_window),
            base_frame,
            base_frame + frame_window,
        ])

    frames_by_index = read_video_frames(video_path, frames_needed)
    print(
        "Frames candidatos cargados: "
        f"{len(frames_by_index)}/{len(set(frames_needed))}"
    )

    if not candidate_events:
        candidates_path = output_directory / "referee_hand_candidates.json"
        updated_events_path = output_directory / "match_events_with_referee.json"

        save_json([], candidates_path)
        save_json(events, updated_events_path)

        return {
            "candidates_path": str(candidates_path),
            "updated_events_path": str(updated_events_path),
            "debug_directory": str(output_directory / "referee_hand_debug"),
            "candidates_detected": 0,
            "new_events": 0,
        }

    debug_directory = output_directory / "referee_hand_debug"
    debug_directory.mkdir(parents=True, exist_ok=True)

    print(f"Eventos candidatos a revisar: {len(candidate_events)}")
    print("Cargando SAM3 + LoHa para buscar mano del arbitro...")

    print("Importando SAM3/PEFT... puede tardar un poco.")
    from src.C_quick_view.sam_segmenter import SAMSegmenter

    sam = SAMSegmenter(
        mode=sam_mode,
        confidence_threshold=sam_confidence,
        api_path="Analizador de video/API_sam3.json",
    )


    referee_hand_candidates = []
    new_events = []

    for event in candidate_events:
        base_frame = int(event["frame_index"])
        source_event_type = event.get("event_type")
        source_data = event.get("data", {})

        object_id = (
            source_data.get("robot_id")
            or source_data.get("object_id")
            or "unknown"
        )

        frames_to_check = sorted({
            max(0, base_frame - frame_window),
            base_frame,
            base_frame + frame_window,
        })

        for frame_index in frames_to_check:
            frame = frames_by_index.get(frame_index)

            if frame is None:
                continue

            target_bbox = event.get("data", {}).get("last_known_bbox")
            if not target_bbox:
                continue
            height, width = frame.shape[:2]
            crop_bbox = expand_bbox(
                bbox=target_bbox,
                margin=80,
                frame_width=width,
                frame_height=height,
                )

            crop = crop_frame(frame, crop_bbox)

            crop_detections = sam.detect(crop, "human hand")

            frame_detections = [
                translate_detection_to_frame(detection, crop_bbox)
                for detection in crop_detections
            ]

            detections = filter_relevant_hand_detections(
                frame_detections,
                target_bbox=target_bbox,
                min_confidence=0.35,
                max_distance=140,
            )

            if not detections:
                continue

            candidate = {
                "source_event": event,
                "frame_index": frame_index,
                "timestamp_seconds": event["timestamp_seconds"],
                "object_id": object_id,
                "detections": detections,
            }

            referee_hand_candidates.append(candidate)

            debug_path = (
                debug_directory
                / f"referee_hand_frame_{frame_index:06d}.jpg"
            )

            annotated = draw_hand_detections(frame, detections)
            cv2.imwrite(str(debug_path), annotated)

            #nuevo

            source_event_type = event.get("event_type")
            source_data = event.get("data", {})

            object_id = (
                source_data.get("robot_id")
                or source_data.get("object_id")
                or "unknown"
            )

            if source_event_type == "robot_inactive_candidate":
                refined_event_type = "robot_grabbed_by_referee"
                description = f"El arbitro probablemente tomo a {object_id}."
            elif source_event_type in {"ball_missing_candidate", "ball_out_of_field"}:
                refined_event_type = "ball_moved_by_referee"
                description = "El arbitro probablemente movio la pelota."
            else:
                refined_event_type = "referee_intervention_candidate"
                description = f"Posible intervencion del arbitro cerca de {object_id}."

            new_events.append({
                "frame_index": frame_index,
                "timestamp_seconds": event["timestamp_seconds"],
                "event_type": refined_event_type,
                "description": description,
                "data": {
                    "source_event_type": source_event_type,
                    "object_id": object_id,
                    "target_bbox": target_bbox,
                    "hand_detections": detections,
                    "debug_image": str(debug_path),
                },

            })

            print(
                f"Mano candidata detectada en frame {frame_index} "
                f"para {object_id}."
            )

    updated_events = events + new_events
    updated_events.sort(
        key=lambda event: (
            event["timestamp_seconds"],
            event["frame_index"],
        )
    )

    candidates_path = output_directory / "referee_hand_candidates.json"
    updated_events_path = output_directory / "match_events_with_referee.json"

    save_json(referee_hand_candidates, candidates_path)
    save_json(updated_events, updated_events_path)

    return {
        "candidates_path": str(candidates_path),
        "updated_events_path": str(updated_events_path),
        "debug_directory": str(debug_directory),
        "candidates_detected": len(referee_hand_candidates),
        "new_events": len(new_events),
    }

def expand_bbox(
    bbox: list[float],
    margin: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    x1, y1, x2, y2 = map(int, bbox)

    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(frame_width, x2 + margin),
        min(frame_height, y2 + margin),
    ]


def crop_frame(frame, crop_bbox: list[int]):
    x1, y1, x2, y2 = crop_bbox
    return frame[y1:y2, x1:x2]


def translate_detection_to_frame(
    detection: dict,
    crop_bbox: list[int],
) -> dict:
    crop_x1, crop_y1, _, _ = crop_bbox
    x1, y1, x2, y2 = detection["bbox_xyxy"]

    translated = detection.copy()
    translated["bbox_xyxy"] = [
        x1 + crop_x1,
        y1 + crop_y1,
        x2 + crop_x1,
        y2 + crop_y1,
    ]

    return translated


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_distance(bbox_a: list[float], bbox_b: list[float]) -> float:
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def bbox_intersects(bbox_a: list[float], bbox_b: list[float]) -> bool:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    return not (
        ax2 < bx1
        or ax1 > bx2
        or ay2 < by1
        or ay1 > by2
    )


def filter_relevant_hand_detections(
    hand_detections: list[dict],
    target_bbox: list[float],
    min_confidence: float = 0.65,
    max_distance: float = 120.0,
) -> list[dict]:
    relevant = []

    for detection in hand_detections:
        hand_bbox = detection["bbox_xyxy"]
        confidence = detection.get("confidence", 0.0)

        if confidence < min_confidence:
            continue

        intersects = bbox_intersects(hand_bbox, target_bbox)
        distance = bbox_distance(hand_bbox, target_bbox)

        if intersects or distance <= max_distance:
            detection = detection.copy()
            detection["relation_to_target"] = {
                "intersects": intersects,
                "distance_px": round(distance, 2),
            }
            relevant.append(detection)

    return relevant

def read_detection_records(detections_path: str | Path):
    with Path(detections_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def get_target_detections(frame_record: dict) -> list[dict]:
    targets = []

    for detection in frame_record.get("detections", []):
        class_name = detection.get("class_name", "").lower()
        bbox = detection.get("bbox_xyxy") or detection.get("box")

        if not bbox:
            continue

        if class_name == "robot":
            targets.append({
                "target_type": "robot",
                "target_id": f"robot_{detection.get('tracking_id', 'sin_id')}",
                "bbox_xyxy": bbox,
                "confidence": detection.get("confidence", 0.0),
            })

        elif class_name in {"orange ball", "ball", "pelota"}:
            targets.append({
                "target_type": "ball",
                "target_id": "ball",
                "bbox_xyxy": bbox,
                "confidence": detection.get("confidence", 0.0),
            })

    return targets


def draw_contact_detections(frame, contacts):
    annotated = frame.copy()

    for contact in contacts:
        target_bbox = contact["target_bbox"]
        tx1, ty1, tx2, ty2 = map(int, target_bbox)

        cv2.rectangle(
            annotated,
            (tx1, ty1),
            (tx2, ty2),
            (255, 0, 255),
            3,
        )

        cv2.putText(
            annotated,
            contact["target_id"],
            (tx1, max(25, ty1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

        for hand in contact["hand_detections"]:
            hx1, hy1, hx2, hy2 = map(int, hand["bbox_xyxy"])
            label = f"hand contact {hand['confidence']:.2f}"

            cv2.rectangle(
                annotated,
                (hx1, hy1),
                (hx2, hy2),
                (0, 255, 255),
                3,
            )

            cv2.putText(
                annotated,
                label,
                (hx1, max(25, hy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    return annotated


def analyze_referee_hand_contacts(
    video_path: str | Path,
    output_directory: str | Path,
    sam_mode: str = "LoHa",
    sam_confidence: float = 0.18,
    frame_stride: int = 10,
    crop_margin: int = 220,
    min_hand_confidence: float = 0.55,
    max_distance: float = 80.0,
    lookahead_frames: int = 20,
    missing_frames_threshold: int = 8,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    detections_path = output_directory / "quick_detections.jsonl"

    if not detections_path.exists():
        raise FileNotFoundError(
            f"No se encontro quick_detections.jsonl en: {output_directory}"
        )

    debug_directory = output_directory / "referee_hand_debug"
    debug_directory.mkdir(parents=True, exist_ok=True)

    preview_path = output_directory / "referee_hand_preview.mp4"
    contacts_path = output_directory / "referee_hand_contacts.json"
    events_path = output_directory / "referee_hand_contact_events.json"

    print("Cargando SAM3 + LoHa para buscar manos cerca de robots/pelota...")

    print("Importando SAM3/PEFT... puede tardar un poco.")
    from src.C_quick_view.sam_segmenter import SAMSegmenter

    sam = SAMSegmenter(
        mode=sam_mode,
        confidence_threshold=sam_confidence,
        api_path="Analizador de video/API_sam3.json",
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    all_contacts = []
    new_events = []

    records_by_frame = {
        int(record["frame_index"]): record
        for record in read_detection_records(detections_path)
    }

    max_frame = max(records_by_frame.keys(), default=-1)

    for frame_index in range(max_frame + 1):
        success, frame = capture.read()

        if not success:
            break

        frame_contacts = []
        frame_record = records_by_frame.get(frame_index)

        if frame_record and frame_index % frame_stride == 0:
            targets = get_target_detections(frame_record)

            for target in targets:
                target_bbox = target["bbox_xyxy"]

                crop_bbox = expand_bbox(
                    bbox=target_bbox,
                    margin=crop_margin,
                    frame_width=width,
                    frame_height=height,
                )

                crop = crop_frame(frame, crop_bbox)

                if crop.size == 0:
                    continue

                crop_detections = sam.detect(crop, "human hand")

                frame_detections = [
                    translate_detection_to_frame(detection, crop_bbox)
                    for detection in crop_detections
                ]

                relevant_hands = filter_relevant_hand_detections(
                    frame_detections,
                    target_bbox=target_bbox,
                    min_confidence=min_hand_confidence,
                    max_distance=max_distance,
                )

                if not relevant_hands:
                    continue

                contact = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "target_type": target["target_type"],
                    "target_id": target["target_id"],
                    "target_bbox": target_bbox,
                    "hand_detections": relevant_hands,
                }

                frame_contacts.append(contact)
                all_contacts.append(contact)

                event_type = "referee_hand_contact_candidate"
                description = f"Mano cerca o tocando {target['target_id']}."

                if target["target_type"] == "robot":
                    was_grabbed = robot_missing_after_contact(
                        robot_id=target["target_id"],
                        contact_frame=frame_index,
                        records_by_frame=records_by_frame,
                        lookahead_frames=lookahead_frames,
                        missing_frames_threshold=missing_frames_threshold,
                    )

                    if was_grabbed:
                        event_type = "robot_grabbed_by_referee"
                        description = f"El arbitro probablemente tomo a {target['target_id']}."

                new_events.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "event_type": event_type,
                    "description": description,
                    "data": contact,
                })

                print(
                    f"Contacto candidato: frame {frame_index}, "
                    f"{target['target_id']}, manos: {len(relevant_hands)}"
                )

        annotated = draw_contact_detections(frame, frame_contacts)
        writer.write(annotated)

        if frame_contacts:
            debug_path = debug_directory / f"hand_contact_{frame_index:06d}.jpg"
            cv2.imwrite(str(debug_path), annotated)

    capture.release()
    writer.release()

    save_json(all_contacts, contacts_path)
    save_json(new_events, events_path)

    return {
        "contacts_path": str(contacts_path),
        "events_path": str(events_path),
        "preview_path": str(preview_path),
        "debug_directory": str(debug_directory),
        "contacts_detected": len(all_contacts),
        "events_detected": len(new_events),
    }

def get_robot_ids_in_frame(frame_record: dict) -> set[str]:
    robot_ids = set()

    for detection in frame_record.get("detections", []):
        class_name = detection.get("class_name", "").lower()

        if class_name != "robot":
            continue

        tracking_id = detection.get("tracking_id")

        if tracking_id is not None:
            robot_ids.add(f"robot_{tracking_id}")

    return robot_ids


def robot_missing_after_contact(
    robot_id: str,
    contact_frame: int,
    records_by_frame: dict[int, dict],
    lookahead_frames: int = 20,
    missing_frames_threshold: int = 8,
) -> bool:
    missing_frames = 0
    checked_frames = 0

    for frame_index in range(
        contact_frame + 1,
        contact_frame + lookahead_frames + 1,
    ):
        frame_record = records_by_frame.get(frame_index)

        if frame_record is None:
            continue

        checked_frames += 1
        robot_ids = get_robot_ids_in_frame(frame_record)

        if robot_id not in robot_ids:
            missing_frames += 1

    return checked_frames > 0 and missing_frames >= missing_frames_threshold