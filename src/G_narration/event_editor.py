from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


PRIORITY = {
    "red_card_robot_removed": 145,
    "goal": 130,
    "ball_out_of_field": 100,
    "robot_grabbed_by_referee": 95,
    "ball_moved_by_referee": 92,
    "referee_intervention_candidate": 88,
    "robot_collision_candidate": 80,
    "robot_inactive_candidate": 68,
    "robot_entered_penalty_area": 62,
    "robot_reactivated": 55,
    "ball_recovered": 54,
    "ball_missing_candidate": 48,
    "possession_change": 32,
}

MAX_PER_TYPE = {
    "red_card_robot_removed": 4,
    "goal": 8,
    "ball_out_of_field": 3,
    "robot_grabbed_by_referee": 3,
    "ball_moved_by_referee": 3,
    "referee_intervention_candidate": 2,
    "robot_collision_candidate": 2,
    "robot_inactive_candidate": 2,
    "robot_entered_penalty_area": 4,
    "robot_reactivated": 2,
    "ball_recovered": 2,
    "ball_missing_candidate": 1,
    "possession_change": 2,
}

@dataclass
class EditorialEvent:
    frame_index: int
    timestamp_seconds: float
    event_type: str
    description: str
    data: dict[str, Any]
    priority: float
    confidence: float
    importance: str
    narration_text: str
    estimated_duration_seconds: float


def _confidence(event: dict[str, Any]) -> float:
    raw = event.get("confidence")
    if raw is not None:
        return max(0.0, min(1.0, float(raw)))
    data = event.get("data") or {}
    for key in ("confidence", "event_confidence", "tracking_confidence"):
        if key in data:
            return max(0.0, min(1.0, float(data[key])))
    defaults = {
        "red_card_robot_removed": 0.92,
        "goal": 0.95,
        "ball_out_of_field": 0.86,
        "robot_grabbed_by_referee": 0.82,
        "ball_moved_by_referee": 0.80,
        "referee_intervention_candidate": 0.65,
        "robot_collision_candidate": 0.65,
        "robot_inactive_candidate": 0.60,
        "robot_entered_penalty_area": 0.78,
        "robot_reactivated": 0.82,
        "ball_recovered": 0.90,
        "ball_missing_candidate": 0.70,
        "possession_change": 0.68,
    }
    return defaults.get(str(event.get("event_type")), 0.55)


def _names(data: dict[str, Any]) -> tuple[str, str]:
    return (
        str(data.get("robot_a_name") or data.get("robot_a") or "un robot"),
        str(data.get("robot_b_name") or data.get("robot_b") or "otro robot"),
    )


def narration_text(event: dict[str, Any], confidence: float) -> str:
    event_type = str(event.get("event_type", ""))
    data = event.get("data") or {}
    uncertain = confidence < 0.72
    prefix = "Posible " if uncertain else ""

    if event_type == "red_card_robot_removed":
        name = data.get("robot_name") or data.get("object_id") or data.get("robot_id") or "un robot"
        return f"Tarjeta roja. El árbitro retira a {name}."
    if event_type == "goal":
        side = str(data.get("goal_side_image") or "detectada")
        return f"¡Gol en la portería {side}!" if not uncertain else f"Posible gol en la portería {side}."
    if event_type == "ball_out_of_field":
        return "El balón sale de la cancha." if not uncertain else "Parece que el balón salió de la cancha."
    if event_type == "robot_collision_candidate":
        a, b = _names(data)
        return f"{prefix}colisión entre {a} y {b}."
    if event_type == "robot_inactive_candidate":
        name = data.get("robot_name") or data.get("robot_id") or "un robot"
        return f"{name} permanece inactivo o fuera de vista."
    if event_type == "ball_missing_candidate":
        return "El balón se pierde momentáneamente de vista."
    if event_type == "ball_recovered":
        return "El balón vuelve a ser visible."
    if event_type == "robot_reactivated":
        name = data.get("robot_name") or data.get("robot_id") or "El robot"
        return f"{name} vuelve a la cancha."
    if event_type == "robot_entered_penalty_area":
        name = data.get("robot_name") or data.get("robot_id") or "Un robot"
        side = data.get("penalty_side") or "detectada"
        return f"{name} entra al área de la portería {side}."
    if event_type == "robot_grabbed_by_referee":
        name = data.get("robot_name") or data.get("robot_id") or "un robot"
        return f"El árbitro retira a {name}."
    if event_type == "ball_moved_by_referee":
        return "El árbitro recoloca el balón."
    if event_type == "referee_intervention_candidate":
        return "Se detecta una posible intervención del árbitro."
    if event_type == "possession_change":
        name = data.get("robot_name") or data.get("robot_id") or "otro robot"
        return f"{name} toma la posesión del balón."
    return str(event.get("description") or "Se registra un evento en la cancha.")


def estimate_speech_seconds(text: str, words_per_second: float = 2.35) -> float:
    return max(1.2, len(text.split()) / max(1.0, words_per_second) + 0.35)


def _dedupe(events: list[dict[str, Any]], cooldown_seconds: float = 2.5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    last_by_key: dict[str, float] = {}
    for event in sorted(events, key=lambda item: float(item.get("timestamp_seconds", 0.0))):
        event_type = str(event.get("event_type", ""))
        data = event.get("data") or {}
        if event_type == "robot_collision_candidate":
            pair = sorted([str(data.get("robot_a", "?")), str(data.get("robot_b", "?"))])
            key = f"{event_type}:{':'.join(pair)}"
        elif event_type in {"robot_inactive_candidate", "robot_reactivated", "robot_entered_penalty_area", "possession_change", "red_card_robot_removed"}:
            key = f"{event_type}:{data.get('robot_id', '?')}"
        else:
            key = event_type
        time = float(event.get("timestamp_seconds", 0.0))
        if key in last_by_key and time - last_by_key[key] < cooldown_seconds:
            continue
        last_by_key[key] = time
        result.append(event)
    return result



def _suppress_transient_pairs(events: list[dict[str, Any]], window_seconds: float = 4.0) -> list[dict[str, Any]]:
    """Remove short-lived missing/recovered chatter from the narration queue.

    The events remain in match_events.json and the report. They are only omitted
    from spoken commentary when recovery happens almost immediately, because
    narrating both sides of a brief occlusion creates noise and crowds important
    events such as goals and cards.
    """
    ordered = sorted(events, key=lambda item: float(item.get("timestamp_seconds", 0.0)))
    suppressed: set[int] = set()

    # Ball disappeared and came back quickly.
    for index, event in enumerate(ordered):
        if str(event.get("event_type", "")) != "ball_missing_candidate":
            continue
        start = float(event.get("timestamp_seconds", 0.0))
        for later_index in range(index + 1, len(ordered)):
            later = ordered[later_index]
            delta = float(later.get("timestamp_seconds", 0.0)) - start
            if delta > window_seconds:
                break
            if str(later.get("event_type", "")) == "ball_recovered":
                suppressed.update({index, later_index})
                break

    # Same idea for a robot that is declared inactive and immediately recovered.
    for index, event in enumerate(ordered):
        if index in suppressed or str(event.get("event_type", "")) != "robot_inactive_candidate":
            continue
        data = event.get("data") or {}
        robot_id = str(data.get("robot_id") or data.get("object_id") or "")
        start = float(event.get("timestamp_seconds", 0.0))
        for later_index in range(index + 1, len(ordered)):
            later = ordered[later_index]
            delta = float(later.get("timestamp_seconds", 0.0)) - start
            if delta > window_seconds:
                break
            if str(later.get("event_type", "")) != "robot_reactivated":
                continue
            later_data = later.get("data") or {}
            later_robot = str(later_data.get("robot_id") or later_data.get("object_id") or "")
            if not robot_id or robot_id == later_robot:
                suppressed.update({index, later_index})
                break

    return [event for index, event in enumerate(ordered) if index not in suppressed]

def select_editorial_events(
    events: list[dict[str, Any]],
    video_duration_seconds: float,
    max_events: int = 12,
    maximum_coverage_ratio: float = 0.45,
    minimum_silence_seconds: float = 2.3,
    minimum_confidence: float = 0.50,
) -> list[EditorialEvent]:
    candidates = []
    for event in _dedupe(_suppress_transient_pairs(events)):
        event_type = str(event.get("event_type", ""))
        if event_type not in PRIORITY:
            continue
        confidence = _confidence(event)
        if confidence < minimum_confidence:
            continue
        text = narration_text(event, confidence)
        duration = estimate_speech_seconds(text)
        score = PRIORITY[event_type] + 25.0 * confidence
        candidates.append((score, event, confidence, text, duration))

    # Always favor important events, but preserve chronological spacing.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, dict[str, Any], float, str, float]] = []
    counts: dict[str, int] = {}
    budget = max(0.0, video_duration_seconds * maximum_coverage_ratio)
    used = 0.0
    for candidate in candidates:
        _, event, _, _, duration = candidate
        event_type = str(event.get("event_type", ""))
        if counts.get(event_type, 0) >= MAX_PER_TYPE.get(event_type, 2):
            continue
        t = float(event.get("timestamp_seconds", 0.0))
        important = PRIORITY[event_type] >= 75
        too_close = any(
            abs(t - float(old[1].get("timestamp_seconds", 0.0))) < minimum_silence_seconds
            for old in selected
        )
        # Los eventos principales no se descartan por cercanía; la línea de audio
        # los acomoda uno después del otro. Los secundarios sí respetan el silencio.
        if too_close and not important:
            continue
        if len(selected) >= max_events:
            continue
        if used + duration > budget and not important:
            continue
        selected.append(candidate)
        counts[event_type] = counts.get(event_type, 0) + 1
        used += duration

    selected.sort(key=lambda item: float(item[1].get("timestamp_seconds", 0.0)))
    output = []
    for score, event, confidence, text, duration in selected:
        output.append(EditorialEvent(
            frame_index=int(event.get("frame_index", 0)),
            timestamp_seconds=float(event.get("timestamp_seconds", 0.0)),
            event_type=str(event.get("event_type", "unknown")),
            description=str(event.get("description", "")),
            data=dict(event.get("data") or {}),
            priority=float(score),
            confidence=float(confidence),
            importance="principal" if PRIORITY.get(str(event.get("event_type")), 0) >= 75 else "secundario",
            narration_text=text,
            estimated_duration_seconds=float(duration),
        ))
    return output


def load_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list):
        raise ValueError("El archivo de eventos debe contener una lista JSON.")
    return value


def save_editorial_manifest(events: list[EditorialEvent], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump([asdict(event) for event in events], file, ensure_ascii=False, indent=2)
    return path
