from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import json
import math


FIELD_LENGTH_CM = 243.0
FIELD_WIDTH_CM = 182.0
MAX_ROBOT_SPEED_MPS = 5.0
MOVING_SPEED_MPS = 0.04


def _load(path: str | Path | None, default: Any) -> Any:
    if path is None:
        return default
    candidate = Path(path)
    if not candidate.exists():
        return default
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return default


def _time(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(round(float(seconds)))), 60)
    return f"{minutes:02d}:{secs:02d}"


def _seconds(seconds: float) -> str:
    value = max(0.0, float(seconds))
    return f"{value:.1f} s"


def normalize_team(value: Any) -> str:
    team = str(value or "").lower().strip()
    if team in {"aliado", "ally", "magenta", "equipo_magenta", "team_magenta"}:
        return "aliado"
    if team in {"rival", "enemigo", "enemy", "azul", "blue", "equipo_azul", "team_blue"}:
        return "rival"
    return "desconocido"


def _normalize_robot_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("robot_"):
        return text
    if text.isdigit():
        return f"robot_{text}"
    return text


def _duration(tracks: dict[str, Any]) -> float:
    timestamps: list[float] = []
    for points in (tracks.get("robots") or {}).values():
        timestamps.extend(float(point.get("timestamp_seconds", 0.0)) for point in points)
    timestamps.extend(float(point.get("timestamp_seconds", 0.0)) for point in tracks.get("ball") or [])
    return max(timestamps, default=0.0)


def _load_external_team_assignments(output_directory: str | Path | None) -> dict[str, str]:
    if output_directory is None:
        return {}
    root = Path(output_directory)
    assignments: dict[str, str] = {}

    # Online/manual clustering is usually the most explicit team decision.
    for filename in ("team_clustering.json", "team_clustering_online.json"):
        payload = _load(root / filename, {})
        for raw_id, raw_team in (payload.get("team_by_id") or {}).items():
            robot_id = _normalize_robot_id(raw_id)
            team = normalize_team(raw_team)
            if robot_id and team != "desconocido":
                assignments[robot_id] = team

    # Offline physical identities can fill gaps when they contain confirmed teams.
    identity = _load(root / "identity_v5.json", {})
    for raw_id, raw_team in (identity.get("team_by_physical") or {}).items():
        robot_id = _normalize_robot_id(raw_id)
        team = normalize_team(raw_team)
        if robot_id and team != "desconocido" and robot_id not in assignments:
            assignments[robot_id] = team

    return assignments


def _team_map(tracks: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, str]:
    external = _load_external_team_assignments(output_directory)
    result: dict[str, str] = {}
    for raw_robot_id, points in (tracks.get("robots") or {}).items():
        robot_id = _normalize_robot_id(raw_robot_id) or str(raw_robot_id)
        votes = Counter(
            normalize_team(point.get("team"))
            for point in points
            if point.get("visible", True) and normalize_team(point.get("team")) != "desconocido"
        )
        direct = votes.most_common(1)[0][0] if votes else "desconocido"
        result[robot_id] = direct if direct != "desconocido" else external.get(robot_id, "desconocido")
    # Keep external identities even when a track is absent from the current export.
    for robot_id, team in external.items():
        result.setdefault(robot_id, team)
    return result


def _event_team(event: dict[str, Any], team_map: dict[str, str]) -> str:
    data = event.get("data") or {}
    direct = normalize_team(data.get("scoring_team") or data.get("team") or event.get("team"))
    if direct != "desconocido":
        return direct
    for key in ("robot_id", "robot_a", "owner_robot_id"):
        robot_id = _normalize_robot_id(data.get(key))
        if robot_id and robot_id in team_map:
            return team_map[robot_id]
    return "desconocido"


def _event_robot_ids(event: dict[str, Any]) -> list[str]:
    data = event.get("data") or {}
    ids: list[str] = []
    for key in ("robot_id", "robot_a", "robot_b", "owner_robot_id", "previous_owner"):
        robot_id = _normalize_robot_id(data.get(key))
        if robot_id and robot_id not in ids:
            ids.append(robot_id)
    return ids


def _coord(point: dict[str, Any]) -> tuple[float, float] | None:
    if not point.get("field_transform_valid", False):
        return None
    x_value, y_value = point.get("field_x"), point.get("field_y")
    if x_value is None or y_value is None:
        x_norm, y_norm = point.get("field_x_norm"), point.get("field_y_norm")
        if x_norm is None or y_norm is None:
            return None
        x_value = float(x_norm) * FIELD_LENGTH_CM
        y_value = float(y_norm) * FIELD_WIDTH_CM
    x_value, y_value = float(x_value), float(y_value)
    if -5.0 <= x_value <= FIELD_LENGTH_CM + 5.0 and -5.0 <= y_value <= FIELD_WIDTH_CM + 5.0:
        return x_value, y_value
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _motion_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda point: float(point.get("timestamp_seconds", 0.0)))
    visible_measured = sum(bool(point.get("visible", True)) and not point.get("predicted", False) for point in ordered)
    confidence_values = [
        float(point.get("confidence", 0.0))
        for point in ordered
        if point.get("visible", True) and not point.get("predicted", False)
    ]
    valid_points: list[tuple[float, float, float]] = []
    for point in ordered:
        coordinate = _coord(point)
        if coordinate is not None:
            valid_points.append((coordinate[0], coordinate[1], float(point.get("timestamp_seconds", 0.0))))

    distance_cm = 0.0
    speeds: list[float] = []
    valid_time = 0.0
    moving_time = 0.0
    previous: tuple[float, float, float] | None = None
    for x_value, y_value, timestamp in valid_points:
        if previous is not None:
            dt = timestamp - previous[2]
            segment_cm = math.hypot(x_value - previous[0], y_value - previous[1])
            if 0.001 < dt <= 1.0:
                speed_mps = segment_cm / 100.0 / dt
                if speed_mps <= MAX_ROBOT_SPEED_MPS:
                    distance_cm += segment_cm
                    speeds.append(speed_mps)
                    valid_time += dt
                    if speed_mps >= MOVING_SPEED_MPS:
                        moving_time += dt
        previous = (x_value, y_value, timestamp)

    coordinate_coverage = 100.0 * len(valid_points) / max(1, len(ordered))
    visibility = 100.0 * visible_measured / max(1, len(ordered))
    mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    tracking_quality = max(0.0, min(100.0, 0.55 * coordinate_coverage + 0.25 * visibility + 20.0 * mean_confidence))
    activity = 100.0 * moving_time / valid_time if valid_time > 0.0 else 0.0
    distance_m = distance_cm / 100.0
    average_speed = sum(speeds) / len(speeds) if speeds else 0.0
    peak_speed = _percentile(speeds, 95.0)

    if len(valid_points) < 5 or valid_time < 0.15:
        status = "SIN DATOS"
    elif distance_m < 0.08:
        status = "SIN MOVIMIENTO"
    elif distance_m < 0.50:
        status = "MOVIMIENTO MÍNIMO"
    elif activity < 12.0:
        status = "ACTIVIDAD BAJA"
    else:
        status = "EN MOVIMIENTO"

    return {
        "distance_m": distance_m,
        "average_speed_mps": average_speed,
        "peak_speed_mps": peak_speed,
        "valid_time_seconds": valid_time,
        "moving_time_seconds": moving_time,
        "activity_percent": activity,
        "visibility_percent": visibility,
        "coordinate_coverage_percent": coordinate_coverage,
        "tracking_quality_percent": tracking_quality,
        "valid_coordinate_count": len(valid_points),
        "status": status,
    }


def _icon(event_type: str, team: str) -> str:
    mapping = {
        "red_card_robot_removed": "tarjeta_roja",
        "robot_grabbed_by_referee": "tarjeta_roja",
        "robot_collision_candidate": "choque",
        "ball_out_of_field": "balon_fuera",
        "robot_inactive_candidate": "inactivo",
        "robot_reactivated": "inactivo",
        "robot_entered_penalty_area": "penal_aliado" if team == "aliado" else "penal_rival",
    }
    if event_type == "goal":
        return "gol_rival" if team == "rival" else "gol_aliado"
    return mapping.get(event_type, "inactivo")


def _title(event_type: str) -> str:
    return {
        "goal": "¡GOL!",
        "red_card_robot_removed": "TARJETA ROJA",
        "robot_grabbed_by_referee": "ROBOT RETIRADO",
        "robot_collision_candidate": "POSIBLE CHOQUE",
        "ball_out_of_field": "BALÓN FUERA",
        "robot_inactive_candidate": "ROBOT INACTIVO",
        "robot_reactivated": "ROBOT REACTIVADO",
        "ball_missing_candidate": "BALÓN OCULTO",
        "ball_recovered": "BALÓN RECUPERADO",
        "possession_change": "CAMBIO DE POSESIÓN",
        "robot_entered_penalty_area": "ENTRADA AL ÁREA",
        "referee_intervention_candidate": "INTERVENCIÓN ARBITRAL",
        "pass": "PASE",
        "pass_completed": "PASE COMPLETADO",
        "shot": "TIRO",
        "shot_on_target": "TIRO A PUERTA",
    }.get(event_type, event_type.replace("_", " ").upper())


def _event_description(event: dict[str, Any]) -> str:
    data = event.get("data") or {}
    description = str(event.get("description") or "").strip()
    if description:
        # Event pages are multipage, so preserve the complete detector description.
        return description
    return str(data.get("robot_name") or data.get("robot_a_name") or "Evento detectado")


def _control_percentages(tracks: dict[str, Any], team_map: dict[str, str]) -> dict[str, float]:
    bins_x, bins_y = 18, 14
    ally = [[0 for _ in range(bins_y)] for _ in range(bins_x)]
    rival = [[0 for _ in range(bins_y)] for _ in range(bins_x)]
    for raw_robot_id, points in (tracks.get("robots") or {}).items():
        robot_id = _normalize_robot_id(raw_robot_id) or str(raw_robot_id)
        team = team_map.get(robot_id, "desconocido")
        target = ally if team == "aliado" else rival if team == "rival" else None
        if target is None:
            continue
        # Sampling limits the influence of long stationary stretches.
        for index, point in enumerate(points):
            if index % 3:
                continue
            coordinate = _coord(point)
            if coordinate is None:
                continue
            x_bin = min(bins_x - 1, max(0, int(coordinate[0] / FIELD_LENGTH_CM * bins_x)))
            y_bin = min(bins_y - 1, max(0, int(coordinate[1] / FIELD_WIDTH_CM * bins_y)))
            target[x_bin][y_bin] += 1

    ally_control = rival_control = visited = 0
    for x_bin in range(bins_x):
        for y_bin in range(bins_y):
            ally_value = ally[x_bin][y_bin]
            rival_value = rival[x_bin][y_bin]
            if ally_value == 0 and rival_value == 0:
                continue
            visited += 1
            if ally_value > rival_value:
                ally_control += 1
            elif rival_value > ally_value:
                rival_control += 1
            else:
                ally_control += 0.5
                rival_control += 0.5
    if not visited:
        return {"aliado": 0.0, "rival": 0.0, "visited_percent": 0.0}
    return {
        "aliado": 100.0 * ally_control / visited,
        "rival": 100.0 * rival_control / visited,
        "visited_percent": 100.0 * visited / (bins_x * bins_y),
    }


def _format_stat(
    name: str,
    ally_value: float,
    rival_value: float,
    *,
    suffix: str = "",
    decimals: int = 0,
    icon: str = "•",
    left_percent: float | None = None,
) -> dict[str, Any]:
    if left_percent is None:
        total = max(0.0, float(ally_value)) + max(0.0, float(rival_value))
        left_percent = 50.0 if total <= 0 else 100.0 * max(0.0, float(ally_value)) / total
    formatter = f"{{:.{decimals}f}}"
    return {
        "icono": icon,
        "nombre": name,
        "aliado": round(float(ally_value), decimals),
        "rival": round(float(rival_value), decimals),
        "aliado_text": formatter.format(float(ally_value)) + suffix,
        "rival_text": formatter.format(float(rival_value)) + suffix,
        "bar_left_pct": max(0.0, min(100.0, float(left_percent))),
    }


def build_report_data(
    events_path: str | Path,
    summary_path: str | Path,
    tracks_path: str | Path,
    max_featured_events: int = 8,
    *,
    metadata_path: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    raw_events = _load(events_path, [])
    events = list(raw_events.get("events") or []) if isinstance(raw_events, dict) else list(raw_events or [])
    summary = _load(summary_path, {})
    tracks = _load(tracks_path, {"robots": {}, "ball": []})
    metadata = _load(metadata_path, {})
    duration = float(metadata.get("duration_seconds") or _duration(tracks))
    team_map = _team_map(tracks, output_directory)
    event_counts = Counter(str(event.get("event_type", "unknown")) for event in events)

    # Score is computed from every confirmed goal, not only the editorial timeline.
    score = {"aliado": 0, "rival": 0}
    unknown_goals = 0
    sorted_events = sorted(events, key=lambda event: float(event.get("timestamp_seconds", 0.0)))
    cumulative_score_by_event: dict[tuple[int, str], tuple[int, int]] = {}
    for event in sorted_events:
        event_type = str(event.get("event_type", "unknown"))
        team = _event_team(event, team_map)
        if event_type == "goal":
            if team in score:
                score[team] += 1
            else:
                unknown_goals += 1
        cumulative_score_by_event[(int(event.get("frame_index", 0)), event_type)] = (score["aliado"], score["rival"])

    # The PDF timeline is an audit log, not an editorial highlight reel.
    # Include every analyzed event in chronological order.  The legacy
    # ``max_featured_events`` argument is kept only for API compatibility.
    timeline: list[dict[str, Any]] = []
    for event_index, event in enumerate(sorted_events):
        event_type = str(event.get("event_type", "unknown"))
        key = (int(event.get("frame_index", 0)), event_type)
        team = _event_team(event, team_map)
        event_score = cumulative_score_by_event.get(key, (0, 0))
        timeline.append(
            {
                "index": event_index + 1,
                "tiempo": _time(float(event.get("timestamp_seconds", 0.0))),
                "imagen": _icon(event_type, team),
                "equipo": team if team != "desconocido" else "",
                "titulo": _title(event_type),
                "desc": _event_description(event),
                "score": f"{event_score[0]} - {event_score[1]}" if event_type == "goal" else "",
                "event_type": event_type,
                "frame_index": int(event.get("frame_index", 0)),
            }
        )
    timeline.append(
        {
            "index": len(timeline) + 1,
            "tiempo": _time(duration),
            "imagen": "fin",
            "equipo": "",
            "titulo": "FIN DEL PARTIDO",
            "desc": "",
            "score": f"{score['aliado']} - {score['rival']}",
            "event_type": "match_end",
            "frame_index": None,
        }
    )

    # Event pages start on PDF page 2. Twelve rows leave enough room for
    # complete descriptions while keeping each event together in Chromium
    # and WeasyPrint.
    events_per_page = 12
    event_pages = [
        timeline[index : index + events_per_page]
        for index in range(0, len(timeline), events_per_page)
    ] or [[]]
    total_page_count = 1 + len(event_pages)
    event_page_models = [
        {
            "events": page_events,
            "page_number": page_index + 2,
            "total_pages": total_page_count,
            "first_event_index": page_events[0]["index"] if page_events else 0,
            "last_event_index": page_events[-1]["index"] if page_events else 0,
        }
        for page_index, page_events in enumerate(event_pages)
    ]

    # Possession: distinguish confirmed team ownership from free/unassigned time.
    possession_seconds = {"aliado": 0.0, "rival": 0.0, "desconocido": 0.0}
    for raw_robot_id, seconds in (summary.get("possession_seconds") or {}).items():
        robot_id = _normalize_robot_id(raw_robot_id) or str(raw_robot_id)
        team = team_map.get(robot_id, "desconocido")
        possession_seconds[team] += max(0.0, float(seconds))
    known_possession = possession_seconds["aliado"] + possession_seconds["rival"]
    total_owned = known_possession + possession_seconds["desconocido"]
    free_seconds = max(0.0, duration - total_owned)
    possession_percent = {
        "aliado": round(100.0 * possession_seconds["aliado"] / known_possession) if known_possession > 0 else 0,
        "rival": round(100.0 * possession_seconds["rival"] / known_possession) if known_possession > 0 else 0,
    }

    # Event counts by team and per robot.
    by_team: dict[str, Counter[str]] = defaultdict(Counter)
    per_robot: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        team = _event_team(event, team_map)
        if team in {"aliado", "rival"}:
            by_team[team][event_type] += 1
        robot_ids = _event_robot_ids(event)
        for robot_id in robot_ids:
            per_robot[robot_id][event_type] += 1
        if event_type == "robot_collision_candidate":
            involved_teams = {team_map.get(robot_id, "desconocido") for robot_id in robot_ids}
            for involved_team in involved_teams:
                if involved_team in {"aliado", "rival"}:
                    by_team[involved_team]["collision_involvement"] += 1

    # Robot and team motion metrics.
    robot_metrics: dict[str, dict[str, Any]] = {}
    team_aggregate: dict[str, dict[str, float]] = {
        "aliado": defaultdict(float),
        "rival": defaultdict(float),
    }
    robots: list[dict[str, Any]] = []
    for raw_robot_id, points in (tracks.get("robots") or {}).items():
        robot_id = _normalize_robot_id(raw_robot_id) or str(raw_robot_id)
        motion = _motion_metrics(points)
        robot_metrics[robot_id] = motion
        team = team_map.get(robot_id, "desconocido")
        names = Counter(
            str(point.get("display_name") or robot_id)
            for point in points
            if point.get("visible", True)
        )
        display_name = names.most_common(1)[0][0] if names else robot_id.replace("robot_", "Robot ")
        possession_time = float((summary.get("possession_seconds") or {}).get(robot_id, 0.0))
        possession_changes = per_robot[robot_id]["possession_change"]
        area_entries = per_robot[robot_id]["robot_entered_penalty_area"]
        collisions = per_robot[robot_id]["robot_collision_candidate"]
        inactivity = per_robot[robot_id]["robot_inactive_candidate"]
        event_total = sum(per_robot[robot_id].values())
        status = motion["status"]
        if inactivity and status == "EN MOVIMIENTO":
            status = "INTERMITENTE"
        if team in team_aggregate:
            aggregate = team_aggregate[team]
            aggregate["distance_m"] += motion["distance_m"]
            aggregate["speed_weighted_sum"] += motion["average_speed_mps"] * motion["valid_time_seconds"]
            aggregate["valid_time_seconds"] += motion["valid_time_seconds"]
            aggregate["moving_time_seconds"] += motion["moving_time_seconds"]
            aggregate["robot_count"] += 1

        enough_motion_data = motion["valid_coordinate_count"] >= 5
        robots.append(
            {
                "robot_id": robot_id,
                "nombre": display_name,
                "equipo": team,
                "distancia": f"{motion['distance_m']:.1f} m" if enough_motion_data else "N/D",
                "vel": f"{motion['average_speed_mps']:.2f} m/s" if enough_motion_data else "N/D",
                "vel_pico": f"{motion['peak_speed_mps']:.2f} m/s" if enough_motion_data else "N/D",
                "posesion": _seconds(possession_time),
                "posesiones": str(possession_changes),
                "entradas_area": str(area_entries),
                "colisiones": str(collisions),
                "eventos": str(event_total),
                "inactividad": str(inactivity),
                "estado": status,
                "actividad": round(motion["activity_percent"]) if enough_motion_data else None,
                "actividad_text": f"{round(motion['activity_percent'])}%" if enough_motion_data else "N/D",
                "calidad_tracking": round(motion["tracking_quality_percent"]),
                "visibility": motion["visibility_percent"],
            }
        )

    def team_metric(team: str, key: str) -> float:
        return float(team_aggregate[team].get(key, 0.0))

    team_speed: dict[str, float] = {}
    team_activity: dict[str, float] = {}
    for team in ("aliado", "rival"):
        valid_time = team_metric(team, "valid_time_seconds")
        team_speed[team] = team_metric(team, "speed_weighted_sum") / valid_time if valid_time > 0 else 0.0
        team_activity[team] = 100.0 * team_metric(team, "moving_time_seconds") / valid_time if valid_time > 0 else 0.0

    control = _control_percentages(tracks, team_map)

    stats = [
        _format_stat(
            "Posesión confirmada",
            possession_percent["aliado"],
            possession_percent["rival"],
            suffix="%",
            icon="●",
            left_percent=possession_percent["aliado"] if known_possession > 0 else 50.0,
        ),
        _format_stat(
            "Tiempo con balón",
            possession_seconds["aliado"],
            possession_seconds["rival"],
            suffix=" s",
            decimals=1,
            icon="◷",
        ),
        _format_stat("Goles confirmados", score["aliado"], score["rival"], icon="◆"),
        _format_stat(
            "Entradas al área",
            by_team["aliado"]["robot_entered_penalty_area"],
            by_team["rival"]["robot_entered_penalty_area"],
            icon="⌂",
        ),
        _format_stat(
            "Distancia total",
            team_metric("aliado", "distance_m"),
            team_metric("rival", "distance_m"),
            suffix=" m",
            decimals=1,
            icon="↝",
        ),
        _format_stat(
            "Velocidad media",
            team_speed["aliado"],
            team_speed["rival"],
            suffix=" m/s",
            decimals=2,
            icon="↯",
        ),
        _format_stat(
            "Control de zonas",
            control["aliado"],
            control["rival"],
            suffix="%",
            decimals=0,
            icon="▦",
            left_percent=control["aliado"] if control["aliado"] + control["rival"] > 0 else 50.0,
        ),
        _format_stat(
            "Alertas de inactividad",
            by_team["aliado"]["robot_inactive_candidate"],
            by_team["rival"]["robot_inactive_candidate"],
            icon="■",
        ),
        _format_stat(
            "Colisiones implicadas",
            by_team["aliado"]["collision_involvement"],
            by_team["rival"]["collision_involvement"],
            icon="✦",
        ),
        _format_stat(
            "Tarjetas rojas",
            by_team["aliado"]["red_card_robot_removed"],
            by_team["rival"]["red_card_robot_removed"],
            icon="▮",
        ),
    ]

    order = {"aliado": 0, "rival": 1, "desconocido": 2}
    robots = sorted(robots, key=lambda robot: (order.get(robot["equipo"], 2), -robot["visibility"], robot["nombre"]))[:4]
    # Ensure the report always renders four cards without inventing statistics.
    while len(robots) < 4:
        index = len(robots) + 1
        robots.append(
            {
                "robot_id": f"missing_{index}",
                "nombre": f"Robot {index}",
                "equipo": "desconocido",
                "distancia": "N/D",
                "vel": "N/D",
                "vel_pico": "N/D",
                "posesion": "N/D",
                "posesiones": "0",
                "entradas_area": "0",
                "colisiones": "0",
                "eventos": "0",
                "inactividad": "0",
                "estado": "SIN DATOS",
                "actividad": None,
                "actividad_text": "N/D",
                "calidad_tracking": 0,
                "visibility": 0.0,
            }
        )

    metadata_candidate = Path(metadata_path) if metadata_path else None
    timestamp_source = metadata_candidate if metadata_candidate and metadata_candidate.exists() else Path(tracks_path)
    dt = datetime.fromtimestamp(timestamp_source.stat().st_mtime)
    name = str(metadata.get("video_name") or Path(output_directory or tracks_path).name)

    ball_points = tracks.get("ball") or []
    ball_visible = sum(bool(point.get("visible", True)) for point in ball_points)
    ball_visibility = 100.0 * ball_visible / len(ball_points) if ball_points else 0.0
    all_robot_points = [point for points in (tracks.get("robots") or {}).values() for point in points]
    valid_field_points = sum(_coord(point) is not None for point in all_robot_points)
    field_coverage = 100.0 * valid_field_points / len(all_robot_points) if all_robot_points else 0.0

    observations: list[str] = []
    unknown_robots = [robot["nombre"] for robot in robots if robot["equipo"] == "desconocido" and not robot["robot_id"].startswith("missing_")]
    if unknown_robots:
        observations.append("Equipo sin confirmar: " + ", ".join(unknown_robots) + ".")
    stopped = [robot["nombre"] for robot in robots if robot["estado"] == "SIN MOVIMIENTO"]
    if stopped:
        observations.append("Sin movimiento medible: " + ", ".join(stopped) + ".")
    low_data = [robot["nombre"] for robot in robots if robot["estado"] == "SIN DATOS"]
    if low_data:
        observations.append("Datos métricos insuficientes para: " + ", ".join(low_data) + ".")
    if event_counts["robot_collision_candidate"]:
        observations.append(f"Se detectaron {event_counts['robot_collision_candidate']} posibles colisiones para revisión.")
    observations.append(f"Visibilidad del balón: {ball_visibility:.1f}%.")
    observations.append(f"Cobertura métrica válida de robots: {field_coverage:.1f}%.")
    if duration > 0:
        observations.append(f"Balón libre o sin dueño confirmado: {100.0 * free_seconds / duration:.1f}% del tiempo.")
    if 0.0 < control["visited_percent"] < 35.0:
        observations.append(f"Control territorial estimado sobre {control['visited_percent']:.1f}% de las zonas visitadas.")
    if unknown_goals:
        observations.append(f"{unknown_goals} gol(es) no pudieron asignarse a un equipo.")

    # Real team header facts replace the old hard-coded strategies.
    team_cards = {
        "aliado": {
            "distancia": f"{team_metric('aliado', 'distance_m'):.1f} m",
            "posesion": _seconds(possession_seconds["aliado"]),
            "actividad": f"{team_activity['aliado']:.0f}%",
        },
        "rival": {
            "distancia": f"{team_metric('rival', 'distance_m'):.1f} m",
            "posesion": _seconds(possession_seconds["rival"]),
            "actividad": f"{team_activity['rival']:.0f}%",
        },
    }

    if score["aliado"] > score["rival"]:
        summary_text = "El equipo magenta terminó arriba en el marcador."
    elif score["rival"] > score["aliado"]:
        summary_text = "El equipo azul terminó arriba en el marcador."
    elif sum(score.values()):
        summary_text = "El partido terminó igualado."
    else:
        summary_text = "No se confirmó un ganador en los eventos analizados."
    if known_possession > 0:
        leader = "magenta" if possession_seconds["aliado"] > possession_seconds["rival"] else "azul" if possession_seconds["rival"] > possession_seconds["aliado"] else "ninguno"
        if leader != "ninguno":
            summary_text += f" La posesión confirmada favoreció al equipo {leader}."
    distance_leader = "magenta" if team_metric("aliado", "distance_m") > team_metric("rival", "distance_m") else "azul" if team_metric("rival", "distance_m") > team_metric("aliado", "distance_m") else None
    if distance_leader:
        summary_text += f" El equipo {distance_leader} acumuló mayor distancia recorrida."
    summary_text += f" Duración analizada: {_time(duration)}."
    if event_counts["robot_collision_candidate"]:
        summary_text += f" Se registraron {event_counts['robot_collision_candidate']} posibles colisiones."

    return {
        "fecha": dt.strftime("%d / %m / %Y"),
        "hora": dt.strftime("%H:%M:%S"),
        "duracion": f"{_time(duration)} min",
        "id_partido": name[:36],
        "marcador": score,
        "equipos": team_cards,
        "estadisticas": stats,
        "eventos": timeline,
        "event_pages": event_page_models,
        "event_count": len(timeline),
        "total_page_count": total_page_count,
        "events_per_page": events_per_page,
        "robots": robots,
        "resumen": summary_text,
        "observaciones": observations[:6],
        "posesion": {
            "aliado_seconds": possession_seconds["aliado"],
            "rival_seconds": possession_seconds["rival"],
            "unknown_seconds": possession_seconds["desconocido"],
            "free_seconds": free_seconds,
            "aliado_percent_confirmed": possession_percent["aliado"],
            "rival_percent_confirmed": possession_percent["rival"],
            "free_percent_match": 100.0 * free_seconds / duration if duration > 0 else 0.0,
        },
        "control": control,
        "calidad_datos": {
            "ball_visibility_percent": ball_visibility,
            "field_coordinate_coverage_percent": field_coverage,
            "unknown_robot_count": len(unknown_robots),
        },
        "team_assignments": team_map,
        "report_version": "11.5.2",
    }
