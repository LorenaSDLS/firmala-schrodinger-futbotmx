"""Reconstrucción offline conservadora de identidades y equipos para FutBotMX V5.

El tracker online prioriza continuidad y velocidad, pero durante colisiones puede
intercambiar IDs. Este módulo procesa el video completo, divide las trayectorias
en tracklets, reconstruye hasta cuatro robots físicos, descubre las dos parejas
visualmente similares y reescribe el JSONL antes de generar eventos y replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.C_quick_view.team_features import (
    TeamFeature,
    extract_team_feature_views,
    rotation_invariant_distance,
)
from src.C_quick_view.yolo_detector import draw_yolo_detections


TEAM_COLORS_HEX = {
    "aliado": "#00AEEF",
    "rival": "#FF00B8",
    "desconocido": "#888888",
}


@dataclass
class OfflineIdentityConfig:
    sample_stride: int = 8
    minimum_sample_confidence: float = 0.64
    minimum_box_area: float = 700.0
    maximum_tracklet_gap_frames: int = 5
    appearance_split_threshold: float = 0.33
    impossible_jump_fraction: float = 0.20
    minimum_tracklet_measurements: int = 5
    maximum_samples_per_tracklet: int = 18
    robot_interpolation_seconds: float = 0.42
    swap_team_labels: bool = False
    render_corrected_preview: bool = True
    minimum_pairing_margin: float = 0.12
    maximum_identity_assignment_cost: float = 0.62
    minimum_identity_assignment_margin: float = 0.045
    preserve_unresolved_detections: bool = True
    force_pairing: bool = False

    @classmethod
    def from_json(
        cls,
        path: str | Path | None,
        *,
        swap_team_labels: bool = False,
    ) -> "OfflineIdentityConfig":
        config = cls(swap_team_labels=swap_team_labels)
        if path is None:
            return config
        config_path = Path(path)
        if not config_path.exists():
            return config
        data = json.loads(config_path.read_text(encoding="utf-8"))
        v4 = data.get("v4", {})
        config.sample_stride = max(1, int(v4.get("muestreo_cada_frames", config.sample_stride)))
        config.minimum_sample_confidence = float(
            v4.get("confianza_minima_muestra", config.minimum_sample_confidence)
        )
        config.minimum_box_area = float(
            v4.get("area_minima_caja", config.minimum_box_area)
        )
        config.maximum_tracklet_gap_frames = max(
            1,
            int(v4.get("maximo_hueco_tracklet_frames", config.maximum_tracklet_gap_frames)),
        )
        config.appearance_split_threshold = float(
            v4.get("umbral_cambio_apariencia", config.appearance_split_threshold)
        )
        config.minimum_tracklet_measurements = max(
            2,
            int(v4.get("mediciones_minimas_tracklet", config.minimum_tracklet_measurements)),
        )
        config.maximum_samples_per_tracklet = max(
            4,
            int(v4.get("muestras_maximas_tracklet", config.maximum_samples_per_tracklet)),
        )
        config.robot_interpolation_seconds = max(
            0.0,
            float(v4.get("interpolacion_robot_segundos", config.robot_interpolation_seconds)),
        )
        config.render_corrected_preview = bool(
            v4.get("renderizar_preview_corregido", config.render_corrected_preview)
        )
        v5 = data.get("v5", {})
        config.minimum_pairing_margin = max(
            0.0,
            float(v5.get("margen_minimo_parejas", config.minimum_pairing_margin)),
        )
        config.maximum_identity_assignment_cost = max(
            0.05,
            float(
                v5.get(
                    "costo_maximo_asignacion_identidad",
                    config.maximum_identity_assignment_cost,
                )
            ),
        )
        config.minimum_identity_assignment_margin = max(
            0.0,
            float(
                v5.get(
                    "margen_minimo_asignacion_identidad",
                    config.minimum_identity_assignment_margin,
                )
            ),
        )
        config.preserve_unresolved_detections = bool(
            v5.get(
                "conservar_detecciones_no_resueltas",
                config.preserve_unresolved_detections,
            )
        )
        config.force_pairing = bool(v5.get("forzar_parejas", config.force_pairing))
        return config


@dataclass
class Observation:
    frame_index: int
    detection_index: int
    timestamp_seconds: float
    online_id: int
    bbox: list[float]
    confidence: float
    measured: bool
    center: np.ndarray
    size: np.ndarray
    feature_views: list[TeamFeature] = field(default_factory=list)


@dataclass
class Tracklet:
    tracklet_id: int
    online_id: int
    observations: list[Observation] = field(default_factory=list)
    feature_views: list[list[TeamFeature]] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return min(observation.frame_index for observation in self.observations)

    @property
    def end_frame(self) -> int:
        return max(observation.frame_index for observation in self.observations)

    @property
    def measured_count(self) -> int:
        return sum(observation.measured for observation in self.observations)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def quality(self) -> float:
        mean_confidence = float(
            np.mean([observation.confidence for observation in self.observations])
        )
        return float(
            min(1.0, self.measured_count / 18.0)
            * min(1.0, len(self.feature_views) / 6.0)
            * mean_confidence
        )

    @property
    def first_center(self) -> np.ndarray:
        return self.observations[0].center

    @property
    def last_center(self) -> np.ndarray:
        return self.observations[-1].center

    def overlaps(self, other: "Tracklet", tolerance_frames: int = 0) -> bool:
        return not (
            self.end_frame < other.start_frame - tolerance_frames
            or other.end_frame < self.start_frame - tolerance_frames
        )


@dataclass
class PhysicalIdentityResult:
    tracklet_to_physical: dict[int, int]
    team_by_physical: dict[int, str]
    team_number_by_physical: dict[int, int]
    tracklets: list[Tracklet]
    cluster_members: dict[int, list[int]]
    pair_distances: dict[str, float]
    selected_pairing: list[list[int]]
    dropped_tracklets: list[int]
    identity_count: int
    clustering_score: float


def _center_size(box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = map(float, box)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    size = np.array([max(2.0, x2 - x1), max(2.0, y2 - y1)], dtype=np.float64)
    return center, size


def _inside_expanded_field(box: list[float], field_box: list[float] | None) -> bool:
    if field_box is None:
        return True
    x1, y1, x2, y2 = map(float, field_box)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_x = 0.08 * width
    margin_y = 0.10 * height
    bx1, by1, bx2, by2 = map(float, box)
    anchor_x = (bx1 + bx2) / 2.0
    anchor_y = by2
    return (
        x1 - margin_x <= anchor_x <= x2 + margin_x
        and y1 - margin_y <= anchor_y <= y2 + margin_y
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _robust_views_distance(
    samples_a: list[list[TeamFeature]],
    samples_b: list[list[TeamFeature]],
    *,
    identity: bool,
) -> float:
    if not samples_a or not samples_b:
        return 1.0
    # Limitar el producto cartesiano mantiene el costo acotado en videos largos.
    subset_a = samples_a[-12:]
    subset_b = samples_b[-12:]
    distances = np.array(
        [
            rotation_invariant_distance(a, b, identity=identity)
            for a in subset_a
            for b in subset_b
        ],
        dtype=np.float64,
    )
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return 1.0
    # ``np.partition`` exige que ``kth`` esté dentro del arreglo. En
    # tracklets con una sola vista solo existe una distancia válida, por
    # lo que el mínimo estadístico solicitado debe limitarse al número
    # real de muestras disponibles.
    keep = min(
        int(distances.size),
        max(2, int(np.ceil(0.30 * distances.size))),
    )
    best = np.partition(distances, keep - 1)[:keep]
    return float(np.median(best))


def _tracklet_identity_distance(
    first: Tracklet,
    second: Tracklet,
    frame_diagonal: float,
    fps: float,
) -> float:
    if first.overlaps(second):
        return float("inf")
    appearance = _robust_views_distance(
        first.feature_views,
        second.feature_views,
        identity=True,
    )

    if first.end_frame < second.start_frame:
        earlier, later = first, second
    else:
        earlier, later = second, first
    gap_frames = max(1, later.start_frame - earlier.end_frame)
    gap_seconds = gap_frames / max(fps, 1.0)
    displacement = float(np.linalg.norm(later.first_center - earlier.last_center))
    displacement /= max(frame_diagonal, 1.0)
    physically_reachable = 0.035 + 0.52 * gap_seconds
    motion_penalty = max(0.0, displacement / max(physically_reachable, 1e-6) - 1.0)
    same_online_bonus = -0.035 if first.online_id == second.online_id else 0.0
    return float(0.78 * appearance + 0.22 * min(2.0, motion_penalty) + same_online_bonus)


def _cluster_cost(
    tracklet: Tracklet,
    members: list[Tracklet],
    distance_matrix: dict[tuple[int, int], float],
) -> float:
    if any(tracklet.overlaps(member) for member in members):
        return float("inf")
    distances = [
        distance_matrix[tuple(sorted((tracklet.tracklet_id, member.tracklet_id)))]
        for member in members
    ]
    finite = [distance for distance in distances if np.isfinite(distance)]
    if not finite:
        return float("inf")
    finite.sort()
    return float(np.mean(finite[: min(3, len(finite))]))


def _initial_medoids(
    tracklets: list[Tracklet],
    k: int,
    distance_matrix: dict[tuple[int, int], float],
) -> list[int]:
    ordered = sorted(
        tracklets,
        key=lambda item: (item.measured_count, item.quality),
        reverse=True,
    )
    medoids = [ordered[0].tracklet_id]
    while len(medoids) < k:
        best_id = None
        best_score = -1.0
        for candidate in ordered:
            if candidate.tracklet_id in medoids:
                continue
            distances = []
            overlap_bonus = 0.0
            for medoid_id in medoids:
                medoid = next(item for item in tracklets if item.tracklet_id == medoid_id)
                if candidate.overlaps(medoid):
                    overlap_bonus += 0.35
                    distance = 1.0
                else:
                    distance = distance_matrix[
                        tuple(sorted((candidate.tracklet_id, medoid_id)))
                    ]
                distances.append(distance if np.isfinite(distance) else 1.0)
            score = min(distances) + overlap_bonus
            if score > best_score:
                best_score = score
                best_id = candidate.tracklet_id
        if best_id is None:
            break
        medoids.append(best_id)
    return medoids


def _constrained_k_medoids(
    tracklets: list[Tracklet],
    k: int,
    distance_matrix: dict[tuple[int, int], float],
    *,
    maximum_assignment_cost: float,
    minimum_assignment_margin: float,
) -> tuple[dict[int, list[int]], list[int], float]:
    medoids = _initial_medoids(tracklets, k, distance_matrix)
    if len(medoids) < k:
        return {}, [item.tracklet_id for item in tracklets], float("inf")

    by_id = {item.tracklet_id: item for item in tracklets}
    unresolved: list[int] = []
    clusters: dict[int, list[int]] = {}

    for _ in range(6):
        clusters = {index: [medoid] for index, medoid in enumerate(medoids)}
        unresolved = []
        remaining = [item for item in tracklets if item.tracklet_id not in medoids]
        remaining.sort(key=lambda item: (item.measured_count, item.quality), reverse=True)

        for tracklet in remaining:
            options: list[tuple[float, int]] = []
            for cluster_index, member_ids in clusters.items():
                members = [by_id[member_id] for member_id in member_ids]
                cost = _cluster_cost(tracklet, members, distance_matrix)
                if np.isfinite(cost):
                    options.append((cost, cluster_index))
            if not options:
                unresolved.append(tracklet.tracklet_id)
                continue
            options.sort(key=lambda item: item[0])
            best_cost, selected = options[0]
            second_cost = options[1][0] if len(options) > 1 else float("inf")
            assignment_margin = second_cost - best_cost
            if (
                best_cost > maximum_assignment_cost
                or assignment_margin < minimum_assignment_margin
            ):
                unresolved.append(tracklet.tracklet_id)
                continue
            clusters[selected].append(tracklet.tracklet_id)

        new_medoids: list[int] = []
        for cluster_index in range(k):
            members = clusters[cluster_index]
            medoid_costs = []
            for candidate_id in members:
                total = 0.0
                for other_id in members:
                    if candidate_id == other_id:
                        continue
                    distance = distance_matrix[tuple(sorted((candidate_id, other_id)))]
                    total += distance if np.isfinite(distance) else 2.0
                medoid_costs.append((total, candidate_id))
            new_medoids.append(min(medoid_costs, key=lambda item: item[0])[1])
        if new_medoids == medoids:
            break
        medoids = new_medoids

    intra: list[float] = []
    for member_ids in clusters.values():
        for first_id, second_id in combinations(member_ids, 2):
            distance = distance_matrix[tuple(sorted((first_id, second_id)))]
            if np.isfinite(distance):
                intra.append(distance)
    mean_intra = float(np.mean(intra)) if intra else 0.0
    score = mean_intra + 0.55 * len(unresolved) / max(1, len(tracklets)) + 0.032 * k
    return clusters, unresolved, score


def _maximum_simultaneous(tracklets: list[Tracklet]) -> int:
    if not tracklets:
        return 0
    events: list[tuple[int, int]] = []
    for tracklet in tracklets:
        events.append((tracklet.start_frame, 1))
        events.append((tracklet.end_frame + 1, -1))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _select_identity_clusters(
    tracklets: list[Tracklet],
    frame_diagonal: float,
    fps: float,
    config: OfflineIdentityConfig,
) -> tuple[dict[int, list[int]], list[int], float]:
    if not tracklets:
        return {}, [], float("inf")
    distance_matrix: dict[tuple[int, int], float] = {}
    for first, second in combinations(tracklets, 2):
        distance_matrix[(first.tracklet_id, second.tracklet_id)] = _tracklet_identity_distance(
            first,
            second,
            frame_diagonal,
            fps,
        )

    lower_bound = max(1, min(4, _maximum_simultaneous(tracklets)))
    upper_bound = min(4, len(tracklets))
    candidates: list[tuple[float, dict[int, list[int]], list[int]]] = []
    for k in range(lower_bound, upper_bound + 1):
        clusters, unresolved, score = _constrained_k_medoids(
            tracklets,
            k,
            distance_matrix,
            maximum_assignment_cost=config.maximum_identity_assignment_cost,
            minimum_assignment_margin=config.minimum_identity_assignment_margin,
        )
        candidates.append((score, clusters, unresolved))
    score, clusters, unresolved = min(candidates, key=lambda item: item[0])
    return clusters, unresolved, score


def _physical_pairing(
    clusters: dict[int, list[int]],
    by_tracklet: dict[int, Tracklet],
    swap_labels: bool,
    *,
    minimum_pairing_margin: float,
    force_pairing: bool,
) -> tuple[
    dict[int, str],
    dict[int, int],
    dict[str, float],
    list[list[int]],
    bool,
    float,
]:
    """Descubre las dos parejas sin imponer una respuesta ambigua.

    La mejor división solo se acepta cuando supera claramente a la segunda.
    De lo contrario, las identidades físicas permanecen como ``desconocido``.
    """
    physical_samples: dict[int, list[list[TeamFeature]]] = {}
    for physical_id, tracklet_ids in clusters.items():
        physical_samples[physical_id] = [
            views
            for tracklet_id in tracklet_ids
            for views in by_tracklet[tracklet_id].feature_views
        ]

    ids = sorted(physical_samples)
    pair_distances: dict[tuple[int, int], float] = {}
    for first, second in combinations(ids, 2):
        pair_distances[(first, second)] = _robust_views_distance(
            physical_samples[first],
            physical_samples[second],
            identity=False,
        )

    candidates: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]]]] = []
    if len(ids) >= 4:
        a, b, c, d = ids[:4]
        candidates = [
            (pair_distances[(a, b)] + pair_distances[(c, d)], ((a, b), (c, d))),
            (pair_distances[(a, c)] + pair_distances[(b, d)], ((a, c), (b, d))),
            (pair_distances[(a, d)] + pair_distances[(b, c)], ((a, d), (b, c))),
        ]
    elif len(ids) == 3:
        for first, second in combinations(ids, 2):
            remaining = tuple(robot_id for robot_id in ids if robot_id not in {first, second})
            candidates.append((pair_distances[(first, second)], ((first, second), remaining)))

    candidates.sort(key=lambda item: item[0])
    proposed: list[tuple[int, ...]] = []
    pairing_margin = 0.0
    pairing_confirmed = False
    if candidates:
        best_cost, best_groups = candidates[0]
        proposed = [tuple(sorted(group)) for group in best_groups]
        if len(candidates) > 1:
            second_cost = candidates[1][0]
            pairing_margin = max(0.0, second_cost - best_cost) / max(second_cost, 1e-6)
            pairing_confirmed = pairing_margin >= minimum_pairing_margin
        elif force_pairing:
            pairing_confirmed = True
            pairing_margin = 1.0
        if force_pairing:
            pairing_confirmed = True

    if pairing_confirmed and proposed:
        ally_pair = tuple(proposed[0])
        rival_pair = tuple(proposed[1]) if len(proposed) > 1 else ()
        # Regla determinista: el grupo que contiene el ID físico menor se llama
        # aliado. ``invertir_etiquetas`` permite intercambiar solo los nombres.
        all_members = sorted((*ally_pair, *rival_pair))
        if all_members and all_members[0] not in ally_pair:
            ally_pair, rival_pair = rival_pair, ally_pair
        if swap_labels:
            ally_pair, rival_pair = rival_pair, ally_pair
        team_by_physical = {
            **{physical_id: "aliado" for physical_id in ally_pair},
            **{physical_id: "rival" for physical_id in rival_pair},
        }
    else:
        team_by_physical = {physical_id: "desconocido" for physical_id in ids}

    team_number: dict[int, int] = {}
    for team in ("aliado", "rival"):
        members = sorted(
            physical_id
            for physical_id, assigned_team in team_by_physical.items()
            if assigned_team == team
        )
        for index, physical_id in enumerate(members, start=1):
            team_number[physical_id] = index

    readable = {
        f"{first}-{second}": round(distance, 6)
        for (first, second), distance in pair_distances.items()
    }
    return (
        team_by_physical,
        team_number,
        readable,
        [list(group) for group in proposed],
        pairing_confirmed,
        float(pairing_margin),
    )

def _run_rts_smoother(
    measurements: dict[int, tuple[np.ndarray, float]],
    start_frame: int,
    end_frame: int,
) -> dict[int, np.ndarray]:
    """Kalman de velocidad constante seguido por suavizado RTS."""
    dimension = 4  # cx, cy, ancho, alto; todas normalizadas.
    state_dimension = dimension * 2
    transition = np.eye(state_dimension, dtype=np.float64)
    transition[:dimension, dimension:] = np.eye(dimension)
    observation = np.zeros((dimension, state_dimension), dtype=np.float64)
    observation[:, :dimension] = np.eye(dimension)
    process_noise = np.diag(
        [2e-5, 2e-5, 8e-6, 8e-6, 8e-5, 8e-5, 3e-5, 3e-5]
    )
    identity = np.eye(state_dimension)

    frame_count = end_frame - start_frame + 1
    filtered_states = np.zeros((frame_count, state_dimension), dtype=np.float64)
    filtered_covariances = np.zeros((frame_count, state_dimension, state_dimension))
    predicted_states = np.zeros_like(filtered_states)
    predicted_covariances = np.zeros_like(filtered_covariances)

    first_measurement = measurements[min(measurements)][0]
    state = np.zeros(state_dimension, dtype=np.float64)
    state[:dimension] = first_measurement
    covariance = np.eye(state_dimension, dtype=np.float64) * 0.05

    for offset, frame_index in enumerate(range(start_frame, end_frame + 1)):
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process_noise
        predicted_states[offset] = predicted_state
        predicted_covariances[offset] = predicted_covariance

        if frame_index in measurements:
            measurement, confidence = measurements[frame_index]
            noise_scale = 1.0 / max(0.20, confidence)
            measurement_noise = np.diag(
                [2.8e-4, 2.8e-4, 4.5e-4, 4.5e-4]
            ) * noise_scale
            innovation = measurement - observation @ predicted_state
            innovation_covariance = (
                observation @ predicted_covariance @ observation.T + measurement_noise
            )
            gain = (
                predicted_covariance
                @ observation.T
                @ np.linalg.pinv(innovation_covariance)
            )
            state = predicted_state + gain @ innovation
            covariance = (identity - gain @ observation) @ predicted_covariance
        else:
            state = predicted_state
            covariance = predicted_covariance

        filtered_states[offset] = state
        filtered_covariances[offset] = covariance

    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for offset in range(frame_count - 2, -1, -1):
        smoother_gain = (
            filtered_covariances[offset]
            @ transition.T
            @ np.linalg.pinv(predicted_covariances[offset + 1])
        )
        smoothed_states[offset] = filtered_states[offset] + smoother_gain @ (
            smoothed_states[offset + 1] - predicted_states[offset + 1]
        )
        smoothed_covariances[offset] = filtered_covariances[offset] + smoother_gain @ (
            smoothed_covariances[offset + 1] - predicted_covariances[offset + 1]
        ) @ smoother_gain.T

    return {
        frame_index: smoothed_states[offset, :dimension]
        for offset, frame_index in enumerate(range(start_frame, end_frame + 1))
    }


def _apply_offline_smoothing(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    fps: float,
    maximum_gap_seconds: float,
) -> None:
    by_robot: dict[int, dict[int, dict[str, Any]]] = {}
    for record in records:
        frame_index = int(record["frame_index"])
        for detection in record.get("detections", []):
            if str(detection.get("class_group", "")).lower() != "robot":
                continue
            physical_id = detection.get("physical_robot_id")
            if physical_id is None or bool(detection.get("predicted", False)):
                continue
            by_robot.setdefault(int(physical_id), {})[frame_index] = detection

    maximum_gap_frames = max(0, int(round(maximum_gap_seconds * fps)))
    for physical_id, detections_by_frame in by_robot.items():
        if len(detections_by_frame) < 2:
            continue
        ordered_frames = sorted(detections_by_frame)
        measurements: dict[int, tuple[np.ndarray, float]] = {}
        for frame_index in ordered_frames:
            detection = detections_by_frame[frame_index]
            center, size = _center_size(list(map(float, detection["bbox_xyxy"])))
            vector = np.array(
                [
                    center[0] / width,
                    center[1] / height,
                    size[0] / width,
                    size[1] / height,
                ],
                dtype=np.float64,
            )
            measurements[frame_index] = (
                vector,
                float(detection.get("confidence", 0.6)),
            )

        smoothed = _run_rts_smoother(
            measurements,
            ordered_frames[0],
            ordered_frames[-1],
        )
        existing_frames = set(detections_by_frame)
        for frame_index, vector in smoothed.items():
            previous_measurements = [frame for frame in ordered_frames if frame <= frame_index]
            next_measurements = [frame for frame in ordered_frames if frame >= frame_index]
            if not previous_measurements or not next_measurements:
                continue
            previous_frame = previous_measurements[-1]
            next_frame = next_measurements[0]
            gap = next_frame - previous_frame - 1
            should_insert = (
                frame_index not in existing_frames
                and previous_frame < frame_index < next_frame
                and gap <= maximum_gap_frames
            )
            if frame_index not in existing_frames and not should_insert:
                continue

            cx = float(np.clip(vector[0], 0.0, 1.0) * width)
            cy = float(np.clip(vector[1], 0.0, 1.0) * height)
            box_width = float(np.clip(vector[2], 0.005, 0.45) * width)
            box_height = float(np.clip(vector[3], 0.005, 0.55) * height)
            bbox = [
                max(0.0, cx - box_width / 2.0),
                max(0.0, cy - box_height / 2.0),
                min(float(width), cx + box_width / 2.0),
                min(float(height), cy + box_height / 2.0),
            ]

            if should_insert:
                source = detections_by_frame[
                    previous_frame
                    if frame_index - previous_frame <= next_frame - frame_index
                    else next_frame
                ].copy()
                source.update(
                    {
                        "bbox_xyxy": [round(value, 2) for value in bbox],
                        "confidence": round(
                            0.32
                            * min(
                                float(detections_by_frame[previous_frame].get("confidence", 0.6)),
                                float(detections_by_frame[next_frame].get("confidence", 0.6)),
                            ),
                            6,
                        ),
                        "predicted": True,
                        "measured": False,
                        "source": "interpolado_offline",
                        "tracking_status": "interpolado",
                        "track_missed_frames": min(
                            frame_index - previous_frame,
                            next_frame - frame_index,
                        ),
                    }
                )
                records[frame_index]["detections"].append(source)
                detection = source
            else:
                detection = detections_by_frame[frame_index]
                detection["bbox_xyxy_raw"] = detection.get("bbox_xyxy")
                detection["bbox_xyxy"] = [round(value, 2) for value in bbox]
                detection["offline_smoothed"] = True

            x1, y1, x2, y2 = bbox
            anchor_x = (x1 + x2) / 2.0
            anchor_y = y2
            detection["anchor_x_px"] = round(anchor_x, 3)
            detection["anchor_y_px"] = round(anchor_y, 3)
            matrix_data = records[frame_index].get("camera_registration", {}).get("matrix")
            if matrix_data:
                matrix = np.asarray(matrix_data, dtype=np.float64)
                point = matrix @ np.array([anchor_x, anchor_y, 1.0])
                denominator = point[2] if abs(point[2]) > 1e-9 else 1.0
                detection["stabilized_x_px"] = round(float(point[0] / denominator), 3)
                detection["stabilized_y_px"] = round(float(point[1] / denominator), 3)

        # Elimina duplicados de la misma identidad en un cuadro.
        for record in records:
            robots = [
                detection
                for detection in record.get("detections", [])
                if str(detection.get("class_group", "")).lower() == "robot"
                and detection.get("physical_robot_id") == physical_id
            ]
            if len(robots) <= 1:
                continue
            keep = max(
                robots,
                key=lambda item: (
                    not bool(item.get("predicted", False)),
                    float(item.get("confidence", 0.0)),
                ),
            )
            record["detections"] = [
                detection
                for detection in record.get("detections", [])
                if not (
                    str(detection.get("class_group", "")).lower() == "robot"
                    and detection.get("physical_robot_id") == physical_id
                    and detection is not keep
                )
            ]



def _save_identity_contact_sheet(
    video_path: Path,
    output_path: Path,
    clusters: dict[int, list[int]],
    by_tracklet: dict[int, Tracklet],
    team_by_physical: dict[int, str],
    team_number: dict[int, int],
) -> str | None:
    """Guarda recortes representativos para auditar el agrupamiento V5."""
    selections: list[tuple[int, Observation]] = []
    for physical_id, tracklet_ids in sorted(clusters.items()):
        observations = [
            observation
            for tracklet_id in tracklet_ids
            for observation in by_tracklet[tracklet_id].observations
            if observation.measured
        ]
        observations.sort(
            key=lambda item: (
                bool(item.feature_views),
                item.confidence,
                float(item.size[0] * item.size[1]),
            ),
            reverse=True,
        )
        chosen: list[Observation] = []
        for observation in observations:
            if any(abs(observation.frame_index - prior.frame_index) < 10 for prior in chosen):
                continue
            chosen.append(observation)
            if len(chosen) >= 3:
                break
        selections.extend((physical_id, observation) for observation in chosen)

    if not selections:
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    cell_width, cell_height = 220, 170
    columns = 3
    rows = max(1, int(np.ceil(len(selections) / columns)))
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 24, dtype=np.uint8)
    try:
        for index, (physical_id, observation) in enumerate(selections):
            capture.set(cv2.CAP_PROP_POS_FRAMES, observation.frame_index)
            success, frame = capture.read()
            if not success:
                continue
            x1, y1, x2, y2 = map(int, map(round, observation.bbox))
            pad_x = max(4, int(0.08 * max(1, x2 - x1)))
            pad_y = max(4, int(0.08 * max(1, y2 - y1)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            available_height = cell_height - 34
            scale = min(
                (cell_width - 12) / max(1, crop.shape[1]),
                (available_height - 8) / max(1, crop.shape[0]),
            )
            resized = cv2.resize(
                crop,
                (
                    max(1, int(round(crop.shape[1] * scale))),
                    max(1, int(round(crop.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            row, column = divmod(index, columns)
            origin_x = column * cell_width
            origin_y = row * cell_height
            paste_x = origin_x + (cell_width - resized.shape[1]) // 2
            paste_y = origin_y + 4 + (available_height - resized.shape[0]) // 2
            sheet[
                paste_y : paste_y + resized.shape[0],
                paste_x : paste_x + resized.shape[1],
            ] = resized
            team = team_by_physical.get(physical_id, "desconocido")
            number = team_number.get(physical_id, physical_id + 1)
            name = (
                f"Aliado {number}"
                if team == "aliado"
                else f"Rival {number}"
                if team == "rival"
                else f"Robot {physical_id + 1}"
            )
            cv2.putText(
                sheet,
                f"Fisico {physical_id + 1} - {name}",
                (origin_x + 8, origin_y + cell_height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
    finally:
        capture.release()
    cv2.imwrite(str(output_path), sheet)
    return str(output_path)

def _render_corrected_preview(
    video_path: Path,
    records: list[dict[str, Any]],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("No se pudo generar la previsualización corregida V5.")

    progress = tqdm(total=len(records), desc="Renderizando preview V5", unit="frame")
    try:
        for record in records:
            success, frame = capture.read()
            if not success:
                break
            writer.write(draw_yolo_detections(frame, record.get("detections", [])))
            progress.update(1)
    finally:
        progress.close()
        capture.release()
        writer.release()


def reconstruct_physical_identities(
    video_path: str | Path,
    detections_path: str | Path,
    output_directory: str | Path,
    config: OfflineIdentityConfig | None = None,
) -> dict[str, Any]:
    config = config or OfflineIdentityConfig()
    video_path = Path(video_path)
    detections_path = Path(detections_path)
    output_directory = Path(output_directory)
    records = _read_jsonl(detections_path)
    if not records:
        raise RuntimeError("No hay detecciones para reconstruir identidades.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diagonal = float(np.hypot(width, height))

    tracklets: list[Tracklet] = []
    active_by_online_id: dict[int, Tracklet] = {}
    mapping_by_key: dict[tuple[int, int], int] = {}
    next_tracklet_id = 0
    progress = tqdm(total=len(records), desc="Reconstruyendo identidades", unit="frame")

    try:
        for record_index, record in enumerate(records):
            success, frame = capture.read()
            if not success:
                break
            frame_index = int(record.get("frame_index", record_index))
            timestamp = float(record.get("timestamp_seconds", frame_index / fps))
            field_detections = [
                detection
                for detection in record.get("detections", [])
                if str(detection.get("class_group", "")).lower() == "field"
            ]
            field_box = None
            if field_detections:
                field_box = max(
                    field_detections,
                    key=lambda item: float(item.get("confidence", 0.0)),
                ).get("bbox_xyxy")

            for detection_index, detection in enumerate(record.get("detections", [])):
                if str(detection.get("class_group", "")).lower() != "robot":
                    continue
                if detection.get("tracking_id") is None:
                    continue
                online_id = int(detection["tracking_id"])
                bbox = list(map(float, detection["bbox_xyxy"]))
                center, size = _center_size(bbox)
                confidence = float(detection.get("confidence", 0.0))
                measured = not bool(detection.get("predicted", False))
                box_area = float(size[0] * size[1])
                clean = (
                    measured
                    and confidence >= config.minimum_sample_confidence
                    and box_area >= config.minimum_box_area
                    and _inside_expanded_field(bbox, field_box)
                )
                active = active_by_online_id.get(online_id)
                feature_views: list[TeamFeature] = []
                sample_interval = max(1, config.sample_stride)
                if active is not None and len(active.feature_views) >= config.maximum_samples_per_tracklet:
                    # Aun se comprueba la apariencia ocasionalmente para poder
                    # cortar un ID intercambiado al final del partido.
                    sample_interval *= 4
                if clean and frame_index % sample_interval == 0:
                    feature_views = extract_team_feature_views(frame, bbox)

                split = active is None
                appearance_jump = False
                split_back_to_frame: int | None = None
                previous_active = active
                if active is not None:
                    gap = frame_index - active.end_frame
                    displacement = float(np.linalg.norm(center - active.last_center))
                    impossible_jump = (
                        gap <= 2
                        and displacement > config.impossible_jump_fraction * frame_diagonal
                    )
                    if feature_views and active.feature_views:
                        recent = active.feature_views[-5:]
                        appearance_distance = min(
                            rotation_invariant_distance(
                                feature_views,
                                prior,
                                identity=True,
                            )
                            for prior in recent
                        )
                        appearance_jump = appearance_distance > config.appearance_split_threshold
                        if appearance_jump:
                            sampled_frames = [
                                observation.frame_index
                                for observation in active.observations
                                if observation.feature_views
                            ]
                            if sampled_frames:
                                # El intercambio pudo ocurrir entre la última
                                # muestra y la actual. Retrocede al punto medio.
                                split_back_to_frame = (
                                    sampled_frames[-1] + frame_index
                                ) // 2 + 1
                    split = (
                        gap > config.maximum_tracklet_gap_frames
                        or impossible_jump
                        or appearance_jump
                    )

                if split:
                    active = Tracklet(
                        tracklet_id=next_tracklet_id,
                        online_id=online_id,
                    )
                    next_tracklet_id += 1
                    tracklets.append(active)
                    active_by_online_id[online_id] = active

                    if (
                        previous_active is not None
                        and appearance_jump
                        and split_back_to_frame is not None
                    ):
                        moved = [
                            observation
                            for observation in previous_active.observations
                            if observation.frame_index >= split_back_to_frame
                        ]
                        if moved:
                            previous_active.observations = [
                                observation
                                for observation in previous_active.observations
                                if observation.frame_index < split_back_to_frame
                            ]
                            previous_active.feature_views = [
                                observation.feature_views
                                for observation in previous_active.observations
                                if observation.feature_views
                            ][: config.maximum_samples_per_tracklet]
                            active.observations.extend(moved)
                            active.feature_views.extend(
                                [
                                    observation.feature_views
                                    for observation in moved
                                    if observation.feature_views
                                ][: config.maximum_samples_per_tracklet]
                            )
                            for moved_observation in moved:
                                mapping_by_key[
                                    (
                                        moved_observation.frame_index,
                                        moved_observation.detection_index,
                                    )
                                ] = active.tracklet_id

                observation = Observation(
                    frame_index=frame_index,
                    detection_index=detection_index,
                    timestamp_seconds=timestamp,
                    online_id=online_id,
                    bbox=bbox,
                    confidence=confidence,
                    measured=measured,
                    center=center,
                    size=size,
                    feature_views=feature_views,
                )
                active.observations.append(observation)
                if feature_views and len(active.feature_views) < config.maximum_samples_per_tracklet:
                    active.feature_views.append(feature_views)
                mapping_by_key[(frame_index, detection_index)] = active.tracklet_id
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    good_tracklets = [
        tracklet
        for tracklet in tracklets
        if tracklet.measured_count >= config.minimum_tracklet_measurements
        and tracklet.feature_views
    ]
    clusters_raw, unresolved_good, clustering_score = _select_identity_clusters(
        good_tracklets,
        frame_diagonal,
        fps,
        config,
    )
    if not clusters_raw:
        # Fallback conservador: si no hubo suficientes recortes limpios, no se
        # destruyen las detecciones. Se conserva un grupo por ID online.
        fallback_groups: dict[int, list[int]] = {}
        for tracklet in tracklets:
            if tracklet.measured_count < config.minimum_tracklet_measurements:
                continue
            fallback_groups.setdefault(tracklet.online_id, []).append(tracklet.tracklet_id)
        clusters_raw = {
            index: member_ids
            for index, (_, member_ids) in enumerate(
                sorted(fallback_groups.items(), key=lambda item: item[0])[:4]
            )
        }
        unresolved_good = []
        clustering_score = 1.0
    # Renumeración estable por aparición inicial.
    ordered_clusters = sorted(
        clusters_raw.values(),
        key=lambda member_ids: min(
            next(tracklet for tracklet in tracklets if tracklet.tracklet_id == tracklet_id).start_frame
            for tracklet_id in member_ids
        ),
    )
    clusters = {physical_id: member_ids for physical_id, member_ids in enumerate(ordered_clusters)}
    tracklet_to_physical = {
        tracklet_id: physical_id
        for physical_id, member_ids in clusters.items()
        for tracklet_id in member_ids
    }
    by_tracklet = {tracklet.tracklet_id: tracklet for tracklet in tracklets}

    # Tracklets cortos se recuperan solamente cuando son compatibles con una
    # identidad y no se superponen temporalmente. El resto se considera ruido.
    dropped_tracklets: list[int] = []
    unassigned = [
        tracklet
        for tracklet in tracklets
        if tracklet.tracklet_id not in tracklet_to_physical
    ]
    for tracklet in unassigned:
        options: list[tuple[float, int]] = []
        for physical_id, member_ids in clusters.items():
            members = [by_tracklet[member_id] for member_id in member_ids]
            if any(tracklet.overlaps(member) for member in members):
                continue
            identity_costs = [
                _tracklet_identity_distance(
                    tracklet,
                    member,
                    frame_diagonal,
                    fps,
                )
                for member in members
            ]
            finite_costs = [cost for cost in identity_costs if np.isfinite(cost)]
            if finite_costs:
                options.append((min(finite_costs), physical_id))
        options.sort(key=lambda item: item[0])
        best_option = options[0] if options else None
        second_cost = options[1][0] if len(options) > 1 else float("inf")
        if (
            best_option is not None
            and best_option[0] <= config.maximum_identity_assignment_cost
            and second_cost - best_option[0] >= config.minimum_identity_assignment_margin
        ):
            _, physical_id = best_option
            tracklet_to_physical[tracklet.tracklet_id] = physical_id
            clusters[physical_id].append(tracklet.tracklet_id)
        elif (
            tracklet.measured_count >= 2 * config.minimum_tracklet_measurements
            and tracklet.quality >= 0.45
            and len(clusters) < 4
        ):
            physical_id = max(clusters, default=-1) + 1
            clusters[physical_id] = [tracklet.tracklet_id]
            tracklet_to_physical[tracklet.tracklet_id] = physical_id
        else:
            dropped_tracklets.append(tracklet.tracklet_id)

    (
        team_by_physical,
        team_number,
        pair_distances,
        selected_pairing,
        team_pairing_confirmed,
        team_pairing_margin,
    ) = _physical_pairing(
        clusters,
        by_tracklet,
        config.swap_team_labels,
        minimum_pairing_margin=config.minimum_pairing_margin,
        force_pairing=config.force_pairing,
    )
    representatives_path = _save_identity_contact_sheet(
        video_path=video_path,
        output_path=output_directory / "identity_v5_representatives.jpg",
        clusters=clusters,
        by_tracklet=by_tracklet,
        team_by_physical=team_by_physical,
        team_number=team_number,
    )

    # Reescritura de detecciones con identidad física estable.
    for record in records:
        frame_index = int(record["frame_index"])
        rewritten: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(record.get("detections", [])):
            if str(detection.get("class_group", "")).lower() != "robot":
                rewritten.append(detection)
                continue
            tracklet_id = mapping_by_key.get((frame_index, detection_index))
            physical_id = tracklet_to_physical.get(tracklet_id) if tracklet_id is not None else None
            if physical_id is None:
                if config.preserve_unresolved_detections:
                    item = detection.copy()
                    item["identity_resolved_offline"] = False
                    item["team"] = "desconocido"
                    item["team_locked"] = False
                    online_id = item.get("tracking_id")
                    item["display_name"] = (
                        f"Robot {int(online_id) + 1}"
                        if online_id is not None and str(online_id).lstrip("-").isdigit()
                        else "Robot sin confirmar"
                    )
                    item["team_color"] = TEAM_COLORS_HEX["desconocido"]
                    rewritten.append(item)
                continue
            item = detection.copy()
            item["online_tracking_id"] = int(detection.get("tracking_id", -1))
            item["tracklet_id"] = int(tracklet_id)
            item["physical_robot_id"] = int(physical_id)
            item["tracking_id"] = int(physical_id)
            item["identity_resolved_offline"] = True
            team = team_by_physical.get(physical_id, "desconocido")
            number = team_number.get(physical_id, physical_id + 1)
            item["team"] = team
            item["team_number"] = number
            item["team_locked"] = bool(team_pairing_confirmed)
            item["display_name"] = (
                f"Aliado {number}"
                if team == "aliado"
                else f"Rival {number}"
                if team == "rival"
                else f"Robot {physical_id + 1}"
            )
            item["team_color"] = TEAM_COLORS_HEX[team]
            rewritten.append(item)
        # Dedupe por identidad física.
        robots_by_id: dict[int, dict[str, Any]] = {}
        others: list[dict[str, Any]] = []
        for item in rewritten:
            if str(item.get("class_group", "")).lower() != "robot":
                others.append(item)
                continue
            physical_id = item.get("physical_robot_id")

            if physical_id is None:
                others.append(item)
                continue
            physical_id = int(physical_id)
            current = robots_by_id.get(physical_id)
            
            if current is None or (
                not bool(item.get("predicted", False)),
                float(item.get("confidence", 0.0)),
            ) > (
                not bool(current.get("predicted", False)),
                float(current.get("confidence", 0.0)),
            ):
                robots_by_id[physical_id] = item
        record["detections"] = others + list(robots_by_id.values())

    _apply_offline_smoothing(
        records,
        width,
        height,
        fps,
        config.robot_interpolation_seconds,
    )
    _write_jsonl(detections_path, records)

    online_preview = output_directory / "quick_preview_online.mp4"
    current_preview = output_directory / "quick_preview.mp4"
    if current_preview.exists():
        if online_preview.exists():
            online_preview.unlink()
        current_preview.replace(online_preview)
    corrected_preview = current_preview
    if config.render_corrected_preview:
        _render_corrected_preview(
            video_path,
            records,
            corrected_preview,
            fps,
            width,
            height,
        )

    summary = {
        "version": "5.0",
        "identity_count": len(clusters),
        "clustering_score": round(float(clustering_score), 6),
        "tracklet_to_physical": {
            str(key): value for key, value in sorted(tracklet_to_physical.items())
        },
        "physical_clusters": {
            str(key): sorted(value) for key, value in sorted(clusters.items())
        },
        "team_by_physical": {
            str(key): value for key, value in sorted(team_by_physical.items())
        },
        "team_number_by_physical": {
            str(key): value for key, value in sorted(team_number.items())
        },
        "pair_distances": pair_distances,
        "selected_pairing": selected_pairing,
        "team_pairing_confirmed": bool(team_pairing_confirmed),
        "team_pairing_margin": round(float(team_pairing_margin), 6),
        "minimum_pairing_margin": round(float(config.minimum_pairing_margin), 6),
        "dropped_tracklets": sorted(dropped_tracklets),
        "unresolved_good_tracklets": sorted(unresolved_good),
        "tracklets": [
            {
                "tracklet_id": tracklet.tracklet_id,
                "online_id": tracklet.online_id,
                "start_frame": tracklet.start_frame,
                "end_frame": tracklet.end_frame,
                "duration_frames": tracklet.duration_frames,
                "measured_count": tracklet.measured_count,
                "appearance_samples": len(tracklet.feature_views),
                "quality": round(tracklet.quality, 6),
                "physical_robot_id": tracklet_to_physical.get(tracklet.tracklet_id),
            }
            for tracklet in tracklets
        ],
        "files": {
            "corrected_preview": str(corrected_preview),
            "online_preview": str(online_preview),
            "detections": str(detections_path),
            "representative_crops": representatives_path,
        },
    }
    summary_path = output_directory / "identity_v5.json"
    summary_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "summary_path": str(summary_path),
        "preview_path": str(corrected_preview),
        "online_preview_path": str(online_preview),
        "identity_count": len(clusters),
        "team_by_physical": summary["team_by_physical"],
        "selected_pairing": selected_pairing,
        "team_pairing_confirmed": bool(team_pairing_confirmed),
        "team_pairing_margin": round(float(team_pairing_margin), 6),
        "dropped_tracklets": sorted(dropped_tracklets),
        "representatives_path": representatives_path,
    }
