"""Agrupamiento visual de tres o cuatro robots en dos familias."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.C_quick_view.team_features import TeamFeature, team_feature_distance


@dataclass(frozen=True)
class PairingResult:
    pair_a: tuple[int, ...]
    pair_b: tuple[int, ...]
    cost: float
    second_best_cost: float
    margin: float
    confidence: float
    pair_distances: dict[str, float]


def _robust_pair_distance(samples_a: list[TeamFeature], samples_b: list[TeamFeature]) -> float:
    if not samples_a or not samples_b:
        return float("inf")
    distances = np.array(
        [team_feature_distance(a, b) for a in samples_a for b in samples_b],
        dtype=np.float64,
    )
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("inf")
    # Algunos robots solo aportan una o dos muestras limpias. Limitar
    # ``keep`` evita solicitar un índice inexistente a ``np.partition``.
    keep = min(
        int(distances.size),
        max(3, int(np.ceil(0.22 * distances.size))),
    )
    best = np.partition(distances, keep - 1)[:keep]
    return float(np.median(best))


def solve_two_team_pairing(samples_by_id: dict[int, list[TeamFeature]]) -> PairingResult | None:
    robot_ids = sorted(robot_id for robot_id, samples in samples_by_id.items() if samples)
    if len(robot_ids) not in {3, 4}:
        return None

    pair_distance: dict[tuple[int, int], float] = {}
    for first, second in combinations(robot_ids, 2):
        pair_distance[(first, second)] = _robust_pair_distance(
            samples_by_id[first], samples_by_id[second]
        )

    scored: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    if len(robot_ids) == 4:
        a, b, c, d = robot_ids
        candidates = [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]
        for group_a, group_b in candidates:
            cost = pair_distance[tuple(sorted(group_a))] + pair_distance[tuple(sorted(group_b))]
            scored.append((float(cost), tuple(sorted(group_a)), tuple(sorted(group_b))))
    else:
        # Con tres robots, la pareja más parecida forma un equipo y el robot
        # restante representa al otro. Esto cubre partidos con menos de cuatro.
        for first, second in combinations(robot_ids, 2):
            remaining = tuple(robot_id for robot_id in robot_ids if robot_id not in {first, second})
            scored.append((pair_distance[(first, second)], (first, second), remaining))

    scored.sort(key=lambda item: item[0])
    best_cost, best_a, best_b = scored[0]
    second_cost = scored[1][0]
    margin = max(0.0, second_cost - best_cost)
    relative_margin = margin / max(second_cost, 1e-6)
    confidence = float(np.clip(relative_margin / 0.18, 0.0, 1.0))
    readable = {
        f"{first}-{second}": round(distance, 6)
        for (first, second), distance in pair_distance.items()
    }
    return PairingResult(
        pair_a=best_a,
        pair_b=best_b,
        cost=best_cost,
        second_best_cost=second_cost,
        margin=relative_margin,
        confidence=confidence,
        pair_distances=readable,
    )
