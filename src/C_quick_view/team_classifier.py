"""Clasificación automática de dos equipos por similitud de construcción.

No requiere conocer los robots previamente. Acumula múltiples vistas limpias de
cada track, construye descriptores principalmente estructurales y evalúa las
tres formas posibles de dividir cuatro robots en dos parejas.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from src.C_quick_view.team_clustering import PairingResult, solve_two_team_pairing
from src.C_quick_view.team_features import TeamFeature, extract_team_feature, team_feature_distance


TEAM_COLORS_HEX = {
    "aliado": "#00AEEF",
    "rival": "#FF00B8",
    "desconocido": "#888888",
}


class TeamClassifier:
    def __init__(
        self,
        mode: str | None = None,
        ally_appearance: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.mode = "auto"
        self.id_map: dict[int, str] = {}
        self.minimum_samples_per_robot = 8
        self.maximum_samples_per_robot = 48
        self.minimum_pairing_margin = 0.025
        self.minimum_detection_confidence = 0.66
        self.minimum_box_area = 850.0
        self.swap_team_labels = False
        self.samples_by_id: dict[int, deque[TeamFeature]] = {}
        self.team_by_id: dict[int, str] = {}
        self.locked = False
        self.locked_robot_count = 0
        self.last_pairing: PairingResult | None = None
        self.ally_appearance = "no_aplica"  # compatibilidad con llamadas anteriores

        if config_path is not None:
            self._load_config(Path(config_path))
        if mode is not None:
            self.mode = str(mode).lower()
        if self.mode not in {"auto", "id", "none"}:
            raise ValueError("mode debe ser 'auto', 'id' o 'none'.")
        _ = ally_appearance  # ya no se usa: el color no decide los equipos.

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la configuración de equipos: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.mode = str(data.get("mode", self.mode)).lower()
        self.minimum_samples_per_robot = max(
            3,
            int(data.get("muestras_minimas_por_robot", self.minimum_samples_per_robot)),
        )
        self.maximum_samples_per_robot = max(
            self.minimum_samples_per_robot,
            int(data.get("muestras_maximas_por_robot", self.maximum_samples_per_robot)),
        )
        self.minimum_pairing_margin = float(
            data.get("margen_minimo_agrupamiento", self.minimum_pairing_margin)
        )
        self.minimum_detection_confidence = float(
            data.get("confianza_minima_muestra", self.minimum_detection_confidence)
        )
        self.swap_team_labels = bool(data.get("invertir_etiquetas", False))
        raw_map = data.get("id_map", data.get("equipos_por_id", {}))
        self.id_map = {
            int(key): str(value).lower()
            for key, value in raw_map.items()
            if str(value).lower() in {"aliado", "rival", "desconocido"}
        }

    def _sample_is_clean(self, detection: dict[str, Any]) -> bool:
        if bool(detection.get("predicted", False)):
            return False
        if float(detection.get("confidence", 0.0)) < self.minimum_detection_confidence:
            return False
        x1, y1, x2, y2 = map(float, detection.get("bbox_xyxy", [0, 0, 0, 0]))
        if max(0.0, x2 - x1) * max(0.0, y2 - y1) < self.minimum_box_area:
            return False
        return True

    def _collect_sample(self, frame: np.ndarray, detection: dict[str, Any]) -> None:
        if not self._sample_is_clean(detection):
            return
        tracking_id = int(detection["tracking_id"])
        if self.locked_robot_count >= 4:
            return
        if self.locked and tracking_id in self.team_by_id:
            return
        feature = extract_team_feature(frame, list(map(float, detection["bbox_xyxy"])))
        if feature is None:
            return
        samples = self.samples_by_id.setdefault(
            tracking_id,
            deque(maxlen=self.maximum_samples_per_robot),
        )

        # Se conservan también cuadros parecidos: la repetición temporal ayuda
        # a confirmar que la apariencia no fue un reflejo aislado.
        samples.append(feature)

    def _try_lock_pairing(self) -> None:
        if self.mode != "auto":
            return
        eligible = {
            tracking_id: list(samples)
            for tracking_id, samples in self.samples_by_id.items()
            if len(samples) >= self.minimum_samples_per_robot
        }
        if len(eligible) not in {3, 4}:
            return
        if len(eligible) <= self.locked_robot_count:
            return
        result = solve_two_team_pairing(eligible)
        if result is None:
            return
        self.last_pairing = result
        if result.margin < self.minimum_pairing_margin:
            return

        # El video puede descubrir dos familias, pero no puede saber cuál es
        # semánticamente "nuestro" equipo. Se usa una regla determinista: la
        # pareja que contiene el tracking_id menor se llama aliado.
        all_ids = sorted((*result.pair_a, *result.pair_b))
        smallest_id = all_ids[0]
        ally_pair = result.pair_a if smallest_id in result.pair_a else result.pair_b
        rival_pair = result.pair_b if ally_pair == result.pair_a else result.pair_a
        if self.swap_team_labels:
            ally_pair, rival_pair = rival_pair, ally_pair
        self.team_by_id = {
            **{tracking_id: "aliado" for tracking_id in ally_pair},
            **{tracking_id: "rival" for tracking_id in rival_pair},
        }
        self.locked = True
        self.locked_robot_count = len(eligible)

    def _team_number(self, tracking_id: int, team: str) -> int:
        if team not in {"aliado", "rival"}:
            return tracking_id + 1
        members = sorted(
            robot_id for robot_id, assigned_team in self.team_by_id.items()
            if assigned_team == team
        )
        return members.index(tracking_id) + 1 if tracking_id in members else 1

    def _visual_team_hint(self, feature: TeamFeature | None) -> tuple[str | None, float]:
        """Classify the current crop against frozen team exemplars.

        This keeps ally/rival labels attached to visual construction when an
        online tracking ID crosses another robot. It is deliberately used only
        after the global two-team pairing is locked.
        """
        if feature is None or not self.locked or len(set(self.team_by_id.values())) < 2:
            return None, 0.0
        scores: dict[str, float] = {}
        for team in ("aliado", "rival"):
            distances: list[float] = []
            for tracking_id, assigned_team in self.team_by_id.items():
                if assigned_team != team:
                    continue
                for sample in list(self.samples_by_id.get(tracking_id, ()))[:36]:
                    distances.append(team_feature_distance(feature, sample))
            if not distances:
                continue
            distances.sort()
            keep = max(2, min(len(distances), int(np.ceil(0.25 * len(distances)))))
            scores[team] = float(np.median(distances[:keep]))
        if len(scores) < 2:
            return None, 0.0
        ordered = sorted((cost, team) for team, cost in scores.items())
        best_cost, best_team = ordered[0]
        second_cost = ordered[1][0]
        margin = second_cost - best_cost
        confidence = float(np.clip(0.50 + 2.8 * margin - 0.35 * best_cost, 0.0, 1.0))
        if margin < 0.035 or best_cost > 0.48:
            return None, confidence
        return best_team, confidence

    def _decorate(
        self,
        detection: dict[str, Any],
        visual_team: str | None = None,
        visual_confidence: float = 0.0,
    ) -> dict[str, Any]:
        result = detection.copy()
        tracking_id = int(result["tracking_id"])
        if self.mode == "none":
            team = "desconocido"
            confidence = 0.0
        elif tracking_id in self.id_map:
            team = self.id_map[tracking_id]
            confidence = 1.0
        elif self.mode == "id":
            team = "desconocido"
            confidence = 0.0
        else:
            team = self.team_by_id.get(tracking_id, "desconocido")
            confidence = self.last_pairing.confidence if self.locked and self.last_pairing else 0.0
            if visual_team in {"aliado", "rival"} and visual_confidence >= 0.58:
                team = visual_team
                confidence = max(confidence, visual_confidence)

        team_number = self._team_number(tracking_id, team)
        if team == "aliado":
            display_name = f"Aliado {team_number}"
        elif team == "rival":
            display_name = f"Rival {team_number}"
        else:
            display_name = f"Robot {tracking_id + 1}"

        result.update(
            {
                "team": team,
                "team_number": team_number,
                "team_confidence": round(float(confidence), 5),
                "team_locked": bool(self.locked or tracking_id in self.id_map),
                "team_source": (
                    "apariencia_visual_bloqueada"
                    if visual_team in {"aliado", "rival"} and visual_confidence >= 0.58
                    else "id_bloqueado"
                    if self.locked or tracking_id in self.id_map
                    else "sin_confirmar"
                ),
                "display_name": display_name,
                "team_color": TEAM_COLORS_HEX[team],
            }
        )
        return result

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for detection in detections:
            if (
                str(detection.get("class_group", "")).lower() == "robot"
                and detection.get("tracking_id") is not None
            ):
                self._collect_sample(frame, detection)
        self._try_lock_pairing()

        output: list[dict[str, Any]] = []
        for detection in detections:
            if (
                str(detection.get("class_group", "")).lower() == "robot"
                and detection.get("tracking_id") is not None
            ):
                visual_team = None
                visual_confidence = 0.0
                if (
                    self.locked
                    and self._sample_is_clean(detection)
                    and (
                        bool(detection.get("association_ambiguous", False))
                        or bool(detection.get("identity_check_required", False))
                    )
                ):
                    current_feature = extract_team_feature(
                        frame,
                        list(map(float, detection["bbox_xyxy"])),
                    )
                    visual_team, visual_confidence = self._visual_team_hint(current_feature)
                decorated = self._decorate(
                    detection,
                    visual_team=visual_team,
                    visual_confidence=visual_confidence,
                )
                decorated["team_visual_confidence"] = round(float(visual_confidence), 5)
                output.append(decorated)
            else:
                output.append(detection.copy())
        return output

    def get_debug_state(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "locked_robot_count": self.locked_robot_count,
            "invertir_etiquetas": self.swap_team_labels,
            "team_by_id": {str(key): value for key, value in self.team_by_id.items()},
            "samples_by_id": {
                str(key): len(value) for key, value in self.samples_by_id.items()
            },
            "pairing": (
                {
                    "pair_a": list(self.last_pairing.pair_a),
                    "pair_b": list(self.last_pairing.pair_b),
                    "cost": round(self.last_pairing.cost, 6),
                    "second_best_cost": round(self.last_pairing.second_best_cost, 6),
                    "margin": round(self.last_pairing.margin, 6),
                    "confidence": round(self.last_pairing.confidence, 6),
                    "pair_distances": self.last_pairing.pair_distances,
                }
                if self.last_pairing is not None
                else None
            ),
        }

    def backfill_jsonl(self, path: str | Path) -> None:
        """Aplica la asignación final a todos los cuadros del JSONL.

        El video de previsualización ya renderizado no se modifica, pero los
        eventos, trayectorias y replay posteriores sí reciben equipos coherentes
        desde el primer cuadro.
        """
        if not self.team_by_id:
            return
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with path.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["detections"] = [
                    self._decorate(detection)
                    if (
                        str(detection.get("class_group", "")).lower() == "robot"
                        and detection.get("tracking_id") is not None
                    )
                    else detection
                    for detection in record.get("detections", [])
                ]
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
