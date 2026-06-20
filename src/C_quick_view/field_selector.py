"""Selección de la cancha principal entre las cajas entregadas por YOLO.

FutBotMX V5 conserva la inferencia original del modelo para la clase cancha
(`imgsz=640`, confianza 0.25) y únicamente decide qué candidato representa la
superficie principal. La selección favorece cobertura geométrica sin ignorar
por completo la confianza de YOLO.
"""

from __future__ import annotations

from typing import Any


def bbox_area(box: list[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_main_field(
    detections: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Devuelve la caja principal y un diagnóstico de todos los candidatos.

    La cobertura recibe 80 % del peso y la confianza 20 %. Esto evita que una
    caja interior pequeña con confianza ligeramente mayor desplace a la caja
    grande que cubre la cancha completa.
    """
    frame_area = max(1.0, float(frame_width * frame_height))
    diagnostics: list[dict[str, Any]] = []

    for index, detection in enumerate(detections):
        box = list(map(float, detection.get("bbox_xyxy", [0, 0, 0, 0])))
        area = bbox_area(box)
        coverage = area / frame_area
        confidence = float(detection.get("confidence", 0.0))

        # Penaliza únicamente cajas geométricamente absurdas. Una cancha
        # parcial sigue siendo válida cuando la cámara no muestra toda la mesa.
        valid_geometry = area > 0.0 and coverage <= 1.02
        score = (0.80 * coverage + 0.20 * confidence) if valid_geometry else -1.0
        diagnostics.append(
            {
                "candidate_index": index,
                "bbox_xyxy": [round(value, 2) for value in box],
                "confidence": round(confidence, 6),
                "coverage": round(coverage, 6),
                "selection_score": round(score, 6),
                "valid_geometry": valid_geometry,
            }
        )

    valid = [item for item in diagnostics if item["valid_geometry"]]
    if not valid:
        return None, diagnostics

    selected_diag = max(valid, key=lambda item: item["selection_score"])
    selected = detections[int(selected_diag["candidate_index"])].copy()
    selected["field_source"] = "yolo_cobertura_confianza"
    selected["field_selection_score"] = selected_diag["selection_score"]
    selected["field_coverage"] = selected_diag["coverage"]
    selected["field_candidates_count"] = len(detections)
    selected["smoothed"] = False
    return selected, diagnostics
