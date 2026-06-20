"""Descriptores visuales para descubrir equipos de robots sin referencias previas.

La señal principal es estructural (HOG, bordes y silueta). El color se conserva
como una señal secundaria para evitar que dos robots con la misma forma pero
acabados muy distintos sean indistinguibles, sin forzar que rojo y negro queden
en equipos separados.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TeamFeature:
    vector: np.ndarray
    structure: np.ndarray
    shape: np.ndarray
    topology: np.ndarray
    signature: np.ndarray
    color: np.ndarray


def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)

    # Incluye ligeramente la parte superior para conservar platillos, torres y
    # otras piezas que suelen distinguir la construcción de un equipo.
    x1 -= 0.05 * box_width
    x2 += 0.05 * box_width
    y1 -= 0.08 * box_height
    y2 += 0.03 * box_height

    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2, iy2


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _pool(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pooled = cv2.resize(image.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return pooled.reshape(-1)


def _green_mask(hsv: np.ndarray) -> np.ndarray:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (
        (hue >= 27)
        & (hue <= 108)
        & (saturation >= 35)
        & (value >= 25)
    )
    return green


def extract_team_feature(frame: np.ndarray, box: list[float]) -> TeamFeature | None:
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return None

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = _green_mask(hsv)
    foreground = (~green).astype(np.uint8)

    # Limpia pequeños huecos sin borrar estructuras delgadas.
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    if int(foreground.sum()) < 180:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)

    # Neutraliza el fondo para que la dirección de la cancha no domine HOG.
    foreground_gray = gray.copy()
    background_level = int(np.median(gray[foreground > 0])) if np.any(foreground) else 127
    foreground_gray[foreground == 0] = background_level
    hog_input = cv2.resize(foreground_gray, (64, 64), interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    ).compute(hog_input)
    hog_vector = _l2_normalize(hog if hog is not None else np.zeros(1764, np.float32))

    edges = cv2.Canny(foreground_gray, 45, 140)
    edges[foreground == 0] = 0
    edge_vector = _l2_normalize(_pool(edges / 255.0, 16, 16))
    silhouette_vector = _l2_normalize(_pool(foreground, 12, 12))

    # Proyecciones de ocupación: útiles para distinguir platillos y torres.
    vertical_profile = cv2.resize(
        foreground.mean(axis=1, keepdims=True).astype(np.float32),
        (1, 16),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)
    horizontal_profile = cv2.resize(
        foreground.mean(axis=0, keepdims=True).astype(np.float32),
        (16, 1),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)

    contours, _ = cv2.findContours(
        (foreground * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    largest_area = 0.0
    perimeter = 0.0
    solidity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(contour)) / (96.0 * 96.0)
        perimeter = float(cv2.arcLength(contour, True)) / (2.0 * (96.0 + 96.0))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(cv2.contourArea(contour)) / hull_area if hull_area > 1e-6 else 0.0

    top_occupancy = float(foreground[:32].mean())
    middle_occupancy = float(foreground[32:64].mean())
    bottom_occupancy = float(foreground[64:].mean())
    edge_density = float((edges > 0).mean())
    row_occupancy = foreground.mean(axis=1)
    column_occupancy = foreground.mean(axis=0)
    topology_scalars = np.array(
        [
            float(foreground.mean()),
            top_occupancy,
            middle_occupancy,
            bottom_occupancy,
            float(np.mean(row_occupancy > 0.25)),
            float(np.mean(row_occupancy > 0.50)),
            float(np.mean(row_occupancy > 0.70)),
            float(np.mean(row_occupancy[:48] > 0.60)),
            float(np.mean(row_occupancy[48:] > 0.60)),
            float(np.percentile(row_occupancy, 50)),
            float(np.percentile(row_occupancy, 80)),
            float(np.percentile(row_occupancy, 95)),
            float(np.percentile(column_occupancy, 50)),
            float(np.percentile(column_occupancy, 80)),
            float(np.percentile(column_occupancy, 95)),
        ],
        dtype=np.float32,
    )
    signature_vector = np.clip(topology_scalars, 0.0, 1.5).astype(np.float32)
    topology_vector = _l2_normalize(
        np.concatenate([topology_scalars, vertical_profile, horizontal_profile])
    )
    shape_scalars = np.array(
        [
            float((x2 - x1) / max(1, y2 - y1)),
            edge_density,
            largest_area,
            perimeter,
            solidity,
        ],
        dtype=np.float32,
    )
    shape_vector = _l2_normalize(shape_scalars)

    # Color deliberadamente con poco peso. LAB es más estable ante iluminación;
    # HSV conserva tonos dominantes sin hacer que rojo y negro decidan el equipo.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mask = (foreground * 255).astype(np.uint8)
    color_parts: list[np.ndarray] = []
    for image, channel, bins, upper in (
        (lab, 0, 8, 256),
        (lab, 1, 8, 256),
        (lab, 2, 8, 256),
        (hsv, 0, 12, 180),
        (hsv, 1, 8, 256),
    ):
        hist = cv2.calcHist([image], [channel], mask, [bins], [0, upper]).reshape(-1)
        color_parts.append(hist.astype(np.float32))
    color_vector = _l2_normalize(np.concatenate(color_parts))

    structure_vector = _l2_normalize(
        np.concatenate(
            [
                np.sqrt(0.62) * hog_vector,
                np.sqrt(0.23) * edge_vector,
                np.sqrt(0.15) * silhouette_vector,
            ]
        )
    )
    # La topología recibe mucho peso: una pareja con platillos anchos debe
    # permanecer junta aunque la iluminación o el color cambien.
    combined = _l2_normalize(
        np.concatenate(
            [
                np.sqrt(0.38) * structure_vector,
                np.sqrt(0.12) * shape_vector,
                np.sqrt(0.45) * topology_vector,
                np.sqrt(0.05) * color_vector,
            ]
        )
    )
    return TeamFeature(
        vector=combined,
        structure=structure_vector,
        shape=shape_vector,
        topology=topology_vector,
        signature=signature_vector,
        color=color_vector,
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(1.0 - similarity)


def team_feature_distance(a: TeamFeature, b: TeamFeature) -> float:
    """Distancia con prioridad explícita en construcción y topología."""
    signature_distance = float(np.mean(np.abs(a.signature - b.signature)))
    return float(
        0.18 * cosine_distance(a.structure, b.structure)
        + 0.08 * cosine_distance(a.shape, b.shape)
        + 0.18 * cosine_distance(a.topology, b.topology)
        + 0.52 * signature_distance
        + 0.04 * cosine_distance(a.color, b.color)
    )


def extract_team_feature_views(
    frame: np.ndarray,
    box: list[float],
    rotations: tuple[int, ...] = (0, 1, 2, 3),
) -> list[TeamFeature]:
    """Extrae varias orientaciones del mismo robot.

    Los robots giran sobre la cancha, por lo que comparar solamente el recorte
    vertical produce distancias falsas. Esta función conserva cuatro vistas
    rotadas y permite comparar usando la mejor orientación compatible.
    """
    if frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return []

    # Se coloca el recorte en un lienzo cuadrado para que rotar no deforme la
    # construcción. El fondo verde sintético es eliminado por el extractor.
    side = max(crop.shape[:2])
    canvas = np.full((side, side, 3), (70, 170, 90), dtype=np.uint8)
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop

    views: list[TeamFeature] = []
    for turns in rotations:
        rotated = np.ascontiguousarray(np.rot90(canvas, int(turns) % 4))
        feature = extract_team_feature(
            rotated,
            [0.0, 0.0, float(rotated.shape[1]), float(rotated.shape[0])],
        )
        if feature is not None:
            views.append(feature)
    return views


def rotation_invariant_distance(
    views_a: list[TeamFeature],
    views_b: list[TeamFeature],
    *,
    identity: bool = False,
) -> float:
    """Distancia mínima entre orientaciones.

    Para identidad física el color recibe más peso, porque ayuda a distinguir
    al robot rojo del negro. Para descubrir parejas de equipo el color casi no
    pesa, porque ambos pueden compartir construcción aunque cambie el acabado.
    """
    if not views_a or not views_b:
        return 1.0
    distance_function = identity_feature_distance if identity else team_feature_distance
    return float(
        min(distance_function(feature_a, feature_b) for feature_a in views_a for feature_b in views_b)
    )


def identity_feature_distance(a: TeamFeature, b: TeamFeature) -> float:
    """Distancia para reconocer al robot físico, no solamente a su equipo."""
    signature_distance = float(np.mean(np.abs(a.signature - b.signature)))
    return float(
        0.24 * cosine_distance(a.structure, b.structure)
        + 0.10 * cosine_distance(a.shape, b.shape)
        + 0.15 * cosine_distance(a.topology, b.topology)
        + 0.21 * signature_distance
        + 0.30 * cosine_distance(a.color, b.color)
    )
