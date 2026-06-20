from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import cv2
import numpy as np

# Canonical normalized feature lines. x runs near->far, y runs left->right.
CANONICAL_LINES_NORMALIZED: dict[str, np.ndarray] = {
    "near": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "far": np.array([1.0, 0.0, -1.0], dtype=np.float64),
    "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, 1.0, -1.0], dtype=np.float64),
    "center": np.array([1.0, 0.0, -0.5], dtype=np.float64),
    "near_area": np.array([1.0, 0.0, -0.18], dtype=np.float64),
    "far_area": np.array([1.0, 0.0, -0.82], dtype=np.float64),
}

TRANSVERSE_FEATURES = frozenset({"near", "far", "center", "near_area", "far_area"})
LONGITUDINAL_FEATURES = frozenset({"left", "right"})

CANONICAL_SEGMENTS_NORMALIZED: dict[str, np.ndarray] = {
    "near": np.float32([[0.0, 0.0], [0.0, 1.0]]),
    "far": np.float32([[1.0, 0.0], [1.0, 1.0]]),
    "left": np.float32([[0.0, 0.0], [1.0, 0.0]]),
    "right": np.float32([[0.0, 1.0], [1.0, 1.0]]),
    "center": np.float32([[0.5, 0.0], [0.5, 1.0]]),
    "near_area": np.float32([[0.18, 0.22], [0.18, 0.78]]),
    "far_area": np.float32([[0.82, 0.22], [0.82, 0.78]]),
}


@dataclass(frozen=True)
class FeatureAnchorScore:
    name: str
    score: float
    perpendicular_error_px: float
    angle_error_deg: float
    canonical_error: float
    along_span: float
    hard_pass: bool


def normalize_line(line: np.ndarray) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64).reshape(3)
    norm = float(np.hypot(line[0], line[1]))
    if norm < 1e-10:
        raise ValueError("Recta degenerada")
    return line / norm


def line_from_segment(segment: np.ndarray) -> np.ndarray:
    points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    first = np.array([points[0, 0], points[0, 1], 1.0], dtype=np.float64)
    second = np.array([points[1, 0], points[1, 1], 1.0], dtype=np.float64)
    return normalize_line(np.cross(first, second))


def _angle_between_segments(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(2, 2)
    b = np.asarray(second, dtype=np.float64).reshape(2, 2)
    va = a[1] - a[0]
    vb = b[1] - b[0]
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 90.0
    va /= na
    vb /= nb
    cosine = float(np.clip(abs(np.dot(va, vb)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def project_segment(segment: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(segment, dtype=np.float32).reshape(1, 2, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(2, 2)


def score_feature_anchor(
    name: str,
    observed_segment: np.ndarray,
    field_to_image: np.ndarray,
    frame_diagonal: float,
) -> FeatureAnchorScore:
    canonical_segment = CANONICAL_SEGMENTS_NORMALIZED[name]
    projected_segment = project_segment(canonical_segment, field_to_image)
    projected_line = line_from_segment(projected_segment)
    observed = np.asarray(observed_segment, dtype=np.float64).reshape(2, 2)
    distances = np.abs(observed @ projected_line[:2] + projected_line[2])
    perpendicular_error = float(np.mean(distances))
    angle_error = _angle_between_segments(observed, projected_segment)

    try:
        image_to_field = np.linalg.inv(np.asarray(field_to_image, dtype=np.float64))
        observed_field = project_segment(observed, image_to_field)
    except np.linalg.LinAlgError:
        observed_field = np.full((2, 2), np.nan, dtype=np.float64)

    if name in TRANSVERSE_FEATURES:
        expected = float(-CANONICAL_LINES_NORMALIZED[name][2])
        canonical_error = float(np.mean(np.abs(observed_field[:, 0] - expected)))
        along = observed_field[:, 1]
    else:
        expected = float(-CANONICAL_LINES_NORMALIZED[name][2])
        canonical_error = float(np.mean(np.abs(observed_field[:, 1] - expected)))
        along = observed_field[:, 0]
    along_span = float(abs(along[1] - along[0])) if np.isfinite(along).all() else 0.0
    along_inside = bool(np.isfinite(along).all() and np.max(along) >= -0.20 and np.min(along) <= 1.20)

    distance_scale = max(3.0, 0.0065 * float(frame_diagonal))
    distance_score = float(np.exp(-0.5 * (perpendicular_error / distance_scale) ** 2))
    angle_score = float(np.exp(-0.5 * (angle_error / 5.5) ** 2))
    canonical_score = float(np.exp(-0.5 * (canonical_error / 0.025) ** 2)) if np.isfinite(canonical_error) else 0.0
    span_score = float(np.clip(along_span / 0.12, 0.0, 1.0))
    score = 0.34 * distance_score + 0.28 * angle_score + 0.30 * canonical_score + 0.08 * span_score
    hard_pass = bool(
        perpendicular_error <= max(6.0, 0.011 * frame_diagonal)
        and angle_error <= 9.0
        and canonical_error <= 0.050
        and along_inside
        and along_span >= 0.035
    )
    return FeatureAnchorScore(
        name=name,
        score=float(np.clip(score, 0.0, 1.0)),
        perpendicular_error_px=perpendicular_error,
        angle_error_deg=angle_error,
        canonical_error=canonical_error,
        along_span=along_span,
        hard_pass=hard_pass,
    )


def score_manual_anchors(
    semantic_segments: dict[str, np.ndarray] | None,
    field_to_image: np.ndarray,
    frame_diagonal: float,
) -> list[FeatureAnchorScore]:
    scores: list[FeatureAnchorScore] = []
    for name, segment in (semantic_segments or {}).items():
        if name not in CANONICAL_SEGMENTS_NORMALIZED:
            continue
        scores.append(score_feature_anchor(name, segment, field_to_image, frame_diagonal))
    return scores


def semantic_family_counts(names: Iterable[str]) -> tuple[int, int]:
    names = set(names)
    return len(names & TRANSVERSE_FEATURES), len(names & LONGITUDINAL_FEATURES)


def has_global_line_support(names: Iterable[str]) -> bool:
    transverse, longitudinal = semantic_family_counts(names)
    return transverse >= 2 and longitudinal >= 2




def _normalized_dlt_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    """Deterministic normalized DLT for small line-grid correspondences.

    ``cv2.findHomography(method=0)`` can take an unexpectedly long path for
    nearly degenerate four-point grids.  Registration evaluates thousands of
    such hypotheses, so use a bounded 8x9 SVD and reject ill-conditioned grids
    explicitly.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(destination, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        return None

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        center = np.mean(points, axis=0)
        centered = points - center
        mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
        if not np.isfinite(mean_distance) or mean_distance < 1e-7:
            return None
        scale = np.sqrt(2.0) / mean_distance
        transform = np.array(
            [[scale, 0.0, -scale * center[0]],
             [0.0, scale, -scale * center[1]],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        normalized = (transform @ homogeneous.T).T[:, :2]
        return normalized, transform

    normalized_src = normalize(src)
    normalized_dst = normalize(dst)
    if normalized_src is None or normalized_dst is None:
        return None
    src_n, source_transform = normalized_src
    dst_n, destination_transform = normalized_dst

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    try:
        _u, singular_values, vh = np.linalg.svd(design, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    if len(singular_values) < 8 or singular_values[-1] < 1e-9:
        return None
    if singular_values[0] / singular_values[-1] > 2.0e7:
        return None

    normalized_h = vh[-1].reshape(3, 3)
    try:
        homography = np.linalg.inv(destination_transform) @ normalized_h @ source_transform
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(homography).all() or abs(float(homography[2, 2])) < 1e-10:
        return None
    homography /= homography[2, 2]

    projected = cv2.perspectiveTransform(src.astype(np.float32).reshape(1, -1, 2), homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    median_error = float(np.median(np.linalg.norm(projected - dst, axis=1)))
    return homography if median_error <= 0.015 else None

def homography_from_semantic_lines(
    semantic_lines: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (image_to_field, field_to_image) from any 2x2 line grid.

    Intersections of every transverse/longitudinal feature pair create point
    correspondences. This avoids inventing invisible outer corners and permits
    combinations such as center+near_area with left+right.
    """
    transverse = [name for name in semantic_lines if name in TRANSVERSE_FEATURES]
    longitudinal = [name for name in semantic_lines if name in LONGITUDINAL_FEATURES]
    if len(transverse) < 2 or len(longitudinal) < 2:
        return None

    image_points: list[np.ndarray] = []
    field_points: list[np.ndarray] = []
    for tx, ly in product(transverse, longitudinal):
        image_point_h = np.cross(semantic_lines[tx], semantic_lines[ly])
        if abs(float(image_point_h[2])) < 1e-10:
            continue
        image_point = image_point_h[:2] / image_point_h[2]
        x = float(-CANONICAL_LINES_NORMALIZED[tx][2])
        y = float(-CANONICAL_LINES_NORMALIZED[ly][2])
        if np.isfinite(image_point).all():
            image_points.append(image_point.astype(np.float32))
            field_points.append(np.array([x, y], dtype=np.float32))
    if len(image_points) < 4:
        return None
    image_array = np.asarray(image_points, dtype=np.float32)
    field_array = np.asarray(field_points, dtype=np.float32)
    if abs(float(cv2.contourArea(cv2.convexHull(image_array)))) < 1.0:
        return None
    image_to_field = _normalized_dlt_homography(image_array, field_array)
    if image_to_field is None:
        return None
    try:
        field_to_image = np.linalg.inv(image_to_field)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(field_to_image).all() or abs(float(field_to_image[2, 2])) < 1e-10:
        return None
    return image_to_field, field_to_image / field_to_image[2, 2]


def local_rectification_from_segments(
    semantic_segments: dict[str, np.ndarray] | None,
    fallback_segments: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray | None, str, int]:
    """Build an image->local affine projective transform without global position.

    The transform only orthogonalizes visible line directions. It deliberately
    contains no canonical translation or field scale.
    """
    x_dirs: list[np.ndarray] = []  # field x direction: left/right features
    y_dirs: list[np.ndarray] = []  # field y direction: transverse features
    origins: list[np.ndarray] = []
    for name, raw in (semantic_segments or {}).items():
        segment = np.asarray(raw, dtype=np.float64).reshape(2, 2)
        vector = segment[1] - segment[0]
        norm = float(np.linalg.norm(vector))
        if norm < 5.0:
            continue
        vector /= norm
        origins.append(np.mean(segment, axis=0))
        if name in LONGITUDINAL_FEATURES:
            x_dirs.append(vector)
        elif name in TRANSVERSE_FEATURES:
            y_dirs.append(vector)

    source = "manual"
    if not x_dirs and not y_dirs and fallback_segments is not None:
        source = "automatico"
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for raw in fallback_segments:
            segment = np.asarray(raw, dtype=np.float64).reshape(2, 2)
            vector = segment[1] - segment[0]
            length = float(np.linalg.norm(vector))
            if length < 20.0:
                continue
            vector /= length
            candidates.append((length, vector, np.mean(segment, axis=0)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            y_dirs.append(candidates[0][1])
            origins.append(candidates[0][2])
        if len(candidates) >= 2:
            first = candidates[0][1]
            best = max(candidates[1:12], key=lambda item: abs(first[0] * item[1][1] - first[1] * item[1][0]))
            if abs(float(first[0] * best[1][1] - first[1] * best[1][0])) > 0.22:
                x_dirs.append(best[1])
                origins.append(best[2])

    if not x_dirs and not y_dirs:
        return None, "none", 0

    def mean_direction(vectors: list[np.ndarray]) -> np.ndarray | None:
        if not vectors:
            return None
        reference = vectors[0]
        aligned = [v if np.dot(v, reference) >= 0 else -v for v in vectors]
        result = np.mean(aligned, axis=0)
        norm = float(np.linalg.norm(result))
        return result / norm if norm > 1e-9 else None

    dx = mean_direction(x_dirs)
    dy = mean_direction(y_dirs)
    if dx is None and dy is not None:
        dx = np.array([dy[1], -dy[0]], dtype=np.float64)
    if dy is None and dx is not None:
        dy = np.array([-dx[1], dx[0]], dtype=np.float64)
    if dx is None or dy is None:
        return None, "none", 0
    if abs(float(dx[0] * dy[1] - dx[1] * dy[0])) < 0.12:
        dy = np.array([-dx[1], dx[0]], dtype=np.float64)

    basis = np.column_stack([dx, dy])
    try:
        linear = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        return None, "none", 0
    origin = np.mean(origins, axis=0) if origins else np.zeros(2, dtype=np.float64)
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = linear
    transform[:2, 2] = -linear @ origin
    level = "local_2d" if x_dirs and y_dirs else "local_1d"
    return transform, f"{level}_{source}", len(x_dirs) + len(y_dirs)
