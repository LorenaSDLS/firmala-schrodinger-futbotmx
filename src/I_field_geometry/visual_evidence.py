from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualLineCandidate:
    line: np.ndarray
    segment: np.ndarray
    support: float
    length_ratio: float
    source: str


@dataclass(frozen=True)
class FieldVisualEvidence:
    marking_mask: np.ndarray
    marking_confidence: np.ndarray
    boundary_mask: np.ndarray
    rail_mask: np.ndarray
    search_region: np.ndarray
    marking_lines: list[VisualLineCandidate]
    boundary_lines: list[VisualLineCandidate]
    marking_pixel_fraction: float


def normalize_line(line: np.ndarray) -> np.ndarray:
    value = np.asarray(line, dtype=np.float64).reshape(3)
    norm = float(np.hypot(value[0], value[1]))
    if norm < 1e-10:
        raise ValueError("Recta degenerada")
    return value / norm


def line_from_segment(segment: np.ndarray) -> np.ndarray | None:
    points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    first = np.array([points[0, 0], points[0, 1], 1.0], dtype=np.float64)
    second = np.array([points[1, 0], points[1, 1], 1.0], dtype=np.float64)
    value = np.cross(first, second)
    if float(np.hypot(value[0], value[1])) < 1e-9:
        return None
    return normalize_line(value)


class AdaptiveFieldEvidenceExtractor:
    """Extract field paint and physical boundary evidence.

    V8 intersected its whiteness threshold with the green segmentation mask.
    This erased exactly the pixels that define many field markings because a
    semantic surface model often labels white paint as background. V9 searches
    in a narrow dilation around the surface and scores pixels by both absolute
    whiteness and local contrast against their neighbourhood.
    """

    def __init__(self, maximum_lines: int = 48) -> None:
        self.maximum_lines = max(12, int(maximum_lines))
        try:
            self._lsd = cv2.createLineSegmentDetector(
                cv2.LSD_REFINE_STD,
                scale=0.8,
                sigma_scale=0.6,
                quant=2.0,
                ang_th=22.5,
                log_eps=0.0,
                density_th=0.55,
                n_bins=1024,
            )
        except Exception:
            self._lsd = None

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(binary)
        largest = max(contours, key=cv2.contourArea)
        output = np.zeros_like(binary)
        cv2.drawContours(output, [largest], -1, 255, cv2.FILLED)
        return output

    @staticmethod
    def _erase_boxes(mask: np.ndarray, boxes: Iterable[list[float]] | None, pad_ratio: float = 0.16) -> None:
        height, width = mask.shape[:2]
        for raw in boxes or []:
            if len(raw) != 4:
                continue
            x1, y1, x2, y2 = map(float, raw)
            pad_x = max(5, int(round(pad_ratio * max(1.0, x2 - x1))))
            pad_y = max(5, int(round(pad_ratio * max(1.0, y2 - y1))))
            cv2.rectangle(
                mask,
                (max(0, int(np.floor(x1)) - pad_x), max(0, int(np.floor(y1)) - pad_y)),
                (min(width - 1, int(np.ceil(x2)) + pad_x), min(height - 1, int(np.ceil(y2)) + pad_y)),
                0,
                cv2.FILLED,
            )

    @staticmethod
    def _robust_percentile(values: np.ndarray, percentile: float, default: float) -> float:
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size < 32:
            return float(default)
        return float(np.percentile(finite, percentile))

    def extract(
        self,
        frame: np.ndarray,
        field_mask: np.ndarray,
        exclusion_boxes: Iterable[list[float]] | None = None,
    ) -> FieldVisualEvidence:
        height, width = frame.shape[:2]
        diagonal = float(np.hypot(width, height))
        field = self._largest_component(field_mask)

        # Search just beyond the semantic surface so paint classified as
        # background is retained, while spectators and room highlights remain
        # outside the allowed region.
        search_radius = max(7, int(round(0.018 * diagonal)))
        search_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * search_radius + 1, 2 * search_radius + 1)
        )
        search = cv2.dilate(field, search_kernel)
        self._erase_boxes(search, exclusion_boxes, pad_ratio=0.18)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        lightness = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)

        surface_values = lightness[field > 0]
        l65 = self._robust_percentile(surface_values, 65.0, 145.0)
        l82 = self._robust_percentile(surface_values, 82.0, 180.0)
        v65 = self._robust_percentile(value[field > 0], 65.0, 145.0)

        blur_size = max(9, int(round(0.020 * diagonal)) | 1)
        local_mean = cv2.GaussianBlur(lightness, (blur_size, blur_size), 0)
        local_contrast = lightness - local_mean
        chroma = np.sqrt((a_channel - 128.0) ** 2 + (b_channel - 128.0) ** 2)

        # Two complementary tests: absolute neutral brightness and a locally
        # bright ridge. The latter catches shaded or compressed white lines.
        absolute_white = (
            (saturation < 105.0)
            & (chroma < 42.0)
            & (lightness > max(138.0, l65 + 10.0))
            & (value > max(135.0, v65 + 5.0))
        )
        contrast_white = (
            (saturation < 145.0)
            & (chroma < 58.0)
            & (local_contrast > 10.0)
            & (lightness > max(118.0, l65 - 12.0))
        )
        very_bright = (
            (saturation < 150.0)
            & (lightness > max(190.0, l82 + 8.0))
            & (value > 185.0)
        )

        confidence = np.zeros((height, width), dtype=np.float32)
        confidence += np.clip((lightness - max(115.0, l65 - 12.0)) / 90.0, 0.0, 1.0) * 0.30
        confidence += np.clip((115.0 - saturation) / 115.0, 0.0, 1.0) * 0.23
        confidence += np.clip(local_contrast / 34.0, 0.0, 1.0) * 0.37
        confidence += np.clip((62.0 - chroma) / 62.0, 0.0, 1.0) * 0.10
        confidence *= (search > 0).astype(np.float32)

        marking = ((absolute_white | contrast_white | very_bright) & (search > 0)).astype(np.uint8) * 255
        self._erase_boxes(marking, exclusion_boxes, pad_ratio=0.22)
        marking = cv2.morphologyEx(marking, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        marking = cv2.morphologyEx(marking, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Remove tiny isolated highlights but preserve thin connected markings.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(marking, 8)
        filtered = np.zeros_like(marking)
        minimum_area = max(10, int(round(0.000015 * width * height)))
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            elongation = max(component_width, component_height) / max(1.0, min(component_width, component_height))
            if area >= minimum_area and (elongation >= 1.7 or area >= 4 * minimum_area):
                filtered[labels == label] = 255
        marking = filtered

        field_binary = (field > 0).astype(np.uint8)
        eroded = cv2.erode(field_binary, np.ones((7, 7), np.uint8))
        boundary = cv2.subtract(field_binary, eroded) * 255
        boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8))

        # A segmentation contour clipped by the camera frame is not a physical
        # field edge.  Rewarding it pulls invisible corners onto the image
        # border and corrupts coordinates.  V9 therefore removes a narrow
        # camera margin from geometric boundary evidence.
        camera_margin_x = max(5, int(round(0.012 * diagonal)))
        camera_margin_y = max(5, int(round(0.012 * diagonal)))
        boundary[:camera_margin_y, :] = 0
        boundary[-camera_margin_y:, :] = 0
        boundary[:, :camera_margin_x] = 0
        boundary[:, -camera_margin_x:] = 0

        outside_annulus = cv2.dilate(field_binary, np.ones((31, 31), np.uint8)) - field_binary
        rail = (
            (outside_annulus > 0)
            & ((value < 120.0) | ((value < 160.0) & (saturation < 80.0)))
        ).astype(np.uint8) * 255
        rail = cv2.morphologyEx(rail, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        rail[:camera_margin_y, :] = 0
        rail[-camera_margin_y:, :] = 0
        rail[:, :camera_margin_x] = 0
        rail[:, -camera_margin_x:] = 0
        self._erase_boxes(rail, exclusion_boxes, pad_ratio=0.15)

        marking_lines = self._extract_lines(
            marking,
            confidence,
            diagonal,
            source="paint",
            minimum_length_ratio=0.026,
        )
        boundary_support = np.maximum(
            (boundary > 0).astype(np.float32),
            0.75 * (rail > 0).astype(np.float32),
        )
        boundary_lines = self._extract_lines(
            cv2.max(boundary, rail),
            boundary_support,
            diagonal,
            source="boundary",
            minimum_length_ratio=0.042,
        )

        pixel_fraction = float(np.count_nonzero(marking)) / max(1.0, float(np.count_nonzero(search)))
        return FieldVisualEvidence(
            marking_mask=marking,
            marking_confidence=confidence,
            boundary_mask=boundary,
            rail_mask=rail,
            search_region=search,
            marking_lines=marking_lines,
            boundary_lines=boundary_lines,
            marking_pixel_fraction=pixel_fraction,
        )

    def _extract_lines(
        self,
        binary: np.ndarray,
        support_map: np.ndarray,
        diagonal: float,
        source: str,
        minimum_length_ratio: float,
    ) -> list[VisualLineCandidate]:
        edge = cv2.Canny(binary, 25, 100)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
        raw_segments: list[np.ndarray] = []
        if self._lsd is not None:
            detected = self._lsd.detect(edge)[0]
            if detected is not None:
                for item in detected:
                    x1, y1, x2, y2 = map(float, item.reshape(4))
                    raw_segments.append(np.array([[x1, y1], [x2, y2]], dtype=np.float64))
        if not raw_segments:
            lines = cv2.HoughLinesP(
                edge,
                1,
                np.pi / 720.0,
                threshold=max(15, int(0.014 * diagonal)),
                minLineLength=max(24, int(minimum_length_ratio * diagonal)),
                maxLineGap=max(12, int(0.030 * diagonal)),
            )
            if lines is not None:
                for item in lines[:, 0, :]:
                    raw_segments.append(
                        np.array([item[:2], item[2:]], dtype=np.float64)
                    )

        candidates: list[VisualLineCandidate] = []
        height, width = binary.shape[:2]
        for segment in raw_segments:
            length = float(np.linalg.norm(segment[1] - segment[0]))
            if length < minimum_length_ratio * diagonal:
                continue
            line = line_from_segment(segment)
            if line is None:
                continue
            samples = (
                np.linspace(0.0, 1.0, 64, dtype=np.float64)[:, None]
                * (segment[1] - segment[0])
                + segment[0]
            )
            xs = np.clip(np.round(samples[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.round(samples[:, 1]).astype(int), 0, height - 1)
            binary_support = float(np.mean(binary[ys, xs] > 0))
            confidence_support = float(np.mean(support_map[ys, xs]))
            support = float(
                np.clip(
                    (0.30 + 0.70 * min(1.0, length / (0.22 * diagonal)))
                    * (0.42 * binary_support + 0.58 * confidence_support),
                    0.0,
                    1.0,
                )
            )
            if support < 0.09:
                continue
            candidates.append(
                VisualLineCandidate(
                    line=line,
                    segment=segment,
                    support=support,
                    length_ratio=length / diagonal,
                    source=source,
                )
            )

        candidates.sort(
            key=lambda item: item.support * (0.45 + item.length_ratio), reverse=True
        )
        selected: list[VisualLineCandidate] = []
        angle_tolerance = np.deg2rad(4.5)
        distance_tolerance = max(10.0, 0.014 * diagonal)
        for candidate in candidates:
            midpoint = np.mean(candidate.segment, axis=0)
            direction = candidate.segment[1] - candidate.segment[0]
            angle = float(np.arctan2(direction[1], direction[0]))
            duplicate = False
            for existing in selected:
                other_midpoint = np.mean(existing.segment, axis=0)
                other_direction = existing.segment[1] - existing.segment[0]
                other_angle = float(np.arctan2(other_direction[1], other_direction[0]))
                delta = abs(np.arctan2(np.sin(angle - other_angle), np.cos(angle - other_angle)))
                delta = min(delta, abs(np.pi - delta))
                distance = min(
                    abs(float(existing.line[0] * midpoint[0] + existing.line[1] * midpoint[1] + existing.line[2])),
                    abs(float(candidate.line[0] * other_midpoint[0] + candidate.line[1] * other_midpoint[1] + candidate.line[2])),
                )
                if delta < angle_tolerance and distance < distance_tolerance:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(candidate)
            if len(selected) >= self.maximum_lines:
                break
        return selected
