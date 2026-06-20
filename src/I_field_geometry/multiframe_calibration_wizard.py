from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.I_field_geometry.calibration import LINE_COLORS, LINE_LABELS_ES

MULTIFRAME_FEATURE_ORDER = ("near", "far", "left", "right", "center")
from src.I_field_geometry.field_template import build_template_points
from src.I_field_geometry.multiframe_calibration import (
    MultiframeLineObservation,
    build_multiframe_calibration,
    observations_to_reference_segments,
    precompute_video_registrations,
    transform_segment,
)


@dataclass
class _HistoryItem:
    name: str
    observation: MultiframeLineObservation


class _State:
    def __init__(self) -> None:
        self.selected_index = 0
        self.pending: list[tuple[int, int]] = []
        self.observations: list[MultiframeLineObservation] = []
        self.history: list[_HistoryItem] = []

    @property
    def selected(self) -> str:
        return MULTIFRAME_FEATURE_ORDER[self.selected_index]

    def cycle(self, delta: int) -> None:
        self.selected_index = (self.selected_index + delta) % len(MULTIFRAME_FEATURE_ORDER)
        self.pending.clear()

    def add(self, frame_index: int, segment_frame: np.ndarray, current_to_reference: np.ndarray) -> None:
        segment_reference = transform_segment(segment_frame, current_to_reference)
        observation = MultiframeLineObservation(
            name=self.selected,
            frame_index=int(frame_index),
            segment_frame=np.asarray(segment_frame, dtype=np.float32).reshape(2, 2),
            segment_reference=segment_reference,
        )
        self.observations.append(observation)
        self.history.append(_HistoryItem(self.selected, observation))
        self.pending.clear()

    def undo(self) -> None:
        self.pending.clear()
        if not self.history:
            return
        item = self.history.pop()
        for index in range(len(self.observations) - 1, -1, -1):
            if self.observations[index] is item.observation:
                self.observations.pop(index)
                break

    def delete_selected(self) -> None:
        self.pending.clear()
        selected = self.selected
        self.observations = [item for item in self.observations if item.name != selected]
        self.history = [item for item in self.history if item.name != selected]

    def reset(self) -> None:
        self.pending.clear()
        self.observations.clear()
        self.history.clear()


def _draw_template(
    image: np.ndarray,
    homography_reference_to_field: np.ndarray,
    current_to_reference: np.ndarray,
    field_width: float,
    field_height: float,
) -> None:
    image_to_field = np.asarray(homography_reference_to_field, dtype=np.float64) @ np.asarray(
        current_to_reference, dtype=np.float64
    )
    try:
        field_to_image = np.linalg.inv(image_to_field)
    except np.linalg.LinAlgError:
        return
    template = build_template_points(density=220)
    field_points = template.points.astype(np.float32).copy()
    field_points[:, 0] *= float(field_width)
    field_points[:, 1] *= float(field_height)
    projected = cv2.perspectiveTransform(
        field_points.reshape(1, -1, 2), field_to_image
    ).reshape(-1, 2)
    for index in range(1, len(projected)):
        if template.groups[index] != template.groups[index - 1]:
            continue
        first, second = projected[index - 1], projected[index]
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            continue
        if float(np.linalg.norm(second - first)) > 0.20 * max(image.shape[:2]):
            continue
        cv2.line(
            image,
            tuple(np.rint(first).astype(int)),
            tuple(np.rint(second).astype(int)),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def calibrate_video_multiframe_interactively(
    video_path: str | Path,
    output_path: str | Path,
    field_width: float = 100.0,
    field_height: float = 60.0,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    frame_indices, matrices, _fps, width, height = precompute_video_registrations(video_path)
    total_keyframes = len(matrices)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    def read_frame(index: int) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer el cuadro {index}.")
        return frame

    state = _State()
    current_position = 0
    current_frame = frame_indices[current_position]
    frame = read_frame(current_frame)
    window = "FutBotMX V10 - calibracion multicuadro"

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state.pending.append((int(x), int(y)))
        if len(state.pending) == 2:
            state.add(
                current_frame,
                np.asarray(state.pending, dtype=np.float32),
                matrices[current_position],
            )

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(width, 1200), min(height, 900))
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as error:
        capture.release()
        raise RuntimeError(
            "OpenCV no pudo abrir una ventana. Ejecuta la calibración en una sesión de escritorio."
        ) from error

    accepted = None
    while True:
        display = frame.copy()
        calibration = None
        error_message = ""
        try:
            if state.observations:
                calibration = build_multiframe_calibration(
                    state.observations,
                    width,
                    height,
                    field_width,
                    field_height,
                    source_frame_index=0,
                )
        except ValueError as error:
            error_message = str(error)

        try:
            reference_segments = observations_to_reference_segments(state.observations)
        except ValueError:
            reference_segments = {}
        reference_to_current = None
        try:
            reference_to_current = np.linalg.inv(matrices[current_position])
        except np.linalg.LinAlgError:
            pass
        if reference_to_current is not None:
            for name, segment_reference in reference_segments.items():
                segment_current = transform_segment(segment_reference, reference_to_current)
                color = LINE_COLORS[name]
                p0, p1 = [tuple(np.rint(point).astype(int)) for point in segment_current]
                cv2.line(display, p0, p1, color, 3, cv2.LINE_AA)
                midpoint = tuple(np.rint(np.mean(segment_current, axis=0)).astype(int))
                count = sum(item.name == name for item in state.observations)
                cv2.putText(
                    display,
                    f"{LINE_LABELS_ES[name]} x{count}",
                    midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if calibration is not None and calibration.is_complete:
            _draw_template(
                display,
                calibration.homography_image_to_field,
                matrices[current_position],
                calibration.field_width,
                calibration.field_height,
            )

        for point in state.pending:
            cv2.circle(display, point, 8, LINE_COLORS[state.selected], -1, cv2.LINE_AA)

        counts = {name: sum(item.name == name for item in state.observations) for name in MULTIFRAME_FEATURE_ORDER}
        complete = bool(calibration is not None and calibration.is_complete)
        panel_height = 190
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), cv2.FILLED)
        display = cv2.addWeighted(overlay, 0.74, display, 0.26, 0)
        messages = [
            f"V10 MULTICUADRO | cuadro {current_frame} | keyframe {current_position + 1}/{total_keyframes} | {LINE_LABELS_ES[state.selected]}",
            f"Observaciones {len(state.observations)} | caracteristicas {sum(value > 0 for value in counts.values())}/5 | global {'LISTA' if complete else 'PENDIENTE'}",
            "Necesitas 2 transversales rectas (gol/centro) + left y right; pueden estar en cuadros distintos.",
            "1-5 seleccionar | clic x2 agregar | A/D +/-1 keyframe | J/L +/-3 | Z/C +/-15",
            "Q/E cambiar | U deshacer | X borrar seleccionada | R reiniciar | ENTER guardar | ESC cancelar",
        ]
        if error_message:
            messages.append(f"Aviso: {error_message[:120]}")
        for index, text in enumerate(messages):
            cv2.putText(
                display,
                text,
                (18, 28 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52 if index else 0.62,
                (0, 255, 255) if index == 0 else (255, 255, 255),
                2 if index == 0 else 1,
                cv2.LINE_AA,
            )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            raise RuntimeError("Calibración multicuadro cancelada por el usuario.")
        if ord("1") <= key <= ord("5"):
            state.selected_index = key - ord("1")
            state.pending.clear()
        elif key in (ord("q"), ord("Q")):
            state.cycle(-1)
        elif key in (ord("e"), ord("E")):
            state.cycle(1)
        elif key in (ord("u"), ord("U")):
            state.undo()
        elif key in (ord("x"), ord("X")):
            state.delete_selected()
        elif key in (ord("r"), ord("R")):
            state.reset()
        elif key in (ord("a"), ord("A"), ord("d"), ord("D"), ord("j"), ord("J"), ord("l"), ord("L"), ord("z"), ord("Z"), ord("c"), ord("C")):
            delta = {
                ord("a"): -1, ord("A"): -1, ord("d"): 1, ord("D"): 1,
                ord("j"): -3, ord("J"): -3, ord("l"): 3, ord("L"): 3,
                ord("z"): -15, ord("Z"): -15, ord("c"): 15, ord("C"): 15,
            }[key]
            current_position = int(np.clip(current_position + delta, 0, total_keyframes - 1))
            current_frame = frame_indices[current_position]
            frame = read_frame(current_frame)
            state.pending.clear()
        elif key in (10, 13):
            if complete:
                accepted = calibration
                break

    capture.release()
    cv2.destroyWindow(window)
    if accepted is None or not accepted.is_complete:
        raise RuntimeError("No se obtuvo una calibración global multicuadro.")
    accepted.save(output_path)
    print(
        f"Calibración V10 multicuadro guardada: {output_path} "
        f"({accepted.transverse_count} transversales, {accepted.longitudinal_count} longitudinales)",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración global V10 usando líneas visibles en varios cuadros.")
    parser.add_argument("video")
    parser.add_argument("--output", default="field_calibration.json")
    parser.add_argument("--field-width", type=float, default=100.0)
    parser.add_argument("--field-height", type=float, default=60.0)
    args = parser.parse_args()
    calibrate_video_multiframe_interactively(
        args.video,
        args.output,
        field_width=args.field_width,
        field_height=args.field_height,
    )


if __name__ == "__main__":
    main()
