from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.I_field_geometry.calibration import (
    FEATURE_ORDER,
    LINE_COLORS,
    LINE_LABELS_ES,
    FieldCalibration,
    create_calibration_from_points,
    line_from_two_points,
)
from src.I_field_geometry.field_template import build_template_points


class _WizardState:
    def __init__(self) -> None:
        self.selected_index = 0
        self.pending: list[tuple[int, int]] = []
        self.segments: dict[str, list[tuple[int, int]]] = {}
        self.history: list[tuple[str, list[tuple[int, int]] | None]] = []

    @property
    def selected(self) -> str:
        return FEATURE_ORDER[self.selected_index]

    def cycle(self, delta: int) -> None:
        self.selected_index = (self.selected_index + delta) % len(FEATURE_ORDER)
        self.pending.clear()

    def add_point(self, point: tuple[int, int]) -> None:
        self.pending.append(point)
        if len(self.pending) == 2:
            name = self.selected
            previous = self.segments.get(name)
            self.history.append((name, previous.copy() if previous else None))
            self.segments[name] = self.pending.copy()
            self.pending.clear()

    def undo(self) -> None:
        if self.pending:
            self.pending.pop()
            return
        if not self.history:
            return
        name, previous = self.history.pop()
        if previous is None:
            self.segments.pop(name, None)
        else:
            self.segments[name] = previous

    def delete_selected(self) -> None:
        self.pending.clear()
        name = self.selected
        if name in self.segments:
            previous = self.segments[name].copy()
            self.history.append((name, previous))
            self.segments.pop(name, None)

    def reset(self) -> None:
        self.pending.clear()
        self.segments.clear()
        self.history.clear()

    def points_by_line(self) -> dict[str, list[tuple[float, float]]]:
        return {
            name: [tuple(map(float, point)) for point in points]
            for name, points in self.segments.items()
        }


def _extended_segment(line: np.ndarray, width: int, height: int):
    a, b, c = map(float, line)
    candidates: list[tuple[float, float]] = []
    if abs(b) > 1e-9:
        for x in (0.0, float(width - 1)):
            y = -(a * x + c) / b
            if -2 * height <= y <= 3 * height:
                candidates.append((x, y))
    if abs(a) > 1e-9:
        for y in (0.0, float(height - 1)):
            x = -(b * y + c) / a
            if -2 * width <= x <= 3 * width:
                candidates.append((x, y))
    if len(candidates) < 2:
        return None
    best = max(
        ((first, second) for i, first in enumerate(candidates) for second in candidates[i + 1 :]),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    return tuple(np.rint(best[0]).astype(int)), tuple(np.rint(best[1]).astype(int))


def _overlay_calibration(frame: np.ndarray, calibration: FieldCalibration) -> np.ndarray:
    canvas = frame.copy()
    if not calibration.is_complete:
        return canvas
    cv2.polylines(
        canvas,
        [np.rint(calibration.corners_image).astype(np.int32)],
        True,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    try:
        inverse = np.linalg.inv(calibration.homography_image_to_field)
    except (np.linalg.LinAlgError, TypeError):
        return canvas
    template = build_template_points(density=220)
    field_points = template.points.copy()
    field_points[:, 0] *= calibration.field_width
    field_points[:, 1] *= calibration.field_height
    projected = cv2.perspectiveTransform(
        field_points.reshape(1, -1, 2).astype(np.float32), inverse
    ).reshape(-1, 2)
    for index in range(1, len(projected)):
        if template.groups[index] != template.groups[index - 1]:
            continue
        if not np.isfinite(projected[index]).all() or not np.isfinite(projected[index - 1]).all():
            continue
        if np.linalg.norm(projected[index] - projected[index - 1]) > 0.16 * max(frame.shape[:2]):
            continue
        cv2.line(
            canvas,
            tuple(np.rint(projected[index - 1]).astype(int)),
            tuple(np.rint(projected[index]).astype(int)),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _draw_text_lines(image: np.ndarray, lines: list[str], y0: int = 28) -> None:
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (18, y0 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54 if index else 0.62,
            (255, 255, 255) if index else (0, 255, 255),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )


def calibrate_video_interactively(
    video_path: str | Path,
    output_path: str | Path,
    frame_index: int = 0,
    field_width: float = 100.0,
    field_height: float = 60.0,
) -> Path:
    """Flexible V8 field annotation with hard semantic anchors.

    The user labels only lines that are genuinely visible. A partial annotation
    is saved without fabricating missing corners; the automatic registrar uses
    it as semantic evidence later. Four outer boundaries still yield an exact
    assisted homography when they are truly visible.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_frame = int(np.clip(frame_index, 0, total_frames - 1))
    state = _WizardState()
    window = "FutBotMX V8 - anclas semanticas visibles"

    def read_frame(index: int) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, image = capture.read()
        if not success:
            raise RuntimeError(f"No se pudo leer el cuadro {index}.")
        return image

    frame = read_frame(current_frame)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_point((int(x), int(y)))

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(width, 1200), min(height, 880))
        cv2.setMouseCallback(window, on_mouse)
    except cv2.error as error:
        capture.release()
        raise RuntimeError(
            "OpenCV no pudo abrir una ventana. Ejecuta la calibración en una sesión de escritorio."
        ) from error

    accepted: FieldCalibration | None = None
    while True:
        candidate = None
        error_message = ""
        if state.segments:
            try:
                candidate = create_calibration_from_points(
                    state.points_by_line(),
                    frame_width=width,
                    frame_height=height,
                    field_width=field_width,
                    field_height=field_height,
                    source_frame_index=current_frame,
                )
            except ValueError as error:
                error_message = str(error)

        display = _overlay_calibration(frame, candidate) if candidate is not None else frame.copy()
        for name, points in state.segments.items():
            color = LINE_COLORS[name]
            first, second = points
            cv2.circle(display, first, 7, color, -1, cv2.LINE_AA)
            cv2.circle(display, second, 7, color, -1, cv2.LINE_AA)
            line = line_from_two_points(np.asarray(first), np.asarray(second))
            extended = _extended_segment(line, width, height)
            if extended is not None:
                cv2.line(display, extended[0], extended[1], color, 3, cv2.LINE_AA)
            midpoint = tuple(np.rint((np.asarray(first) + np.asarray(second)) * 0.5).astype(int))
            cv2.putText(display, LINE_LABELS_ES[name], midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)

        for point in state.pending:
            cv2.circle(display, point, 8, LINE_COLORS[state.selected], -1, cv2.LINE_AA)

        panel_height = 166
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), cv2.FILLED)
        display = cv2.addWeighted(overlay, 0.72, display, 0.28, 0)
        selected_label = LINE_LABELS_ES[state.selected]
        if candidate is not None and candidate.has_global_registration:
            mode_text = "GLOBAL: anclas suficientes y consistentes"
        elif candidate is not None and candidate.has_local_registration:
            mode_text = "LOCAL: orientacion disponible; posicion global desconocida"
        else:
            mode_text = "EVIDENCIA: guardada sin inventar homografia"
        lines = [
            f"Cuadro {current_frame}/{total_frames - 1} | seleccionada: {selected_label}",
            (
                f"Marcadas: {len(state.segments)} | "
                f"transversales {candidate.transverse_count if candidate else 0} | "
                f"longitudinales {candidate.longitudinal_count if candidate else 0} | {mode_text}"
            ),
            "1-7 seleccionar | Q/E cambiar | clic x2 añadir/reemplazar | X borrar seleccionada",
            "U deshacer | R reiniciar | A/D +/-1 | J/L +/-30 | ENTER guardar | ESC cancelar",
        ]
        if error_message:
            lines.append(f"Aviso: {error_message[:110]}")
        _draw_text_lines(display, lines)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")) and key == 27:
            raise RuntimeError("Calibración cancelada por el usuario.")
        if ord("1") <= key <= ord("7"):
            state.selected_index = key - ord("1")
            state.pending.clear()
        elif key == ord("q"):
            state.cycle(-1)
        elif key == ord("e"):
            state.cycle(1)
        elif key in (ord("u"), 8):
            state.undo()
        elif key in (ord("x"), 127):
            state.delete_selected()
        elif key == ord("r"):
            state.reset()
        elif key in (13, 10) and candidate is not None:
            accepted = candidate
            break
        elif not state.pending and key in (ord("a"), ord("d"), ord("j"), ord("l")):
            delta = {ord("a"): -1, ord("d"): 1, ord("j"): -30, ord("l"): 30}[key]
            current_frame = int(np.clip(current_frame + delta, 0, total_frames - 1))
            frame = read_frame(current_frame)

    capture.release()
    cv2.destroyWindow(window)
    if accepted is None:
        raise RuntimeError("No se generó una calibración.")
    accepted.save(output_path)
    print(
        f"Calibración V8 guardada con {accepted.feature_count} ancla(s): "
        f"{'global' if accepted.has_global_registration else ('local' if accepted.has_local_registration else 'evidencia')}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Etiqueta solamente las líneas de cancha que sean visibles."
    )
    parser.add_argument("video_path")
    parser.add_argument("--output", default="field_calibration.json")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--field-width", type=float, default=100.0)
    parser.add_argument("--field-height", type=float, default=60.0)
    arguments = parser.parse_args()
    path = calibrate_video_interactively(
        video_path=arguments.video_path,
        output_path=arguments.output,
        frame_index=arguments.frame,
        field_width=arguments.field_width,
        field_height=arguments.field_height,
    )
    print(f"Calibración guardada: {path}")


if __name__ == "__main__":
    main()
