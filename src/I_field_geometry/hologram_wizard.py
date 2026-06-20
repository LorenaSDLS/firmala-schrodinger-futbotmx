from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.I_field_geometry.field_spec import FieldSpec
from src.I_field_geometry.field_template import FieldTemplateConfig, build_template_points
from src.I_field_geometry.hologram_calibration import HologramCalibration, HologramKeyframe
from src.I_field_geometry.multiframe_calibration import precompute_video_registrations


CANVAS_SIZE = (1800, 1120)
PANEL_HEIGHT = 244
VIEWPORT_MARGIN = 22
MIN_VIDEO_ZOOM = 0.18
MAX_VIDEO_ZOOM = 1.20
FIELD_CORNER_LABELS = ("AMARILLA / y=0", "AZUL / y=0", "AZUL / y=max", "AMARILLA / y=max")
CORNER_COLORS = ((0, 220, 255), (255, 170, 30), (255, 170, 30), (0, 220, 255))


def _ascii_ui(text: object) -> str:
    """Return text that OpenCV's Hershey fonts can render reliably.

    ``cv2.putText`` does not support UTF-8.  Accents and symbols such as
    ``±`` used to appear as ``??`` on Windows.  Keeping this conversion in one
    place also prevents future UI labels from reintroducing the problem.
    """

    value = str(text)
    replacements = {
        "×": "x",
        "±": "+/-",
        "→": "->",
        "←": "<-",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for source, destination in replacements.items():
        value = value.replace(source, destination)
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _put_text(
    image: np.ndarray,
    text: object,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        _ascii_ui(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).reshape(-1)
    return np.float32(
        [
            values[np.argmin(sums)],
            values[np.argmin(differences)],
            values[np.argmax(sums)],
            values[np.argmax(differences)],
        ]
    )


def initial_hologram_corners(frame: np.ndarray) -> np.ndarray:
    """Produce an editable first guess; it is never treated as proof."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([28, 28, 28]), np.array([105, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 0.05 * width * height:
            rectangle = cv2.minAreaRect(contour)
            corners = _order_quad(cv2.boxPoints(rectangle))
            center = np.mean(corners, axis=0)
            corners = center + 1.08 * (corners - center)
            return corners.astype(np.float32)
    margin_x = 0.12 * width
    margin_y = 0.18 * height
    return np.float32(
        [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y],
        ]
    )


def _project_template(corners: np.ndarray, spec: FieldSpec) -> tuple[np.ndarray, np.ndarray]:
    field_corners = np.float32(
        [[0.0, 0.0], [spec.surface_length_cm, 0.0], [spec.surface_length_cm, spec.surface_width_cm], [0.0, spec.surface_width_cm]]
    )
    matrix = cv2.getPerspectiveTransform(field_corners, np.asarray(corners, dtype=np.float32))
    template = build_template_points(FieldTemplateConfig.from_spec(spec), density=330)
    metric = template.points.astype(np.float32).copy()
    metric[:, 0] *= spec.surface_length_cm
    metric[:, 1] *= spec.surface_width_cm
    projected = cv2.perspectiveTransform(metric.reshape(1, -1, 2), matrix).reshape(-1, 2)
    return projected, template.groups


@dataclass
class _CanvasTransform:
    scale: float
    offset_x: float
    offset_y: float

    def image_to_canvas(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        return values * self.scale + np.float32([self.offset_x, self.offset_y])

    def canvas_to_image(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        return (values - np.float32([self.offset_x, self.offset_y])) / self.scale


class HologramEditor:
    def __init__(
        self,
        video_path: Path,
        frame_indices: list[int],
        matrices_current_to_reference: list[np.ndarray],
        fps: float,
        width: int,
        height: int,
        spec: FieldSpec,
    ) -> None:
        self.video_path = video_path
        self.frame_indices = frame_indices
        self.matrices = matrices_current_to_reference
        self.fps = fps
        self.width = width
        self.height = height
        self.spec = spec
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        self.position = 0
        self.frame = self._read_frame(self.frame_indices[0])
        self.keyframes: dict[int, np.ndarray] = {}
        self.current_corners = initial_hologram_corners(self.frame)
        self.drag_corner: int | None = None
        self.drag_all = False
        self.last_mouse_image: np.ndarray | None = None
        # Zoom is relative to the largest size that fits below the help panel.
        # A low minimum is intentional: in difficult videos the projected
        # field can extend far beyond the image and the user needs white space.
        self.video_zoom = 0.58
        self.blur_enabled = True
        self.hologram_opacity = 0.38
        self.window = "FutBotMX V11.2 - holograma asistido"
        self.transform = self._canvas_transform()
        self.button_regions: dict[str, tuple[int, int, int, int]] = {}
        self.finish_requested = False
        self.cancel_requested = False

    def close(self) -> None:
        self.capture.release()
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass

    def _read_frame(self, index: int) -> np.ndarray:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer el cuadro {index}.")
        return frame

    def _canvas_transform(self) -> _CanvasTransform:
        canvas_width, canvas_height = CANVAS_SIZE
        viewport_width = canvas_width - 2 * VIEWPORT_MARGIN
        viewport_height = canvas_height - PANEL_HEIGHT - 2 * VIEWPORT_MARGIN
        fit_scale = min(viewport_width / self.width, viewport_height / self.height)
        scale = fit_scale * self.video_zoom
        return _CanvasTransform(
            scale=float(scale),
            offset_x=float(0.5 * (canvas_width - self.width * scale)),
            offset_y=float(
                PANEL_HEIGHT
                + VIEWPORT_MARGIN
                + 0.5 * (viewport_height - self.height * scale)
            ),
        )

    def zoom_video(self, factor: float) -> None:
        self.video_zoom = float(
            np.clip(self.video_zoom * float(factor), MIN_VIDEO_ZOOM, MAX_VIDEO_ZOOM)
        )
        self.transform = self._canvas_transform()

    def reset_video_zoom(self) -> None:
        self.video_zoom = 0.58
        self.transform = self._canvas_transform()

    def fit_wide_view(self) -> None:
        """Leave generous white space for a field projected outside the frame."""
        self.video_zoom = 0.42
        self.transform = self._canvas_transform()

    def _dispatch_button(self, action: str) -> None:
        actions = {
            "zoom_out": lambda: self.zoom_video(1.0 / 1.16),
            "zoom_in": lambda: self.zoom_video(1.16),
            "fit": self.fit_wide_view,
            "reset": self.reset_video_zoom,
            "prev3": lambda: self.navigate(-3),
            "prev": lambda: self.navigate(-1),
            "next": lambda: self.navigate(1),
            "next3": lambda: self.navigate(3),
            "save": self.save_current,
            "delete": self.remove_current,
            "opacity_down": lambda: setattr(self, "hologram_opacity", max(0.10, self.hologram_opacity - 0.05)),
            "opacity_up": lambda: setattr(self, "hologram_opacity", min(0.82, self.hologram_opacity + 0.05)),
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "blur": lambda: setattr(self, "blur_enabled", not self.blur_enabled),
            "finish": lambda: setattr(self, "finish_requested", bool(self.keyframes)),
        }
        callback = actions.get(action)
        if callback is not None:
            callback()

    def _button_at(self, x: int, y: int) -> str | None:
        for action, (x1, y1, x2, y2) in self.button_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def _predict_from_nearest_anchor(self, position: int) -> np.ndarray | None:
        if not self.keyframes:
            return None
        frame_index = self.frame_indices[position]
        nearest_frame = min(self.keyframes, key=lambda item: abs(item - frame_index))
        anchor_position = min(
            range(len(self.frame_indices)), key=lambda item: abs(self.frame_indices[item] - nearest_frame)
        )
        current_to_reference = np.asarray(self.matrices[position], dtype=np.float64)
        anchor_to_reference = np.asarray(self.matrices[anchor_position], dtype=np.float64)
        field_corners = np.float32(
            [[0.0, 0.0], [self.spec.surface_length_cm, 0.0], [self.spec.surface_length_cm, self.spec.surface_width_cm], [0.0, self.spec.surface_width_cm]]
        )
        anchor_h = cv2.getPerspectiveTransform(field_corners, self.keyframes[nearest_frame].astype(np.float32))
        try:
            prediction_h = np.linalg.inv(current_to_reference) @ anchor_to_reference @ anchor_h
        except np.linalg.LinAlgError:
            return None
        projected = cv2.perspectiveTransform(field_corners.reshape(1, -1, 2), prediction_h).reshape(4, 2)
        return projected.astype(np.float32) if np.isfinite(projected).all() else None

    def navigate(self, delta: int) -> None:
        self.position = int(np.clip(self.position + delta, 0, len(self.frame_indices) - 1))
        frame_index = self.frame_indices[self.position]
        self.frame = self._read_frame(frame_index)
        if frame_index in self.keyframes:
            self.current_corners = self.keyframes[frame_index].copy()
        else:
            prediction = self._predict_from_nearest_anchor(self.position)
            self.current_corners = prediction if prediction is not None else initial_hologram_corners(self.frame)
        self.drag_corner = None
        self.drag_all = False

    def save_current(self) -> None:
        self.keyframes[self.frame_indices[self.position]] = self.current_corners.copy()

    def remove_current(self) -> None:
        self.keyframes.pop(self.frame_indices[self.position], None)

    def flip_x(self) -> None:
        self.current_corners = self.current_corners[[1, 0, 3, 2]].copy()

    def flip_y(self) -> None:
        self.current_corners = self.current_corners[[3, 2, 1, 0]].copy()

    def scale_hologram(self, factor: float) -> None:
        center = np.mean(self.current_corners, axis=0)
        self.current_corners = center + float(factor) * (self.current_corners - center)

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and y < PANEL_HEIGHT:
            action = self._button_at(x, y)
            if action is not None:
                self._dispatch_button(action)
            return
        if y < PANEL_HEIGHT:
            return
        point_image = self.transform.canvas_to_image(np.float32([[x, y]]))[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = np.linalg.norm(self.current_corners - point_image, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] * self.transform.scale <= 34.0:
                self.drag_corner = nearest
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drag_all = True
            self.last_mouse_image = point_image.copy()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_corner is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                self.current_corners[self.drag_corner] = point_image
            elif self.drag_all and flags & cv2.EVENT_FLAG_RBUTTON and self.last_mouse_image is not None:
                delta = point_image - self.last_mouse_image
                self.current_corners += delta
                self.last_mouse_image = point_image.copy()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_corner = None
        elif event == cv2.EVENT_RBUTTONUP:
            self.drag_all = False
            self.last_mouse_image = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            if hasattr(cv2, "getMouseWheelDelta"):
                wheel_delta = int(cv2.getMouseWheelDelta(flags))
            else:
                wheel_delta = 1 if flags > 0 else -1
            direction = 1 if wheel_delta > 0 else -1
            # Normal wheel zooms the video/canvas. Holding SHIFT preserves the
            # old behaviour and changes only the hologram size.
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.scale_hologram(1.04 if direction > 0 else 1.0 / 1.04)
            else:
                self.zoom_video(1.08 if direction > 0 else 1.0 / 1.08)

    def render(self) -> np.ndarray:
        canvas_width, canvas_height = CANVAS_SIZE
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        frame = self.frame
        if self.blur_enabled:
            frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.8, sigmaY=1.8)
        frame = cv2.addWeighted(frame, 0.82, np.full_like(frame, 255), 0.18, 0.0)
        display_size = (int(round(self.width * self.transform.scale)), int(round(self.height * self.transform.scale)))
        resized = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(self.transform.offset_x)), int(round(self.transform.offset_y))
        x1, y1 = x0 + resized.shape[1], y0 + resized.shape[0]
        destination_x0, destination_y0 = max(0, x0), max(PANEL_HEIGHT, y0)
        destination_x1, destination_y1 = min(canvas_width, x1), min(canvas_height, y1)
        if destination_x1 > destination_x0 and destination_y1 > destination_y0:
            source_x0 = destination_x0 - x0
            source_y0 = destination_y0 - y0
            source_x1 = source_x0 + destination_x1 - destination_x0
            source_y1 = source_y0 + destination_y1 - destination_y0
            canvas[destination_y0:destination_y1, destination_x0:destination_x1] = resized[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (185, 185, 185), 2)

        overlay = canvas.copy()
        corners_canvas = self.transform.image_to_canvas(self.current_corners).astype(np.int32)
        cv2.fillConvexPoly(overlay, corners_canvas, (70, 235, 215), cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, self.hologram_opacity, canvas, 1.0 - self.hologram_opacity, 0.0)

        projected, groups = _project_template(self.current_corners, self.spec)
        projected_canvas = self.transform.image_to_canvas(projected)
        group_colors = {0: (245, 245, 245), 1: (255, 255, 255), 2: (230, 255, 255), 3: (230, 255, 255), 4: (255, 220, 60)}
        for index in range(1, len(projected_canvas)):
            if groups[index] != groups[index - 1]:
                continue
            first, second = projected_canvas[index - 1], projected_canvas[index]
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                continue
            if np.linalg.norm(second - first) > 0.22 * max(CANVAS_SIZE):
                continue
            cv2.line(
                canvas,
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(second).astype(int)),
                group_colors.get(int(groups[index]), (255, 255, 255)),
                3 if int(groups[index]) == 0 else 2,
                cv2.LINE_AA,
            )

        # Goal orientation is explicit: x=0 yellow, x=243 blue.
        yellow = tuple(np.rint(corners_canvas[[0, 3]].mean(axis=0)).astype(int))
        blue = tuple(np.rint(corners_canvas[[1, 2]].mean(axis=0)).astype(int))
        _put_text(canvas, "PORTERIA AMARILLA (x=0)", yellow, 0.55, (0, 180, 240), 2)
        _put_text(canvas, "PORTERIA AZUL (x=243 cm)", blue, 0.55, (240, 140, 20), 2)
        for index, point in enumerate(corners_canvas):
            color = CORNER_COLORS[index]
            cv2.circle(canvas, tuple(point), 11, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(point), 15, (20, 20, 20), 2, cv2.LINE_AA)
            label_origin = tuple(point + np.array([14, -12]))
            _put_text(
                canvas,
                f"{index + 1} {FIELD_CORNER_LABELS[index]}",
                label_origin,
                0.48,
                (20, 20, 20),
                2,
            )

        frame_index = self.frame_indices[self.position]
        is_anchor = frame_index in self.keyframes
        panel = np.full((PANEL_HEIGHT, canvas_width, 3), 246, dtype=np.uint8)
        _put_text(
            panel,
            f"V11.2 HOLOGRAMA | cuadro {frame_index} ({frame_index / max(self.fps, 1e-6):.2f}s) | vista {self.position + 1}/{len(self.frame_indices)} | anclas {len(self.keyframes)} | zoom {self.video_zoom * 100:.0f}%",
            (22, 31),
            0.66,
            (25, 25, 25),
            2,
        )
        _put_text(
            panel,
            "Arrastra una esquina con clic izquierdo. Mueve toda la cancha con clic derecho. La plantilla puede quedar fuera del video.",
            (22, 61),
            0.52,
            (65, 65, 65),
            1,
        )

        self.button_regions = {}
        def button(action: str, label: str, x: int, y: int, w: int, *, primary: bool = False, active: bool = False) -> None:
            h = 42
            self.button_regions[action] = (x, y, x + w, y + h)
            if primary:
                fill, border, text_color = (55, 165, 75), (30, 120, 45), (255, 255, 255)
            elif active:
                fill, border, text_color = (215, 244, 224), (65, 155, 90), (20, 80, 35)
            else:
                fill, border, text_color = (255, 255, 255), (165, 165, 165), (35, 35, 35)
            cv2.rectangle(panel, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
            cv2.rectangle(panel, (x, y), (x + w, y + h), border, 2, cv2.LINE_AA)
            label_ascii = _ascii_ui(label)
            (tw, th), _ = cv2.getTextSize(label_ascii, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            _put_text(panel, label_ascii, (x + max(8, (w - tw) // 2), y + (h + th) // 2 - 2), 0.50, text_color, 1)

        row1_y = 78
        x = 22
        for action, label, width_button in (
            ("zoom_out", "- ALEJAR", 120),
            ("zoom_in", "+ ACERCAR", 120),
            ("fit", "VISTA AMPLIA", 160),
            ("reset", "REINICIAR VISTA", 170),
            ("opacity_down", "- OPACIDAD", 135),
            ("opacity_up", "+ OPACIDAD", 135),
            ("blur", "DESENFOQUE", 145),
        ):
            button(action, label, x, row1_y, width_button, active=(action == "blur" and self.blur_enabled))
            x += width_button + 10

        row2_y = 132
        x = 22
        for action, label, width_button in (
            ("prev3", "<<< 3 VISTAS", 145),
            ("prev", "< ANTERIOR", 140),
            ("next", "SIGUIENTE >", 140),
            ("next3", "3 VISTAS >>>", 145),
            ("flip_x", "INVERTIR X", 135),
            ("flip_y", "INVERTIR Y", 135),
            ("delete", "BORRAR ANCLA", 160),
        ):
            button(action, label, x, row2_y, width_button)
            x += width_button + 10

        row3_y = 186
        button("save", "GUARDAR / CORREGIR ANCLA", 22, row3_y, 285, active=is_anchor)
        _put_text(
            panel,
            f"Estado: {'ANCLA GUARDADA' if is_anchor else 'prediccion editable'} | cancha 243 x 182 cm | atajos de teclado siguen activos",
            (330, row3_y + 27),
            0.52,
            (35, 35, 35),
            1,
        )
        button("finish", "FINALIZAR Y PROCESAR", canvas_width - 285, row3_y, 255, primary=bool(self.keyframes))
        canvas[:PANEL_HEIGHT] = panel
        return canvas

    def run(self) -> tuple[HologramKeyframe, ...]:
        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, CANVAS_SIZE[0], CANVAS_SIZE[1])
            cv2.setMouseCallback(self.window, self.on_mouse)
        except cv2.error as error:
            raise RuntimeError("OpenCV no pudo abrir el editor holográfico. Ejecuta en una sesión de escritorio.") from error
        while True:
            self.transform = self._canvas_transform()
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(20) & 0xFF
            if self.finish_requested and self.keyframes:
                break
            if key == 27:
                raise RuntimeError("Calibración holográfica cancelada por el usuario.")
            if key in (10, 13):
                if self.keyframes:
                    break
            elif key in (ord("k"), ord("K")):
                self.save_current()
            elif key in (ord("u"), ord("U")):
                self.remove_current()
            elif key in (ord("x"), ord("X")):
                self.flip_x()
            elif key in (ord("y"), ord("Y")):
                self.flip_y()
            elif key in (ord("b"), ord("B")):
                self.blur_enabled = not self.blur_enabled
            elif key in (ord("q"), ord("Q"), ord("1"), ord("[")):
                self.zoom_video(1.0 / 1.10)
            elif key in (ord("e"), ord("E"), ord("2"), ord("]")):
                self.zoom_video(1.10)
            elif key == ord("0"):
                self.reset_video_zoom()
            elif key in (ord("-"), ord("_")):
                self.hologram_opacity = max(0.10, self.hologram_opacity - 0.04)
            elif key in (ord("+"), ord("=")):
                self.hologram_opacity = min(0.78, self.hologram_opacity + 0.04)
            elif key in (ord("a"), ord("A"), ord("d"), ord("D"), ord("j"), ord("J"), ord("l"), ord("L"), ord("z"), ord("Z"), ord("c"), ord("C")):
                delta = {
                    ord("a"): -1, ord("A"): -1, ord("d"): 1, ord("D"): 1,
                    ord("j"): -3, ord("J"): -3, ord("l"): 3, ord("L"): 3,
                    ord("z"): -15, ord("Z"): -15, ord("c"): 15, ord("C"): 15,
                }[key]
                self.navigate(delta)
        return tuple(
            HologramKeyframe(frame_index=index, corners_image=corners, confidence=1.0)
            for index, corners in sorted(self.keyframes.items())
        )


def calibrate_video_hologram_interactively(
    video_path: str | Path,
    output_path: str | Path,
    field_spec_path: str | Path | None = None,
    sample_stride: int | None = None,
) -> Path:
    video_path = Path(video_path).expanduser().resolve()
    spec = FieldSpec.load(field_spec_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    stride = int(sample_stride or max(5, round(fps * 0.25)))
    print("Preparando keyframes visuales del editor holográfico...", flush=True)
    frame_indices, matrices, fps, width, height = precompute_video_registrations(
        video_path,
        processing_max_width=520,
        sample_stride=stride,
    )
    editor = HologramEditor(video_path, frame_indices, matrices, fps, width, height, spec)
    try:
        keyframes = editor.run()
    finally:
        editor.close()
    calibration = HologramCalibration(
        frame_width=width,
        frame_height=height,
        fps=fps,
        total_frames=total,
        field_spec=spec,
        keyframes=keyframes,
    )
    destination = calibration.save(output_path)
    print(f"Calibración holográfica V11 guardada: {destination} ({len(keyframes)} anclas)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Editor V11 de holograma métrico de cancha.")
    parser.add_argument("video")
    parser.add_argument("--output", default="field_hologram_calibration.json")
    parser.add_argument("--field-spec", default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    arguments = parser.parse_args()
    calibrate_video_hologram_interactively(
        arguments.video,
        arguments.output,
        field_spec_path=arguments.field_spec,
        sample_stride=arguments.sample_stride,
    )


if __name__ == "__main__":
    main()
