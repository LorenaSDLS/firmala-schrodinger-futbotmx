import argparse
from pathlib import Path
import unicodedata

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arc, Rectangle
from tqdm.auto import tqdm

from src.F_simulation.mesa_replay_model import FutbotReplayModel
from src.I_field_geometry.field_spec import FieldSpec


EVENT_NAMES_ES = {
    "possession_change": "Cambio de posesión",
    "ball_out_of_field": "Balón fuera de la cancha",
    "robot_inactive_candidate": "Posible robot inactivo",
    "robot_collision_candidate": "Posible colisión",
    "ball_missing_candidate": "Balón no visible",
    "ball_recovered": "Balón recuperado",
    "robot_reactivated": "Robot reactivado",
    "robot_entered_penalty_area": "Entrada al área penal",
    "red_card_robot_removed": "Tarjeta roja / robot retirado",
    "robot_grabbed_by_referee": "Robot retirado por el árbitro",
    "ball_moved_by_referee": "Balón movido por el árbitro",
    "referee_intervention_candidate": "Posible intervención arbitral",
    "goal": "Gol",
}


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def describe_event(event: dict) -> str:
    timestamp = format_time(float(event.get("timestamp_seconds", 0.0)))
    event_type = str(event.get("event_type", "evento"))
    event_name = EVENT_NAMES_ES.get(event_type, event_type.replace("_", " ").title())
    description = str(event.get("description", ""))
    return f"{timestamp} | {event_name}\n{description}"



def _ascii_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return normalized.encode("ascii", "ignore").decode("ascii")


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    text = str(value).lstrip("#")
    if len(text) != 6:
        return (128, 128, 128)
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return (blue, green, red)


def _export_opencv_replay(
    model: FutbotReplayModel,
    output_path: Path,
    frames: list[int],
    output_fps: float,
    field_width: float,
    field_height: float,
    goal_depth: float,
    goal_width: float,
    field_spec: FieldSpec,
) -> str | None:
    """Fast renderer used by default; Matplotlib remains the fallback."""
    canvas_width, canvas_height = 1440, 810
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        writer.release()
        return None

    plot_left, plot_top = 58, 74
    plot_right, plot_bottom = 1068, 754
    event_left = 1090
    total_x = field_width + 2.0 * goal_depth

    def project(x_value: float, y_value: float) -> tuple[int, int]:
        px = plot_left + (float(x_value) + goal_depth) / max(total_x, 1e-9) * (plot_right - plot_left)
        py = plot_bottom - float(y_value) / max(field_height, 1e-9) * (plot_bottom - plot_top)
        return int(round(px)), int(round(py))

    model.reset()
    last_snapshot = {"agents": {}, "events": []}
    progress = tqdm(total=len(frames), desc="Renderizando repeticion rapida", unit="frame")
    try:
        for target_frame in frames:
            while model.current_frame <= target_frame and not model.is_done():
                last_snapshot = model.step()
            image = np.full((canvas_height, canvas_width, 3), (246, 246, 248), dtype=np.uint8)
            # Header and panels.
            cv2.rectangle(image, (20, 18), (canvas_width - 20, canvas_height - 18), (225, 225, 232), 2)
            cv2.putText(
                image,
                _ascii_text(f"FutBotMX - Mesa Replay - cuadro {target_frame}"),
                (42, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                (24, 24, 34),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(image, (plot_left, plot_top), (plot_right, plot_bottom), (64, 64, 64), 2)

            # Carpet plane.
            p00 = project(0.0, 0.0)
            p11 = project(field_width, field_height)
            field_x1, field_y_bottom = p00
            field_x2, field_y_top = p11
            cv2.rectangle(
                image,
                (field_x1, field_y_top),
                (field_x2, field_y_bottom),
                (58, 164, 105),
                -1,
            )
            cv2.rectangle(image, (field_x1, field_y_top), (field_x2, field_y_bottom), (245, 245, 245), 3)
            center_top = project(field_spec.center_line_x_cm, field_height)
            center_bottom = project(field_spec.center_line_x_cm, 0.0)
            cv2.line(image, center_top, center_bottom, (250, 250, 250), 2, cv2.LINE_AA)

            # D-shaped penalty areas.
            center_y = field_height / 2.0
            left_center = project(0.0, center_y)
            right_center = project(field_width, center_y)
            radius_x = max(1, abs(project(field_spec.penalty_area_depth_cm, center_y)[0] - left_center[0]))
            radius_y = max(1, abs(project(0.0, center_y + field_spec.penalty_area_width_cm / 2.0)[1] - left_center[1]))
            cv2.ellipse(image, left_center, (radius_x, radius_y), 0.0, -90.0, 90.0, (250, 250, 250), 2, cv2.LINE_AA)
            cv2.ellipse(image, right_center, (radius_x, radius_y), 0.0, 90.0, 270.0, (250, 250, 250), 2, cv2.LINE_AA)

            # Physical goals.
            goal_y1 = (field_height - goal_width) / 2.0
            goal_y2 = goal_y1 + goal_width
            yellow_a = project(-goal_depth, goal_y1)
            yellow_b = project(0.0, goal_y2)
            blue_a = project(field_width, goal_y1)
            blue_b = project(field_width + goal_depth, goal_y2)
            cv2.rectangle(image, (yellow_a[0], yellow_b[1]), (yellow_b[0], yellow_a[1]), (0, 215, 255), 3)
            cv2.rectangle(image, (blue_a[0], blue_b[1]), (blue_b[0], blue_a[1]), (230, 125, 25), 3)

            # Agents.
            for agent_id, state in last_snapshot.get("agents", {}).items():
                if not state or not bool(state.get("visible", True)):
                    continue
                x_value, y_value = state.get("x_field"), state.get("y_field")
                if x_value is None or y_value is None:
                    continue
                point = project(float(x_value), float(y_value))
                agent = model.agents[agent_id]
                color = _hex_to_bgr(agent.color)
                if agent_id == "ball":
                    cv2.circle(image, point, 10, color, -1, cv2.LINE_AA)
                    cv2.circle(image, point, 10, (30, 30, 30), 2, cv2.LINE_AA)
                    label = "Balon"
                else:
                    if agent.team == "rival":
                        cv2.rectangle(image, (point[0] - 12, point[1] - 12), (point[0] + 12, point[1] + 12), color, -1)
                        cv2.rectangle(image, (point[0] - 12, point[1] - 12), (point[0] + 12, point[1] + 12), (20, 20, 20), 2)
                    else:
                        cv2.circle(image, point, 14, color, -1, cv2.LINE_AA)
                        cv2.circle(image, point, 14, (20, 20, 20), 2, cv2.LINE_AA)
                    label = agent.display_name
                cv2.putText(image, _ascii_text(label), (point[0] + 14, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

            # Event side panel.
            cv2.rectangle(image, (event_left, plot_top), (canvas_width - 36, plot_bottom), (255, 255, 255), -1)
            cv2.rectangle(image, (event_left, plot_top), (canvas_width - 36, plot_bottom), (210, 210, 220), 2)
            cv2.putText(image, "EVENTOS", (event_left + 18, plot_top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 45), 2, cv2.LINE_AA)
            visible_events = [event for event in model.events if int(event.get("frame_index", 0)) <= target_frame][-8:]
            y_cursor = plot_top + 74
            for event in reversed(visible_events):
                title = EVENT_NAMES_ES.get(str(event.get("event_type", "")), str(event.get("event_type", "evento")))
                stamp = format_time(float(event.get("timestamp_seconds", 0.0)))
                cv2.putText(image, _ascii_text(f"{stamp}  {title}"), (event_left + 18, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 50), 1, cv2.LINE_AA)
                y_cursor += 42

            writer.write(image)
            progress.update(1)
    finally:
        progress.close()
        writer.release()
    return str(output_path)


def export_mesa_replay_video(
    json_path: str | Path,
    output_path: str | Path | None = None,
    fps: float | None = None,
    frame_stride: int = 1,
    field_width: float | None = None,
    field_height: float | None = None,
    ball_interpolation_max_seconds: float = 8.0,
    goal_depth: float | None = None,
    goal_width: float | None = None,
    field_spec_path: str | Path | None = None,
    fast_renderer: bool = True,
) -> str:
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.parent / "mesa_replay.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_spec = FieldSpec.load(field_spec_path)
    field_width = float(field_width) if field_width is not None else field_spec.surface_length_cm
    field_height = float(field_height) if field_height is not None else field_spec.surface_width_cm
    model = FutbotReplayModel(
        json_path=json_path,
        field_width=field_width,
        field_height=field_height,
        ball_interpolation_max_seconds=ball_interpolation_max_seconds,
    )
    resolved_goal_depth = (
        float(goal_depth) if goal_depth is not None else field_spec.goal_depth_cm
    )
    resolved_goal_width = (
        float(goal_width) if goal_width is not None else field_spec.goal_width_cm
    )
    frame_stride = max(1, int(frame_stride))
    source_fps = float(model.data.get("video", {}).get("fps") or 30.0)
    output_fps = float(fps) if fps is not None else source_fps / frame_stride
    output_fps = max(output_fps, 1.0)
    frames = list(range(0, model.max_frame + 1, frame_stride))

    if fast_renderer and output_path.suffix.lower() == ".mp4":
        fast_result = _export_opencv_replay(
            model=model,
            output_path=output_path,
            frames=frames,
            output_fps=output_fps,
            field_width=field_width,
            field_height=field_height,
            goal_depth=resolved_goal_depth,
            goal_width=resolved_goal_width,
            field_spec=field_spec,
        )
        if fast_result is not None:
            return fast_result

    fig, (ax, event_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [4, 1.4]},
    )
    last_snapshot = {"agents": {}, "events": []}

    def draw(frame_index: int):
        nonlocal last_snapshot
        while model.current_frame <= frame_index and not model.is_done():
            last_snapshot = model.step()
        snapshot = last_snapshot

        ax.clear()
        ax.set_xlim(-resolved_goal_depth, field_width + resolved_goal_depth)
        ax.set_ylim(0, field_height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#16a36a")
        ax.set_title(f"FutBotMX — repetición de mesa — cuadro {frame_index}")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.plot(
            [0, field_width, field_width, 0, 0],
            [0, 0, field_height, field_height, 0],
            color="white",
            linewidth=2,
        )
        ax.axvline(field_spec.center_line_x_cm, color="white", linewidth=1.7, alpha=0.85)

        # Áreas reales: semielipses de 25 x 80 cm, sin círculo central.
        penalty_height = float(field_spec.penalty_area_width_cm)
        penalty_depth = float(field_spec.penalty_area_depth_cm)
        penalty_y = field_height / 2.0
        ax.add_patch(
            Arc(
                (0.0, penalty_y),
                2.0 * penalty_depth,
                penalty_height,
                theta1=-90.0,
                theta2=90.0,
                color="white",
                linewidth=1.7,
            )
        )
        ax.add_patch(
            Arc(
                (field_width, penalty_y),
                2.0 * penalty_depth,
                penalty_height,
                theta1=90.0,
                theta2=270.0,
                color="white",
                linewidth=1.7,
            )
        )

        # Porterías físicas de 60 x 10 cm. Amarilla en x=0 y azul en x=max.
        goal_y = (field_height - resolved_goal_width) / 2.0
        near_goal = Rectangle(
            (-resolved_goal_depth, goal_y),
            resolved_goal_depth,
            resolved_goal_width,
            fill=True,
            facecolor="#FFD40033",
            edgecolor="#D9A600",
            linewidth=2.5,
            hatch="///",
        )
        far_goal = Rectangle(
            (field_width, goal_y),
            resolved_goal_depth,
            resolved_goal_width,
            fill=True,
            facecolor="#168CFF33",
            edgecolor="#0874D1",
            linewidth=2.5,
            hatch="///",
        )
        ax.add_patch(near_goal)
        ax.add_patch(far_goal)
        ax.text(
            -resolved_goal_depth * 0.5,
            goal_y + resolved_goal_width + 1.2,
            "PORTERÍA AMARILLA",
            ha="center",
            fontsize=7,
            fontweight="bold",
            color="#8A6900",
        )
        ax.text(
            field_width + resolved_goal_depth * 0.5,
            goal_y + resolved_goal_width + 1.2,
            "PORTERÍA AZUL",
            ha="center",
            fontsize=7,
            fontweight="bold",
            color="#045491",
        )

        for agent_id, state in snapshot["agents"].items():
            if not state or not state.get("visible", True):
                continue
            x = state.get("x_field")
            y = state.get("y_field")
            if x is None or y is None:
                continue

            agent = model.agents[agent_id]
            if agent_id == "ball":
                ax.scatter(x, y, s=90, color=agent.color, edgecolors="black", zorder=4)
                ax.text(x + 1, y + 1, "Balón", fontsize=8, color="black")
            else:
                marker = "o" if agent.team == "aliado" else "s"
                ax.scatter(
                    x,
                    y,
                    s=190,
                    color=agent.color,
                    edgecolors="black",
                    marker=marker,
                    zorder=3,
                )
                ax.text(x + 1, y + 1, agent.display_name, fontsize=8, color="black")

        frame_events = snapshot.get("events", [])
        if frame_events:
            event_text = "\n".join(
                f"- {EVENT_NAMES_ES.get(event['event_type'], event['event_type'])}"
                for event in frame_events[:4]
            )
            ax.text(
                1,
                field_height - 3,
                event_text,
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.8},
            )

        event_ax.clear()
        event_ax.set_facecolor("#f4f4f4")
        event_ax.axis("off")
        event_ax.set_title("Eventos", fontsize=12, fontweight="bold")
        visible_events = [
            event
            for event in model.events
            if int(event.get("frame_index", 0)) <= frame_index
        ]
        latest_events = visible_events[-10:]
        if not latest_events:
            event_ax.text(0.05, 0.95, "Sin eventos todavía", va="top", fontsize=9)
        else:
            y_position = 0.95
            for event in reversed(latest_events):
                event_ax.text(
                    0.05,
                    y_position,
                    describe_event(event),
                    va="top",
                    fontsize=7.5,
                    wrap=True,
                )
                y_position -= 0.12

    animation = FuncAnimation(
        fig,
        draw,
        frames=frames,
        interval=1000 / output_fps,
        repeat=False,
    )

    progress_bar = tqdm(total=len(frames), desc="Renderizando repetición", unit="frame")

    def update_progress(_current_frame: int, _total_frames: int) -> None:
        progress_bar.update(1)

    try:
        try:
            animation.save(
                str(output_path),
                fps=output_fps,
                progress_callback=update_progress,
            )
        except Exception as mp4_error:
            progress_bar.close()
            output_path = output_path.with_suffix(".gif")
            progress_bar = tqdm(
                total=len(frames),
                desc="Renderizando repetición GIF",
                unit="frame",
            )
            animation.save(
                str(output_path),
                writer=PillowWriter(fps=output_fps),
                progress_callback=update_progress,
            )
            print(f"No se pudo crear MP4 ({mp4_error}). Se generó un GIF.")
    finally:
        progress_bar.close()
        plt.close(fig)

    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una repetición tipo Mesa desde futbot_unity_mesa.json."
    )
    parser.add_argument("json_path", help="Ruta a futbot_unity_mesa.json.")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida. Por defecto: mesa_replay.mp4 junto al JSON.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS del replay. Por defecto conserva el tiempo real del video.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Procesa cada N cuadros. Usa 2 o 3 para pruebas rápidas.",
    )
    parser.add_argument("--goal-depth", type=float, default=None)
    parser.add_argument("--goal-width", type=float, default=None)
    parser.add_argument(
        "--ball-interpolation-max-seconds",
        "--interpolacion-balon-max-segundos",
        dest="ball_interpolation_max_seconds",
        type=float,
        default=8.0,
        help="Duración máxima de un hueco del balón que se rellena en el replay.",
    )
    args = parser.parse_args()
    output_path = export_mesa_replay_video(
        json_path=args.json_path,
        output_path=args.output,
        fps=args.fps,
        frame_stride=args.frame_stride,
        ball_interpolation_max_seconds=args.ball_interpolation_max_seconds,
        goal_depth=args.goal_depth,
        goal_width=args.goal_width,
    )
    print(f"Animación generada: {output_path}")


if __name__ == "__main__":
    main()
