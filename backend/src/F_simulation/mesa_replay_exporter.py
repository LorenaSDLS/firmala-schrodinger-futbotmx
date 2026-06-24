import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from src.F_simulation.mesa_replay_model import FutbotReplayModel


EVENT_NAMES_ES = {
    "possession_change": "Cambio de posesión",
    "ball_out_of_field": "Balón fuera de la cancha",
    "robot_inactive_candidate": "Posible robot inactivo",
    "robot_collision_candidate": "Posible colisión",
    "ball_missing_candidate": "Balón no visible",
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


def export_mesa_replay_video(
    json_path: str | Path,
    output_path: str | Path | None = None,
    fps: float | None = None,
    frame_stride: int = 1,
    field_width: float = 100.0,
    field_height: float = 60.0,
    ball_interpolation_max_seconds: float = 8.0,
    goal_depth: float | None = None,
    goal_width: float | None = None,
) -> str:
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.parent / "mesa_replay.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = FutbotReplayModel(
        json_path=json_path,
        field_width=field_width,
        field_height=field_height,
        ball_interpolation_max_seconds=ball_interpolation_max_seconds,
    )
    resolved_goal_depth = (
        float(goal_depth) if goal_depth is not None else max(5.0, field_width * 0.09)
    )
    resolved_goal_width = (
        float(goal_width) if goal_width is not None else field_height * 0.34
    )
    frame_stride = max(1, int(frame_stride))
    source_fps = float(model.data.get("video", {}).get("fps") or 30.0)
    output_fps = float(fps) if fps is not None else source_fps / frame_stride
    output_fps = max(output_fps, 1.0)
    frames = list(range(0, model.max_frame + 1, frame_stride))

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
        ax.set_xlabel("Posición X")
        ax.set_ylabel("Posición Y")
        ax.plot(
            [0, field_width, field_width, 0, 0],
            [0, 0, field_height, field_height, 0],
            color="white",
            linewidth=2,
        )
        ax.axvline(field_width / 2, color="white", linewidth=1, alpha=0.7)
        center_circle = plt.Circle(
            (field_width / 2, field_height / 2),
            min(field_width, field_height) * 0.12,
            fill=False,
            color="white",
            linewidth=1,
            alpha=0.75,
        )
        ax.add_patch(center_circle)

        # Plano canónico temporal: la portería cercana a la cámara se coloca a
        # la izquierda (defendida por aliados) y la lejana a la derecha. Las
        # dimensiones exactas se sustituirán cuando estén disponibles.
        goal_y = (field_height - resolved_goal_width) / 2.0
        near_goal = Rectangle(
            (-resolved_goal_depth, goal_y),
            resolved_goal_depth,
            resolved_goal_width,
            fill=True,
            facecolor="#00AEEF22",
            edgecolor="#00AEEF",
            linewidth=2.5,
            hatch="///",
        )
        far_goal = Rectangle(
            (field_width, goal_y),
            resolved_goal_depth,
            resolved_goal_width,
            fill=True,
            facecolor="#FF00B822",
            edgecolor="#FF00B8",
            linewidth=2.5,
            hatch="///",
        )
        ax.add_patch(near_goal)
        ax.add_patch(far_goal)
        ax.text(
            -resolved_goal_depth * 0.5,
            goal_y + resolved_goal_width + 1.2,
            "PORTERÍA ALIADA",
            ha="center",
            fontsize=7,
            fontweight="bold",
            color="#006f99",
        )
        ax.text(
            field_width + resolved_goal_depth * 0.5,
            goal_y + resolved_goal_width + 1.2,
            "PORTERÍA RIVAL",
            ha="center",
            fontsize=7,
            fontweight="bold",
            color="#a00075",
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
