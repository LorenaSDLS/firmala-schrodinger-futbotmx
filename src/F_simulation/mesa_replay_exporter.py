import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.F_simulation.mesa_replay_model import FutbotReplayModel


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def describe_event(event: dict) -> str:
    timestamp = format_time(float(event.get("timestamp_seconds", 0.0)))
    event_type = event.get("event_type", "evento")
    description = event.get("description", "")

    return f"{timestamp} | {event_type}\n{description}"


def export_mesa_replay_video(
    json_path: str | Path,
    output_path: str | Path | None = None,
    fps: int = 30,
    frame_stride: int = 1,
    field_width: float = 100.0,
    field_height: float = 60.0,
) -> str:
    json_path = Path(json_path)

    if output_path is None:
        output_path = json_path.parent / "mesa_replay.mp4"

    output_path = Path(output_path)

    model = FutbotReplayModel(
        json_path=json_path,
        field_width=field_width,
        field_height=field_height,
    )

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
        ax.set_xlim(0, field_width)
        ax.set_ylim(0, field_height)
        ax.set_facecolor("#16a36a")
        ax.set_title(f"FutBotMX Mesa Replay - frame {frame_index}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.plot(
            [0, field_width, field_width, 0, 0],
            [0, 0, field_height, field_height, 0],
            color="white",
            linewidth=2,
        )

        ax.axvline(field_width / 2, color="white", linewidth=1, alpha=0.7)

        agents = snapshot["agents"]

        for agent_id, state in agents.items():
            if not state:
                continue

            if not state.get("visible", True):
                continue

            x = state.get("x_field")
            y = state.get("y_field")

            if x is None or y is None:
                continue

            if agent_id == "ball":
                ax.scatter(x, y, s=90, color="#ff6a00", edgecolors="black")
                ax.text(x + 1, y + 1, "ball", fontsize=8, color="black")
            else:
                color = model.agents[agent_id].color
                ax.scatter(x, y, s=180, color=color, edgecolors="black")
                ax.text(x + 1, y + 1, agent_id, fontsize=8, color="black")

        frame_events = snapshot.get("events", [])

        if frame_events:
            event_text = "\n".join(
                f"- {event['event_type']}"
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
            event_ax.text(
                0.05,
                0.95,
                "Sin eventos todavia",
                va="top",
                fontsize=9,
            )
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
        interval=1000 / fps,
        repeat=False,
    )

    try:
        animation.save(output_path, fps=fps)
    except Exception:
        output_path = output_path.with_suffix(".gif")
        animation.save(output_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera una animacion tipo Mesa desde futbot_unity_mesa.json."
    )

    parser.add_argument(
        "json_path",
        help="Ruta a futbot_unity_mesa.json.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida. Default: mesa_replay.mp4 junto al JSON.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Procesa cada N frames. Usa 2 o 3 para pruebas mas rapidas.",
    )

    args = parser.parse_args()

    output_path = export_mesa_replay_video(
        json_path=args.json_path,
        output_path=args.output,
        fps=args.fps,
        frame_stride=args.frame_stride,
    )

    print(f"Animacion generada: {output_path}")


if __name__ == "__main__":
    main()