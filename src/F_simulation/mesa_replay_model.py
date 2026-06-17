import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayAgent:
    agent_id: str
    agent_type: str
    color: str
    current_state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, state: dict[str, Any]) -> None:
        self.current_state = state
        self.history.append(state)

    @property
    def visible(self) -> bool:
        return bool(self.current_state.get("visible", False))

    @property
    def x(self) -> float:
        return float(self.current_state.get("x_field", 0.0))

    @property
    def y(self) -> float:
        return float(self.current_state.get("y_field", 0.0))


class FutbotReplayModel:
    def __init__(
        self,
        json_path: str | Path,
        field_width: float = 100.0,
        field_height: float = 60.0,
    ) -> None:
        self.json_path = Path(json_path)
        self.field_width = field_width
        self.field_height = field_height

        self.data = self._load_json(self.json_path)
        self.tracks = self.data.get("tracks", {})
        self.events = self.data.get("events", [])

        self.current_frame = 0
        self.max_frame = self._get_max_frame()

        self.robot_points_by_frame = self._index_robot_points_by_frame()
        self.ball_points_by_frame = self._index_ball_points_by_frame()
        self.events_by_frame = self._index_events_by_frame()

        self.agents = self._create_agents()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _get_max_frame(self) -> int:
        max_frame = 0

        for points in self.tracks.get("robots", {}).values():
            for point in points:
                max_frame = max(max_frame, int(point["frame_index"]))

        for point in self.tracks.get("ball", []):
            max_frame = max(max_frame, int(point["frame_index"]))

        return max_frame

    def _index_robot_points_by_frame(self) -> dict[int, dict[str, dict[str, Any]]]:
        indexed: dict[int, dict[str, dict[str, Any]]] = {}

        for robot_id, points in self.tracks.get("robots", {}).items():
            for point in points:
                frame_index = int(point["frame_index"])
                indexed.setdefault(frame_index, {})[robot_id] = point

        return indexed

    def _index_ball_points_by_frame(self) -> dict[int, dict[str, Any]]:
        indexed = {}

        for point in self.tracks.get("ball", []):
            indexed[int(point["frame_index"])] = point

        return indexed

    def _index_events_by_frame(self) -> dict[int, list[dict[str, Any]]]:
        indexed: dict[int, list[dict[str, Any]]] = {}

        for event in self.events:
            frame_index = int(event["frame_index"])
            indexed.setdefault(frame_index, []).append(event)

        return indexed

    def _create_agents(self) -> dict[str, ReplayAgent]:
        agents: dict[str, ReplayAgent] = {}

        robot_colors = ["#00BFFF", "#FF00CC", "#7CFF00", "#FFB000"]

        for index, robot_id in enumerate(self.tracks.get("robots", {}).keys()):
            agents[robot_id] = ReplayAgent(
                agent_id=robot_id,
                agent_type="robot",
                color=robot_colors[index % len(robot_colors)],
            )

        agents["ball"] = ReplayAgent(
            agent_id="ball",
            agent_type="ball",
            color="#FF6A00",
        )

        return agents

    def _to_field_coordinates(self, point: dict[str, Any]) -> dict[str, Any]:
        x_norm = float(point.get("x_norm", 0.0))
        y_norm = float(point.get("y_norm", 0.0))

        state = point.copy()
        state["x_field"] = round(x_norm * self.field_width, 4)
        state["y_field"] = round((1.0 - y_norm) * self.field_height, 4)

        return state

    def step(self) -> dict[str, Any]:
        robot_points = self.robot_points_by_frame.get(self.current_frame, {})
        ball_point = self.ball_points_by_frame.get(self.current_frame)
        frame_events = self.events_by_frame.get(self.current_frame, [])

        for robot_id, point in robot_points.items():
            self.agents[robot_id].update(self._to_field_coordinates(point))

        if ball_point is not None:
            self.agents["ball"].update(self._to_field_coordinates(ball_point))

        snapshot = self.get_snapshot(frame_events)

        self.current_frame += 1
        return snapshot

    def get_snapshot(
        self,
        frame_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "frame_index": self.current_frame,
            "agents": {
                agent_id: agent.current_state
                for agent_id, agent in self.agents.items()
            },
            "events": frame_events or [],
        }

    def reset(self) -> None:
        self.current_frame = 0

        for agent in self.agents.values():
            agent.current_state = {}
            agent.history = []

    def is_done(self) -> bool:
        return self.current_frame > self.max_frame