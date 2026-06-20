import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TEAM_COLORS = {
    "aliado": "#00AEEF",
    "rival": "#FF00B8",
    "desconocido": "#888888",
}


@dataclass
class ReplayAgent:
    agent_id: str
    agent_type: str
    color: str
    display_name: str
    team: str = "desconocido"
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
        ball_interpolation_max_seconds: float = 8.0,
    ) -> None:
        self.json_path = Path(json_path)
        self.field_width = field_width
        self.field_height = field_height
        self.ball_interpolation_max_seconds = max(0.0, float(ball_interpolation_max_seconds))
        self.data = self._load_json(self.json_path)
        self.source_fps = max(1.0, float(self.data.get("video", {}).get("fps") or 30.0))
        raw_tracks = self.data.get("tracks", {})
        self.tracks = self._prepare_tracks(raw_tracks)
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


    @staticmethod
    def _estimate_velocity(
        points: list[dict[str, Any]],
        index: int,
        direction: int,
    ) -> np.ndarray:
        origin = points[index]
        origin_frame = int(origin.get("frame_index", 0))
        origin_position = np.array(
            [float(origin.get("x_norm", 0.0)), float(origin.get("y_norm", 0.0))],
            dtype=np.float64,
        )
        neighbor_index = index + direction
        if neighbor_index < 0 or neighbor_index >= len(points):
            return np.zeros(2, dtype=np.float64)
        neighbor = points[neighbor_index]
        neighbor_frame = int(neighbor.get("frame_index", 0))
        neighbor_position = np.array(
            [float(neighbor.get("x_norm", 0.0)), float(neighbor.get("y_norm", 0.0))],
            dtype=np.float64,
        )
        frame_delta = neighbor_frame - origin_frame
        if frame_delta == 0:
            return np.zeros(2, dtype=np.float64)
        return (neighbor_position - origin_position) / float(frame_delta)

    @classmethod
    def _interpolate_ball_gaps(
        cls,
        points: list[dict[str, Any]],
        max_gap_frames: int,
    ) -> list[dict[str, Any]]:
        """Rellena huecos acotados con una curva Hermite suave.

        Los puntos creados se marcan como ``interpolado`` para distinguirlos
        de mediciones reales. El replay obtiene continuidad visual sin ocultar
        que la trayectoria fue estimada.
        """
        ordered = sorted(points, key=lambda point: int(point.get("frame_index", 0)))
        if len(ordered) < 2 or max_gap_frames <= 0:
            return ordered

        output: list[dict[str, Any]] = []
        for index, start in enumerate(ordered[:-1]):
            end = ordered[index + 1]
            output.append(start.copy())
            start_frame = int(start.get("frame_index", 0))
            end_frame = int(end.get("frame_index", 0))
            missing = end_frame - start_frame - 1
            if missing <= 0 or missing > max_gap_frames:
                continue

            p0 = np.array(
                [float(start.get("x_norm", 0.0)), float(start.get("y_norm", 0.0))],
                dtype=np.float64,
            )
            p1 = np.array(
                [float(end.get("x_norm", 0.0)), float(end.get("y_norm", 0.0))],
                dtype=np.float64,
            )
            v0 = cls._estimate_velocity(ordered, index, -1)
            v1 = cls._estimate_velocity(ordered, index + 1, 1)
            total_frames = float(end_frame - start_frame)

            # Limita tangentes para evitar curvas con bucles cuando las
            # detecciones vecinas son ruidosas.
            chord = p1 - p0
            chord_length = float(np.linalg.norm(chord))
            tangent_limit = max(0.003, 1.8 * chord_length / max(total_frames, 1.0))
            for velocity in (v0, v1):
                speed = float(np.linalg.norm(velocity))
                if speed > tangent_limit and speed > 1e-9:
                    velocity *= tangent_limit / speed

            low = np.minimum(p0, p1) - 0.08
            high = np.maximum(p0, p1) + 0.08
            start_confidence = float(start.get("confidence", 0.5))
            end_confidence = float(end.get("confidence", 0.5))

            for offset in range(1, missing + 1):
                t = offset / total_frames
                h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
                h10 = t**3 - 2.0 * t**2 + t
                h01 = -2.0 * t**3 + 3.0 * t**2
                h11 = t**3 - t**2
                position = (
                    h00 * p0
                    + h10 * total_frames * v0
                    + h01 * p1
                    + h11 * total_frames * v1
                )
                position = np.clip(position, low, high)
                position = np.clip(position, -0.25, 1.25)
                center_weight = 1.0 - abs(2.0 * t - 1.0)
                confidence = min(start_confidence, end_confidence) * (0.48 - 0.18 * center_weight)
                interpolated = start.copy()
                interpolated.update(
                    {
                        "frame_index": start_frame + offset,
                        "timestamp_seconds": (
                            float(start.get("timestamp_seconds", start_frame))
                            + t
                            * (
                                float(end.get("timestamp_seconds", end_frame))
                                - float(start.get("timestamp_seconds", start_frame))
                            )
                        ),
                        "x_norm": round(float(position[0]), 6),
                        "y_norm": round(float(position[1]), 6),
                        "confidence": round(float(max(0.05, confidence)), 6),
                        "predicted": True,
                        "measured": False,
                        "visible": True,
                        "source": "interpolado",
                        "tracking_status": "interpolado",
                        "interpolation_gap_frames": missing,
                    }
                )
                output.append(interpolated)
        output.append(ordered[-1].copy())
        return sorted(output, key=lambda point: int(point.get("frame_index", 0)))

    @staticmethod
    def _smooth_points(
        points: list[dict[str, Any]],
        base_alpha: float,
        max_step: float,
    ) -> list[dict[str, Any]]:
        """Suavizado Kalman + RTS sin retraso visible.

        Primero limita teletransportes imposibles y después usa una pasada hacia
        adelante y otra hacia atrás. ``base_alpha`` se conserva en la interfaz
        por compatibilidad y controla indirectamente el ruido de medición.
        """
        ordered = sorted(points, key=lambda point: int(point.get("frame_index", 0)))
        if len(ordered) <= 1:
            return [point.copy() for point in ordered]

        measurements: list[np.ndarray] = []
        frames: list[int] = []
        prepared: list[dict[str, Any]] = []
        previous: np.ndarray | None = None
        previous_frame: int | None = None

        for raw in ordered:
            point = raw.copy()
            frame_index = int(point.get("frame_index", 0))
            current = np.clip(
                np.array(
                    [float(point.get("x_norm", 0.0)), float(point.get("y_norm", 0.0))],
                    dtype=np.float64,
                ),
                -0.25,
                1.25,
            )
            if previous is not None and previous_frame is not None:
                frame_gap = max(1, frame_index - previous_frame)
                delta = current - previous
                distance = float(np.linalg.norm(delta))
                allowed = max_step * frame_gap
                if distance > allowed and distance > 1e-9:
                    current = previous + delta * (allowed / distance)
                    point["replay_jump_clamped"] = True
            point["x_norm_raw"] = round(float(current[0]), 6)
            point["y_norm_raw"] = round(float(current[1]), 6)
            measurements.append(current)
            frames.append(frame_index)
            prepared.append(point)
            previous = current
            previous_frame = frame_index

        count = len(measurements)
        filtered = np.zeros((count, 4), dtype=np.float64)
        filtered_cov = np.zeros((count, 4, 4), dtype=np.float64)
        predicted = np.zeros_like(filtered)
        predicted_cov = np.zeros_like(filtered_cov)
        transitions: list[np.ndarray] = [np.eye(4, dtype=np.float64)]

        state = np.array([measurements[0][0], measurements[0][1], 0.0, 0.0])
        covariance = np.eye(4, dtype=np.float64) * 0.03
        observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        identity = np.eye(4, dtype=np.float64)

        for index, (measurement, point) in enumerate(zip(measurements, prepared)):
            dt = 1.0 if index == 0 else float(max(1, frames[index] - frames[index - 1]))
            transition = np.array(
                [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            if index > 0:
                transitions.append(transition)
            process_scale = 1.5e-5 * max(1.0, dt)
            process_noise = np.diag(
                [process_scale, process_scale, 5.0 * process_scale, 5.0 * process_scale]
            )
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise
            predicted[index] = predicted_state
            predicted_cov[index] = predicted_covariance

            confidence = float(point.get("confidence", 0.6))
            noise = (0.00065 - 0.00038 * min(max(base_alpha, 0.0), 1.0))
            noise /= max(0.20, confidence)
            if bool(point.get("predicted", False)):
                noise *= 3.0
            measurement_noise = np.eye(2, dtype=np.float64) * noise
            innovation = measurement - observation @ predicted_state
            innovation_covariance = (
                observation @ predicted_covariance @ observation.T + measurement_noise
            )
            gain = predicted_covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
            state = predicted_state + gain @ innovation
            covariance = (identity - gain @ observation) @ predicted_covariance
            filtered[index] = state
            filtered_cov[index] = covariance

        smoothed = filtered.copy()
        smoothed_cov = filtered_cov.copy()
        for index in range(count - 2, -1, -1):
            transition = transitions[index + 1]
            smoother_gain = (
                filtered_cov[index]
                @ transition.T
                @ np.linalg.pinv(predicted_cov[index + 1])
            )
            smoothed[index] = filtered[index] + smoother_gain @ (
                smoothed[index + 1] - predicted[index + 1]
            )
            smoothed_cov[index] = filtered_cov[index] + smoother_gain @ (
                smoothed_cov[index + 1] - predicted_cov[index + 1]
            ) @ smoother_gain.T

        output: list[dict[str, Any]] = []
        for point, state in zip(prepared, smoothed):
            position = np.clip(state[:2], -0.25, 1.25)
            point["x_norm"] = round(float(position[0]), 6)
            point["y_norm"] = round(float(position[1]), 6)
            point["replay_smoothed_bidirectional"] = True
            output.append(point)
        return output

    def _prepare_tracks(self, tracks: dict[str, Any]) -> dict[str, Any]:
        prepared = {"robots": {}, "ball": []}
        for robot_id, points in tracks.get("robots", {}).items():
            # GameState conserva la última caja mientras el robot está perdido.
            # Esos puntos invisibles no son mediciones y deformarían el RTS.
            visible_points = [
                point for point in points if bool(point.get("visible", True))
            ]
            prepared["robots"][robot_id] = self._smooth_points(
                visible_points,
                base_alpha=0.34,
                max_step=0.032,
            )
        max_gap_frames = int(round(self.ball_interpolation_max_seconds * self.source_fps))
        # Los tracks de dominio conservan la última caja aun cuando el balón no
        # está visible. Esos puntos no son mediciones y ocultarían los huecos que
        # queremos reconstruir, por eso se eliminan antes de interpolar.
        visible_ball_points = [
            point
            for point in tracks.get("ball", [])
            if bool(point.get("visible", True))
        ]
        interpolated_ball = self._interpolate_ball_gaps(
            visible_ball_points,
            max_gap_frames=max_gap_frames,
        )
        prepared["ball"] = self._smooth_points(
            interpolated_ball,
            base_alpha=0.54,
            max_step=0.075,
        )
        return prepared

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
                indexed.setdefault(int(point["frame_index"]), {})[robot_id] = point
        return indexed

    def _index_ball_points_by_frame(self) -> dict[int, dict[str, Any]]:
        return {
            int(point["frame_index"]): point
            for point in self.tracks.get("ball", [])
        }

    def _index_events_by_frame(self) -> dict[int, list[dict[str, Any]]]:
        indexed: dict[int, list[dict[str, Any]]] = {}
        for event in self.events:
            indexed.setdefault(int(event["frame_index"]), []).append(event)
        return indexed

    def _create_agents(self) -> dict[str, ReplayAgent]:
        agents: dict[str, ReplayAgent] = {}
        for robot_id, points in self.tracks.get("robots", {}).items():
            team_votes: dict[str, int] = {}
            for point in points:
                team_value = str(point.get("team", "desconocido"))
                if team_value != "desconocido":
                    team_votes[team_value] = team_votes.get(team_value, 0) + 1
            team = (
                max(team_votes, key=team_votes.get)
                if team_votes
                else "desconocido"
            )
            named = [
                str(point.get("display_name"))
                for point in points
                if point.get("display_name")
                and str(point.get("team", "desconocido")) == team
            ]
            display_name = named[-1] if named else robot_id
            agents[robot_id] = ReplayAgent(
                agent_id=robot_id,
                agent_type="robot",
                color=TEAM_COLORS.get(team, TEAM_COLORS["desconocido"]),
                display_name=display_name,
                team=team,
            )
        agents["ball"] = ReplayAgent(
            agent_id="ball",
            agent_type="ball",
            color="#FF6A00",
            display_name="Balón",
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

        # Explicitly hide agents that have no visible observation in this frame.
        for robot_id, agent in self.agents.items():
            if robot_id == "ball":
                continue
            if robot_id not in robot_points and agent.current_state:
                agent.current_state = {**agent.current_state, "visible": False}

        for robot_id, point in robot_points.items():
            self.agents[robot_id].update(self._to_field_coordinates(point))

        if ball_point is not None:
            self.agents["ball"].update(self._to_field_coordinates(ball_point))
        elif self.agents["ball"].current_state:
            self.agents["ball"].current_state = {
                **self.agents["ball"].current_state,
                "visible": False,
            }

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
                agent_id: agent.current_state for agent_id, agent in self.agents.items()
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
