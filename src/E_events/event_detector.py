import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.D_domain.game_state import GameState


@dataclass
class MatchEvent:
    frame_index: int
    timestamp_seconds: float
    event_type: str
    description: str
    data: dict[str, Any]


class EventDetector:
    def __init__(self) -> None:
        self.game_state = GameState()
        self.events: list[MatchEvent] = []
        self.previous_ball_owner: str | None = None
        self.previous_ball_out = False
        self.inactive_robots_reported: set[str] = set()

    def process_frame_record(self, frame_record: dict[str, Any]) -> None:
        self.game_state.update_from_frame_record(frame_record)

        self._detect_possession_change()
        self._detect_ball_out_of_field()
        self._detect_robot_inactive()
        self._detect_robot_collision()

    def _add_event(
        self,
        event_type: str,
        description: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            MatchEvent(
                frame_index=self.game_state.frame_index,
                timestamp_seconds=self.game_state.timestamp_seconds,
                event_type=event_type,
                description=description,
                data=data or {},
            )
        )

    def _detect_possession_change(self) -> None:
        ball = self.game_state.ball

        if ball is None:
            return

        current_owner = ball.owner_robot_id

        if current_owner != self.previous_ball_owner:
            if current_owner is not None:
                self._add_event(
                    event_type="possession_change",
                    description=f"{current_owner} obtiene posesion de la pelota.",
                    data={
                        "robot_id": current_owner,
                        "previous_owner": self.previous_ball_owner,
                    },
                )

            self.previous_ball_owner = current_owner

    def _detect_ball_out_of_field(self) -> None:
        ball = self.game_state.ball
        field = self.game_state.field

        if ball is None or field is None or not ball.visible:
            return

        ball_out = not field.contains_bbox_center(ball.bbox)

        if ball_out and not self.previous_ball_out:
            self._add_event(
                event_type="ball_out_of_field",
                description="La pelota salio del area detectada de la cancha.",
                data={
                    "ball_bbox": ball.bbox.to_xyxy(),
                },
            )

        self.previous_ball_out = ball_out

    def _detect_robot_inactive(self) -> None:
        for robot_id, robot in self.game_state.robots.items():
            if robot.active:
                continue

            if robot_id in self.inactive_robots_reported:
                continue

            self.inactive_robots_reported.add(robot_id)

            self._add_event(
                event_type="robot_inactive_candidate",
                description=(
                    f"{robot_id} desaparecio varios frames. "
                    "Puede ser oclusion o intervencion del arbitro."
                ),
                data={
                    "robot_id": robot_id,
                    "frames_missing": robot.frames_missing,
                },
            )

    def _detect_robot_collision(self) -> None:
        robots = [
            robot
            for robot in self.game_state.robots.values()
            if robot.active
        ]

        for index, robot_a in enumerate(robots):
            for robot_b in robots[index + 1:]:
                if robot_a.bbox.expanded(10).intersects(robot_b.bbox):
                    self._add_event(
                        event_type="robot_collision_candidate",
                        description=(
                            f"{robot_a.robot_id} y {robot_b.robot_id} "
                            "estan demasiado cerca o intersectando."
                        ),
                        data={
                            "robot_a": robot_a.robot_id,
                            "robot_b": robot_b.robot_id,
                        },
                    )


def read_detection_records(detections_path: str | Path) -> list[dict[str, Any]]:
    records = []

    with Path(detections_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def generate_events(detections_path: str | Path) -> tuple[list[dict], dict]:
    detector = EventDetector()
    records = read_detection_records(detections_path)

    possession_frames: dict[str, int] = {}
    total_frames = 0

    for record in records:
        detector.process_frame_record(record)
        total_frames += 1

        ball = detector.game_state.ball
        if ball is not None and ball.owner_robot_id is not None:
            possession_frames[ball.owner_robot_id] = (
                possession_frames.get(ball.owner_robot_id, 0) + 1
            )

    fps = 30.0
    if len(records) > 1:
        duration = records[-1]["timestamp_seconds"] - records[0]["timestamp_seconds"]
        fps = total_frames / duration if duration > 0 else 30.0

    summary = {
        "total_frames_analyzed": total_frames,
        "total_events": len(detector.events),
        "event_counts": {},
        "possession_frames": possession_frames,
        "possession_seconds": {
            robot_id: round(frames / fps, 3)
            for robot_id, frames in possession_frames.items()
        },
    }

    for event in detector.events:
        summary["event_counts"][event.event_type] = (
            summary["event_counts"].get(event.event_type, 0) + 1
        )

    return [asdict(event) for event in detector.events], summary