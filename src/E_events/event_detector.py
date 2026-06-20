import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.D_domain.game_state import GameState
from src.I_field_geometry.field_spec import FieldSpec


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
        self.ball_missing_reported = False
        self.last_event_frame_by_key: dict[str, int] = {}
        self.collision_cooldown_frames = 30
        self.possession_stability_frames = 5
        self.candidate_ball_owner: str | None = None
        self.candidate_ball_owner_frames = 0
        self.goal_inside_frames: dict[str, int] = {}
        self.goal_outside_frames: dict[str, int] = {}
        self.goal_armed: dict[str, bool] = {}
        self.goal_confirmation_frames = 4
        self.goal_rearm_frames = 8
        self.field_spec = FieldSpec.load()
        self.penalty_presence_frames: dict[str, int] = {}
        self.penalty_inside_reported: set[str] = set()
        self.penalty_confirmation_frames = 3

    def process_frame_record(self, frame_record: dict[str, Any]) -> None:
        self.game_state.update_from_frame_record(frame_record)
        self._detect_possession_change()
        self._detect_goal()
        self._detect_ball_out_of_field()
        self._detect_robot_inactive()
        self._detect_robot_collision()
        self._detect_ball_missing()
        self._detect_penalty_area_entries()

    def _robot_name(self, robot_id: str | None) -> str:
        if robot_id is None:
            return "Robot desconocido"
        robot = self.game_state.robots.get(robot_id)
        return robot.display_name if robot and robot.display_name else robot_id

    def _add_event(
        self,
        event_type: str,
        description: str,
        data: dict[str, Any] | None = None,
        event_key: str | None = None,
        cooldown_frames: int = 0,
    ) -> None:
        key = event_key or event_type
        last_frame = self.last_event_frame_by_key.get(key)
        if (
            last_frame is not None
            and self.game_state.frame_index - last_frame < cooldown_frames
        ):
            return
        self.last_event_frame_by_key[key] = self.game_state.frame_index
        self.events.append(
            MatchEvent(
                frame_index=self.game_state.frame_index,
                timestamp_seconds=self.game_state.timestamp_seconds,
                event_type=event_type,
                description=description,
                data=data or {},
            )
        )

    def _detect_goal(self) -> None:
        """Confirma un gol cuando el centro medido del balón entra al cajón.

        Esta primera versión no decide qué equipo anotó. La orientación y la
        línea exacta de gol se refinarán después con la máscara/homografía.
        """
        ball = self.game_state.ball
        visible_goal_ids = {
            goal_id
            for goal_id, goal in self.game_state.goals.items()
            if goal.visible
        }
        for goal_id in list(self.goal_inside_frames):
            if goal_id not in visible_goal_ids:
                self.goal_inside_frames[goal_id] = 0

        if ball is None or not ball.visible or ball.predicted:
            return

        for goal_id, goal in self.game_state.goals.items():
            if not goal.visible:
                continue
            armed = self.goal_armed.setdefault(goal_id, True)
            inside = goal.contains_ball_center(ball.bbox, inset_ratio=0.02)
            if inside:
                self.goal_inside_frames[goal_id] = self.goal_inside_frames.get(goal_id, 0) + 1
                self.goal_outside_frames[goal_id] = 0
            else:
                self.goal_inside_frames[goal_id] = 0
                self.goal_outside_frames[goal_id] = self.goal_outside_frames.get(goal_id, 0) + 1
                if self.goal_outside_frames[goal_id] >= self.goal_rearm_frames:
                    self.goal_armed[goal_id] = True
                continue

            if not armed or self.goal_inside_frames[goal_id] < self.goal_confirmation_frames:
                continue

            confidence = max(0.0, min(1.0, min(ball.confidence, goal.confidence)))
            side = goal.side_image if goal.side_image != "desconocida" else "detectada"
            goal_side_field = "desconocida"
            scoring_team = "desconocido"
            if (
                goal.field_transform_valid
                and goal.field_x_norm is not None
                and goal.field_transform_confidence >= 0.45
            ):
                goal_side_field = "cercana" if goal.field_x_norm < 0.5 else "lejana"
                # Convención temporal acordada: la portería cercana es defendida
                # por aliados y la lejana por rivales. Anota el equipo contrario.
                scoring_team = "rival" if goal_side_field == "cercana" else "aliado"

            self._add_event(
                event_type="goal",
                description=(
                    f"Gol detectado en la portería {goal_side_field}."
                    if goal_side_field != "desconocida"
                    else f"Gol detectado en la portería {side}."
                ),
                data={
                    "goal_id": goal_id,
                    "goal_side_image": goal.side_image,
                    "goal_side_field": goal_side_field,
                    "goal_bbox_xyxy": goal.bbox.to_xyxy(),
                    "goal_field_polygon": goal.field_polygon,
                    "ball_bbox_xyxy": ball.bbox.to_xyxy(),
                    "confirmation_frames": self.goal_inside_frames[goal_id],
                    "confidence": round(confidence, 4),
                    "scoring_team": scoring_team,
                    "detection_method": "ball_center_inside_goal_box",
                    "homography_confidence": round(goal.field_transform_confidence, 4),
                    "requires_homography_refinement": not goal.field_transform_valid,
                },
                event_key=f"goal:{goal_id}",
            )
            self.goal_armed[goal_id] = False

    def _detect_possession_change(self) -> None:
        ball = self.game_state.ball
        if ball is None:
            return
        current_owner = ball.owner_robot_id
        if current_owner is None:
            self.candidate_ball_owner = None
            self.candidate_ball_owner_frames = 0
            return

        if current_owner == self.candidate_ball_owner:
            self.candidate_ball_owner_frames += 1
        else:
            self.candidate_ball_owner = current_owner
            self.candidate_ball_owner_frames = 1
        if self.candidate_ball_owner_frames < self.possession_stability_frames:
            return

        if current_owner != self.previous_ball_owner:
            robot = self.game_state.robots.get(current_owner)
            self._add_event(
                event_type="possession_change",
                description=f"{self._robot_name(current_owner)} obtiene la posesión del balón.",
                data={
                    "robot_id": current_owner,
                    "robot_name": self._robot_name(current_owner),
                    "team": robot.team if robot else "desconocido",
                    "previous_owner": self.previous_ball_owner,
                },
                event_key="possession_change",
                cooldown_frames=15,
            )
            self.previous_ball_owner = current_owner

    def _detect_ball_out_of_field(self) -> None:
        ball = self.game_state.ball
        field = self.game_state.field
        if ball is None or not ball.visible:
            return

        if ball.field_transform_valid and ball.inside_surface is not None:
            ball_out = not ball.inside_surface
            method = "homografia_superficie"
        elif field is not None:
            ball_out = not field.contains_bbox_center(ball.bbox)
            method = "caja_cancha_fallback"
        else:
            return

        if ball_out and not self.previous_ball_out:
            self._add_event(
                event_type="ball_out_of_field",
                description="El balón salió de la superficie detectada de la cancha.",
                data={
                    "object_id": "ball",
                    "ball_bbox": ball.bbox.to_xyxy(),
                    "last_known_bbox": ball.bbox.to_xyxy(),
                    "field_x_norm": ball.field_x_norm,
                    "field_y_norm": ball.field_y_norm,
                    "detection_method": method,
                    "homography_confidence": ball.field_transform_confidence,
                },
            )
        self.previous_ball_out = ball_out

    def _detect_robot_inactive(self) -> None:
        for robot_id, robot in self.game_state.robots.items():
            if robot.active:
                if robot_id in self.inactive_robots_reported:
                    self.inactive_robots_reported.remove(robot_id)
                    self._add_event(
                        event_type="robot_reactivated",
                        description=f"{self._robot_name(robot_id)} volvió a la cancha.",
                        data={
                            "robot_id": robot_id,
                            "robot_name": self._robot_name(robot_id),
                            "team": robot.team,
                            "confidence": round(float(robot.confidence), 4),
                        },
                        event_key=f"robot_reactivated:{robot_id}",
                        cooldown_frames=20,
                    )
                continue
            if robot_id in self.inactive_robots_reported:
                continue
            self.inactive_robots_reported.add(robot_id)
            self._add_event(
                event_type="robot_inactive_candidate",
                description=(
                    f"{self._robot_name(robot_id)} desapareció durante varios cuadros. "
                    "Puede tratarse de una oclusión o de una intervención arbitral."
                ),
                data={
                    "robot_id": robot_id,
                    "robot_name": self._robot_name(robot_id),
                    "team": robot.team,
                    "frames_missing": robot.frames_missing,
                    "last_known_bbox": robot.bbox.to_xyxy(),
                },
            )

    def _detect_ball_missing(self) -> None:
        ball = self.game_state.ball
        if ball is None:
            return
        if ball.visible:
            if self.ball_missing_reported:
                self._add_event(
                    event_type="ball_recovered",
                    description="El balón volvió a ser visible.",
                    data={
                        "object_id": "ball",
                        "confidence": round(float(ball.confidence), 4),
                        "predicted": bool(ball.predicted),
                        "field_x_norm": ball.field_x_norm,
                        "field_y_norm": ball.field_y_norm,
                    },
                    event_key="ball_recovered",
                    cooldown_frames=15,
                )
            self.ball_missing_reported = False
            return
        if self.ball_missing_reported:
            return
        self.ball_missing_reported = True
        self._add_event(
            event_type="ball_missing_candidate",
            description=(
                "El balón desapareció durante varios cuadros. "
                "Puede tratarse de una oclusión o de una intervención arbitral."
            ),
            data={
                "object_id": "ball",
                "frames_missing": ball.frames_missing,
                "last_known_bbox": ball.bbox.to_xyxy(),
            },
        )

    def _inside_penalty_area(self, x_norm: float, y_norm: float) -> str | None:
        depth = self.field_spec.penalty_depth_ratio
        half_width = 0.5 * self.field_spec.penalty_width_ratio
        if abs(float(y_norm) - 0.5) > half_width:
            return None
        x = float(x_norm)
        y_scaled = (float(y_norm) - 0.5) / max(half_width, 1e-9)
        if 0.0 <= x <= depth:
            # D-shaped semiellipse whose diameter lies on the goal line.
            if (x / max(depth, 1e-9)) ** 2 + y_scaled**2 <= 1.08:
                return "amarilla"
        if 1.0 - depth <= x <= 1.0:
            if ((1.0 - x) / max(depth, 1e-9)) ** 2 + y_scaled**2 <= 1.08:
                return "azul"
        return None

    def _detect_penalty_area_entries(self) -> None:
        for robot_id, robot in self.game_state.robots.items():
            valid = (
                robot.active
                and robot.field_transform_valid
                and robot.field_x_norm is not None
                and robot.field_y_norm is not None
            )
            side = self._inside_penalty_area(robot.field_x_norm, robot.field_y_norm) if valid else None
            if side is None:
                self.penalty_presence_frames[robot_id] = 0
                self.penalty_inside_reported.discard(robot_id)
                continue
            frames = self.penalty_presence_frames.get(robot_id, 0) + 1
            self.penalty_presence_frames[robot_id] = frames
            if frames < self.penalty_confirmation_frames or robot_id in self.penalty_inside_reported:
                continue
            self.penalty_inside_reported.add(robot_id)
            self._add_event(
                event_type="robot_entered_penalty_area",
                description=f"{self._robot_name(robot_id)} entró al área de la portería {side}.",
                data={
                    "robot_id": robot_id,
                    "robot_name": self._robot_name(robot_id),
                    "team": robot.team,
                    "penalty_side": side,
                    "field_x_norm": robot.field_x_norm,
                    "field_y_norm": robot.field_y_norm,
                    "confirmation_frames": frames,
                    "confidence": round(float(robot.field_transform_confidence), 4),
                },
                event_key=f"penalty_entry:{robot_id}:{side}",
                cooldown_frames=30,
            )

    def _detect_robot_collision(self) -> None:
        robots = [robot for robot in self.game_state.robots.values() if robot.active]
        for index, robot_a in enumerate(robots):
            for robot_b in robots[index + 1 :]:
                if not robot_a.bbox.expanded(10).intersects(robot_b.bbox):
                    continue
                pair = sorted([robot_a.robot_id, robot_b.robot_id])
                event_key = f"collision:{pair[0]}:{pair[1]}"
                self._add_event(
                    event_type="robot_collision_candidate",
                    description=(
                        f"{self._robot_name(pair[0])} y {self._robot_name(pair[1])} "
                        "están demasiado cerca o sus cajas se intersectan."
                    ),
                    data={
                        "robot_a": pair[0],
                        "robot_b": pair[1],
                        "robot_a_name": self._robot_name(pair[0]),
                        "robot_b_name": self._robot_name(pair[1]),
                    },
                    event_key=event_key,
                    cooldown_frames=self.collision_cooldown_frames,
                )


def read_detection_records(detections_path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(detections_path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def create_track_point(
    frame_index: int,
    timestamp_seconds: float,
    bbox,
    confidence: float,
    visible: bool = True,
    anchor: str = "center",
    stabilized_x_px: float | None = None,
    stabilized_y_px: float | None = None,
    registration_valid: bool = False,
    registration_quality: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if anchor == "bottom_center":
        x, y = bbox.bottom_center
    else:
        x, y = bbox.center

    point = {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "x_px": round(x, 2),
        "y_px": round(y, 2),
        "stabilized_x_px": (
            round(float(stabilized_x_px), 2) if stabilized_x_px is not None else None
        ),
        "stabilized_y_px": (
            round(float(stabilized_y_px), 2) if stabilized_y_px is not None else None
        ),
        "registration_valid": bool(registration_valid),
        "registration_quality": round(float(registration_quality), 5),
        "bbox_xyxy": bbox.to_xyxy(),
        "confidence": confidence,
        "visible": visible,
        "anchor": anchor,
    }
    if extra:
        point.update(extra)
    return point


def generate_events(detections_path: str | Path) -> tuple[list[dict], dict, dict]:
    detector = EventDetector()
    records = read_detection_records(detections_path)
    possession_frames: dict[str, int] = {}
    tracks = {"robots": {}, "ball": [], "goals": {}}
    total_frames = 0

    for record in records:
        detector.process_frame_record(record)
        total_frames += 1
        frame_index = detector.game_state.frame_index
        timestamp_seconds = detector.game_state.timestamp_seconds
        field_bbox = (
            detector.game_state.field.bbox.to_xyxy()
            if detector.game_state.field is not None
            else None
        )

        for robot_id, robot in detector.game_state.robots.items():
            tracks["robots"].setdefault(robot_id, []).append(
                create_track_point(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    bbox=robot.bbox,
                    confidence=robot.confidence,
                    visible=robot.active and robot.frames_missing <= 2,
                    anchor="bottom_center",
                    stabilized_x_px=robot.stabilized_x_px,
                    stabilized_y_px=robot.stabilized_y_px,
                    registration_valid=robot.registration_valid,
                    registration_quality=robot.registration_quality,
                    extra={
                        "robot_id": robot_id,
                        "team": robot.team,
                        "team_number": robot.team_number,
                        "display_name": robot.display_name,
                        "active": robot.active,
                        "has_ball": robot.has_ball,
                        "frames_missing": robot.frames_missing,
                        "predicted": robot.predicted,
                        "field_bbox_xyxy": field_bbox,
                        "field_x": robot.field_x,
                        "field_y": robot.field_y,
                        "field_x_norm": robot.field_x_norm,
                        "field_y_norm": robot.field_y_norm,
                        "inside_surface": robot.inside_surface,
                        "field_transform_valid": robot.field_transform_valid,
                        "field_transform_confidence": robot.field_transform_confidence,
                        "field_transform_source": robot.field_transform_source,
                    },
                )
            )

        for goal_id, goal in detector.game_state.goals.items():
            tracks["goals"].setdefault(goal_id, []).append(
                create_track_point(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    bbox=goal.bbox,
                    confidence=goal.confidence,
                    visible=goal.visible,
                    anchor="center",
                    extra={
                        "goal_id": goal_id,
                        "goal_side_image": goal.side_image,
                        "frames_missing": goal.frames_missing,
                        "field_x": goal.field_x,
                        "field_y": goal.field_y,
                        "field_x_norm": goal.field_x_norm,
                        "field_y_norm": goal.field_y_norm,
                        "field_transform_valid": goal.field_transform_valid,
                        "field_transform_confidence": goal.field_transform_confidence,
                        "field_transform_source": goal.field_transform_source,
                        "field_polygon": goal.field_polygon,
                    },
                )
            )

        ball = detector.game_state.ball
        if ball is not None:
            tracks["ball"].append(
                create_track_point(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    bbox=ball.bbox,
                    confidence=ball.confidence,
                    visible=ball.visible and ball.frames_missing <= 2,
                    anchor="center",
                    stabilized_x_px=ball.stabilized_x_px,
                    stabilized_y_px=ball.stabilized_y_px,
                    registration_valid=ball.registration_valid,
                    registration_quality=ball.registration_quality,
                    extra={
                        "object_id": "ball",
                        "owner_robot_id": ball.owner_robot_id,
                        "frames_missing": ball.frames_missing,
                        "predicted": ball.predicted,
                        "field_bbox_xyxy": field_bbox,
                        "field_x": ball.field_x,
                        "field_y": ball.field_y,
                        "field_x_norm": ball.field_x_norm,
                        "field_y_norm": ball.field_y_norm,
                        "inside_surface": ball.inside_surface,
                        "field_transform_valid": ball.field_transform_valid,
                        "field_transform_confidence": ball.field_transform_confidence,
                        "field_transform_source": ball.field_transform_source,
                    },
                )
            )

        if ball is not None and ball.owner_robot_id is not None:
            possession_frames[ball.owner_robot_id] = (
                possession_frames.get(ball.owner_robot_id, 0) + 1
            )

    fps = 30.0
    if len(records) > 1:
        duration = records[-1]["timestamp_seconds"] - records[0]["timestamp_seconds"]
        fps = (len(records) - 1) / duration if duration > 0 else 30.0

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
    summary["track_counts"] = {
        "robots": {
            robot_id: len(points) for robot_id, points in tracks["robots"].items()
        },
        "ball": len(tracks["ball"]),
        "goals": {
            goal_id: len(points) for goal_id, points in tracks["goals"].items()
        },
    }
    return [asdict(event) for event in detector.events], summary, tracks
