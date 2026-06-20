from dataclasses import dataclass, field
from typing import Any

from src.D_domain.ball import Ball
from src.D_domain.field import Field
from src.D_domain.geometry import BBox
from src.D_domain.goal import Goal
from src.D_domain.referee_hand import RefereeHand
from src.D_domain.robot import Robot


@dataclass
class GameState:
    frame_index: int = 0
    timestamp_seconds: float = 0.0
    robots: dict[str, Robot] = field(default_factory=dict)
    goals: dict[str, Goal] = field(default_factory=dict)
    ball: Ball | None = None
    field: Field | None = None
    referee_hand: RefereeHand | None = None
    max_robots: int = 4
    robot_match_distance: float = 180.0

    def update_from_frame_record(self, frame_record: dict[str, Any]) -> None:
        self.frame_index = int(frame_record["frame_index"])
        self.timestamp_seconds = float(frame_record["timestamp_seconds"])

        detections = frame_record.get("detections", [])
        seen_robot_ids: set[str] = set()
        ball_seen = False
        field_seen = False
        seen_goal_ids: set[str] = set()

        for detection in detections:
            class_name = str(detection.get("class_name", "")).lower()
            class_group = str(detection.get("class_group", "")).lower()
            bbox_values = detection.get("bbox_xyxy") or detection.get("box")
            if not bbox_values:
                continue

            bbox = BBox.from_xyxy(bbox_values)
            confidence = float(detection.get("confidence", 0.0))

            if class_group == "robot" or class_name in {"robot", "robots"}:
                robot_id = self._resolve_robot_id(detection, bbox, seen_robot_ids)
                seen_robot_ids.add(robot_id)
                self._update_robot(
                    robot_id=robot_id,
                    bbox=bbox,
                    confidence=confidence,
                    predicted=bool(detection.get("predicted", False)),
                    missed_frames=int(detection.get("track_missed_frames", 0)),
                    team=str(detection.get("team", "desconocido")),
                    team_number=detection.get("team_number"),
                    display_name=detection.get("display_name"),
                    stabilized_x_px=detection.get("stabilized_x_px"),
                    stabilized_y_px=detection.get("stabilized_y_px"),
                    registration_valid=bool(detection.get("registration_valid", False)),
                    registration_quality=float(detection.get("registration_quality", 0.0)),
                    field_x=detection.get("field_x"),
                    field_y=detection.get("field_y"),
                    field_x_norm=detection.get("field_x_norm"),
                    field_y_norm=detection.get("field_y_norm"),
                    inside_surface=detection.get("inside_surface"),
                    field_transform_valid=bool(detection.get("field_transform_valid", False)),
                    field_transform_confidence=float(detection.get("field_transform_confidence", 0.0)),
                    field_transform_source=str(detection.get("field_transform_source", "sin_calibracion")),
                )

            elif class_group == "ball" or class_name in {
                "orange ball", "ball", "pelota", "balon", "balón"
            }:
                ball_seen = True
                self._update_ball(
                    bbox=bbox,
                    confidence=confidence,
                    predicted=bool(detection.get("predicted", False)),
                    missed_frames=int(detection.get("track_missed_frames", 0)),
                    stabilized_x_px=detection.get("stabilized_x_px"),
                    stabilized_y_px=detection.get("stabilized_y_px"),
                    registration_valid=bool(detection.get("registration_valid", False)),
                    registration_quality=float(detection.get("registration_quality", 0.0)),
                    field_x=detection.get("field_x"),
                    field_y=detection.get("field_y"),
                    field_x_norm=detection.get("field_x_norm"),
                    field_y_norm=detection.get("field_y_norm"),
                    inside_surface=detection.get("inside_surface"),
                    field_transform_valid=bool(detection.get("field_transform_valid", False)),
                    field_transform_confidence=float(detection.get("field_transform_confidence", 0.0)),
                    field_transform_source=str(detection.get("field_transform_source", "sin_calibracion")),
                )

            elif class_group == "field" or class_name in {
                "field", "playing field", "cancha", "campo"
            }:
                field_seen = True
                self._update_field(bbox, confidence)

            elif class_group == "goal" or class_name in {
                "goal", "goals", "goal box", "goal_box", "goal mouth",
                "goal_mouth", "porteria", "portería", "arco"
            }:
                goal_id = str(detection.get("goal_id") or "goal_desconocida")
                side_image = str(detection.get("goal_side_image") or "desconocida")
                seen_goal_ids.add(goal_id)
                self._update_goal(
                    goal_id,
                    side_image,
                    bbox,
                    confidence,
                    field_x=detection.get("field_x"),
                    field_y=detection.get("field_y"),
                    field_x_norm=detection.get("field_x_norm"),
                    field_y_norm=detection.get("field_y_norm"),
                    field_transform_valid=bool(detection.get("field_transform_valid", False)),
                    field_transform_confidence=float(detection.get("field_transform_confidence", 0.0)),
                    field_transform_source=str(detection.get("field_transform_source", "sin_calibracion")),
                    field_polygon=detection.get("field_polygon"),
                )

            elif class_name in {"hand", "human hand", "referee hand", "mano"}:
                self._update_referee_hand(bbox, confidence)

        self._mark_missing_objects(seen_robot_ids, ball_seen, field_seen, seen_goal_ids)
        self._update_ball_owner()

    def _next_robot_id(self) -> str:
        for index in range(self.max_robots):
            candidate = f"robot_{index}"
            if candidate not in self.robots:
                return candidate
        return f"robot_{len(self.robots)}"

    def _closest_robot_id_any(self, bbox: BBox) -> str:
        if not self.robots:
            return self._next_robot_id()
        return min(
            self.robots,
            key=lambda robot_id: self.robots[robot_id].distance_to_bbox(bbox),
        )

    def _resolve_robot_id(
        self,
        detection: dict[str, Any],
        bbox: BBox,
        seen_robot_ids: set[str],
    ) -> str:
        tracking_id = detection.get("tracking_id")
        if tracking_id is not None:
            return f"robot_{tracking_id}"

        closest_robot_id = None
        closest_distance = float("inf")
        for robot_id, robot in self.robots.items():
            if robot_id in seen_robot_ids:
                continue
            distance = robot.distance_to_bbox(bbox)
            if distance < closest_distance:
                closest_distance = distance
                closest_robot_id = robot_id

        if closest_robot_id is not None:
            if closest_distance <= self.robot_match_distance:
                return closest_robot_id
            if len(self.robots) >= self.max_robots:
                return closest_robot_id
        if len(self.robots) < self.max_robots:
            return self._next_robot_id()
        return self._closest_robot_id_any(bbox)

    def _update_robot(
        self,
        robot_id: str,
        bbox: BBox,
        confidence: float,
        predicted: bool = False,
        missed_frames: int = 0,
        team: str = "desconocido",
        team_number: int | None = None,
        display_name: str | None = None,
        stabilized_x_px: float | None = None,
        stabilized_y_px: float | None = None,
        registration_valid: bool = False,
        registration_quality: float = 0.0,
        field_x: float | None = None,
        field_y: float | None = None,
        field_x_norm: float | None = None,
        field_y_norm: float | None = None,
        inside_surface: bool | None = None,
        field_transform_valid: bool = False,
        field_transform_confidence: float = 0.0,
        field_transform_source: str = "sin_calibracion",
    ) -> None:
        if robot_id not in self.robots:
            self.robots[robot_id] = Robot(
                robot_id=robot_id,
                bbox=bbox,
                confidence=confidence,
                team=team,
                team_number=(int(team_number) if team_number is not None else None),
                display_name=display_name,
            )

        self.robots[robot_id].update(
            frame_index=self.frame_index,
            bbox=bbox,
            confidence=confidence,
            predicted=predicted,
            missed_frames=missed_frames,
            team=team,
            team_number=(int(team_number) if team_number is not None else None),
            display_name=display_name,
            stabilized_x_px=stabilized_x_px,
            stabilized_y_px=stabilized_y_px,
            registration_valid=registration_valid,
            registration_quality=registration_quality,
            field_x=field_x,
            field_y=field_y,
            field_x_norm=field_x_norm,
            field_y_norm=field_y_norm,
            inside_surface=inside_surface,
            field_transform_valid=field_transform_valid,
            field_transform_confidence=field_transform_confidence,
            field_transform_source=field_transform_source,
        )

    def _update_ball(
        self,
        bbox: BBox,
        confidence: float,
        predicted: bool = False,
        missed_frames: int = 0,
        stabilized_x_px: float | None = None,
        stabilized_y_px: float | None = None,
        registration_valid: bool = False,
        registration_quality: float = 0.0,
        field_x: float | None = None,
        field_y: float | None = None,
        field_x_norm: float | None = None,
        field_y_norm: float | None = None,
        inside_surface: bool | None = None,
        field_transform_valid: bool = False,
        field_transform_confidence: float = 0.0,
        field_transform_source: str = "sin_calibracion",
    ) -> None:
        if self.ball is None:
            self.ball = Ball(bbox=bbox, confidence=confidence)
        self.ball.update(
            frame_index=self.frame_index,
            bbox=bbox,
            confidence=confidence,
            predicted=predicted,
            missed_frames=missed_frames,
            stabilized_x_px=stabilized_x_px,
            stabilized_y_px=stabilized_y_px,
            registration_valid=registration_valid,
            registration_quality=registration_quality,
            field_x=field_x,
            field_y=field_y,
            field_x_norm=field_x_norm,
            field_y_norm=field_y_norm,
            inside_surface=inside_surface,
            field_transform_valid=field_transform_valid,
            field_transform_confidence=field_transform_confidence,
            field_transform_source=field_transform_source,
        )

    def _update_field(self, bbox: BBox, confidence: float) -> None:
        if self.field is None:
            self.field = Field(bbox=bbox, confidence=confidence)
        else:
            self.field.update(bbox=bbox, confidence=confidence)

    def _update_goal(
        self,
        goal_id: str,
        side_image: str,
        bbox: BBox,
        confidence: float,
        field_x: float | None = None,
        field_y: float | None = None,
        field_x_norm: float | None = None,
        field_y_norm: float | None = None,
        field_transform_valid: bool = False,
        field_transform_confidence: float = 0.0,
        field_transform_source: str = "sin_calibracion",
        field_polygon: list[list[float]] | None = None,
    ) -> None:
        if goal_id not in self.goals:
            self.goals[goal_id] = Goal(
                goal_id=goal_id,
                side_image=side_image,
                bbox=bbox,
                confidence=confidence,
            )
        self.goals[goal_id].update(
            bbox,
            confidence,
            side_image,
            field_x=field_x,
            field_y=field_y,
            field_x_norm=field_x_norm,
            field_y_norm=field_y_norm,
            field_transform_valid=field_transform_valid,
            field_transform_confidence=field_transform_confidence,
            field_transform_source=field_transform_source,
            field_polygon=field_polygon,
        )

    def _update_referee_hand(self, bbox: BBox, confidence: float) -> None:
        if self.referee_hand is None:
            self.referee_hand = RefereeHand(bbox=bbox, confidence=confidence)
        else:
            self.referee_hand.update(bbox=bbox, confidence=confidence)

    def _mark_missing_objects(
        self,
        seen_robot_ids: set[str],
        ball_seen: bool,
        field_seen: bool,
        seen_goal_ids: set[str],
    ) -> None:
        for robot_id, robot in self.robots.items():
            if robot_id not in seen_robot_ids:
                robot.mark_missing()
        if self.ball is not None and not ball_seen:
            self.ball.mark_missing()
        for goal_id, goal in self.goals.items():
            if goal_id not in seen_goal_ids:
                goal.mark_missing()
        # The last valid field remains available as a coarse reference.
        _ = field_seen

    def _update_ball_owner(self) -> None:
        if self.ball is None or not self.ball.visible:
            return

        closest_robot_id = None
        closest_distance = float("inf")
        for robot_id, robot in self.robots.items():
            if not robot.active:
                continue
            distance = robot.distance_to_bbox(self.ball.bbox)
            if distance < closest_distance:
                closest_distance = distance
                closest_robot_id = robot_id

        if closest_robot_id is not None and closest_distance <= 90:
            self.ball.owner_robot_id = closest_robot_id
            for robot in self.robots.values():
                robot.has_ball = robot.robot_id == closest_robot_id
        else:
            self.ball.owner_robot_id = None
            for robot in self.robots.values():
                robot.has_ball = False
