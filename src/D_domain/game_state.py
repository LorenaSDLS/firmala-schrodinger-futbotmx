from dataclasses import dataclass, field
from typing import Any

from src.D_domain.ball import Ball
from src.D_domain.field import Field
from src.D_domain.geometry import BBox
from src.D_domain.referee_hand import RefereeHand
from src.D_domain.robot import Robot


@dataclass
class GameState:
    frame_index: int = 0
    timestamp_seconds: float = 0.0
    robots: dict[str, Robot] = field(default_factory=dict)
    ball: Ball | None = None
    field: Field | None = None
    referee_hand: RefereeHand | None = None

    def update_from_frame_record(self, frame_record: dict[str, Any]) -> None:
        self.frame_index = int(frame_record["frame_index"])
        self.timestamp_seconds = float(frame_record["timestamp_seconds"])

        detections = frame_record.get("detections", [])
        seen_robot_ids = set()
        ball_seen = False
        field_seen = False

        for detection in detections:
            class_name = detection.get("class_name", "").lower()
            bbox_values = (
                detection.get("bbox_xyxy")
                or detection.get("box")
            )

            if not bbox_values:
                continue

            bbox = BBox.from_xyxy(bbox_values)
            confidence = float(detection.get("confidence", 0.0))

            if class_name == "robot":
                robot_id = self._resolve_robot_id(detection, len(seen_robot_ids))
                seen_robot_ids.add(robot_id)
                self._update_robot(robot_id, bbox, confidence)

            elif class_name in {"orange ball", "ball", "pelota"}:
                ball_seen = True
                self._update_ball(bbox, confidence)

            elif class_name in {"field", "playing field", "cancha", "campo"}:
                field_seen = True
                self._update_field(bbox, confidence)

            elif class_name in {"hand", "human hand", "referee hand", "mano"}:
                self._update_referee_hand(bbox, confidence)

        self._mark_missing_objects(
            seen_robot_ids=seen_robot_ids,
            ball_seen=ball_seen,
            field_seen=field_seen,
        )

        self._update_ball_owner()

    def _resolve_robot_id(
        self,
        detection: dict[str, Any],
        fallback_index: int,
    ) -> str:
        tracking_id = detection.get("tracking_id")

        if tracking_id is not None:
            return f"robot_{tracking_id}"

        return f"robot_{fallback_index}"

    def _update_robot(
        self,
        robot_id: str,
        bbox: BBox,
        confidence: float,
    ) -> None:
        if robot_id not in self.robots:
            self.robots[robot_id] = Robot(
                robot_id=robot_id,
                bbox=bbox,
                confidence=confidence,
            )
        else:
            self.robots[robot_id].update(
                frame_index=self.frame_index,
                bbox=bbox,
                confidence=confidence,
            )

    def _update_ball(self, bbox: BBox, confidence: float) -> None:
        if self.ball is None:
            self.ball = Ball(bbox=bbox, confidence=confidence)

        self.ball.update(
            frame_index=self.frame_index,
            bbox=bbox,
            confidence=confidence,
        )

    def _update_field(self, bbox: BBox, confidence: float) -> None:
        if self.field is None:
            self.field = Field(bbox=bbox, confidence=confidence)
        else:
            self.field.update(bbox=bbox, confidence=confidence)

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
    ) -> None:
        for robot_id, robot in self.robots.items():
            if robot_id not in seen_robot_ids:
                robot.mark_missing()

        if self.ball is not None and not ball_seen:
            self.ball.mark_missing()

        # La cancha puede permanecer como referencia aunque no se detecte cada frame.
        if self.field is not None and not field_seen:
            pass

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