import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.C_quick_view.temporal_tracker import FutbotTemporalTracker
from src.E_events.event_detector import generate_events


class TemporalTrackerTests(unittest.TestCase):
    def test_robot_ids_survive_shuffled_detections_and_short_occlusion(self):
        random.seed(4)
        tracker = FutbotTemporalTracker(
            fps=60,
            frame_width=1280,
            frame_height=720,
        )

        positions = {
            0: np.array([150.0, 200.0]),
            1: np.array([520.0, 300.0]),
            2: np.array([900.0, 180.0]),
        }
        velocities = {
            0: np.array([2.0, 0.4]),
            1: np.array([-1.2, 0.2]),
            2: np.array([-1.8, 0.8]),
        }
        logical_to_truth: dict[int, list[int]] = {}

        for frame_index in range(150):
            detections = []
            for true_id in positions:
                positions[true_id] += velocities[true_id]
                if true_id == 1 and 55 <= frame_index <= 63:
                    continue

                rng = np.random.default_rng(frame_index * 10 + true_id)
                center = positions[true_id] + rng.normal(0.0, 4.0, 2)
                width, height = 70.0, 85.0
                detections.append({
                    "class_id": 2,
                    "class_name": "robot",
                    "confidence": 0.88,
                    "bbox_xyxy": [
                        center[0] - width / 2,
                        center[1] - height / 2,
                        center[0] + width / 2,
                        center[1] + height / 2,
                    ],
                })

            random.shuffle(detections)
            output = tracker.update(detections)

            for detection in output:
                box = detection["bbox_xyxy"]
                center = np.array([
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2,
                ])
                nearest_truth = min(
                    positions,
                    key=lambda key: np.linalg.norm(center - positions[key]),
                )
                logical_to_truth.setdefault(detection["tracking_id"], []).append(nearest_truth)

        self.assertEqual(len(logical_to_truth), 3)
        for sequence in logical_to_truth.values():
            switches = sum(a != b for a, b in zip(sequence, sequence[1:]))
            self.assertEqual(switches, 0)

    def test_ball_track_is_written_once_per_frame(self):
        records = []
        for frame_index in range(6):
            records.append({
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / 30,
                "detections": [
                    {
                        "class_name": "field",
                        "confidence": 0.9,
                        "bbox_xyxy": [0, 0, 640, 480],
                    },
                    {
                        "class_name": "robot",
                        "confidence": 0.9,
                        "tracking_id": 0,
                        "bbox_xyxy": [100, 100, 150, 160],
                    },
                    {
                        "class_name": "robot",
                        "confidence": 0.9,
                        "tracking_id": 1,
                        "bbox_xyxy": [300, 200, 350, 260],
                    },
                    {
                        "class_name": "orange ball",
                        "confidence": 0.8,
                        "bbox_xyxy": [200, 220, 212, 232],
                    },
                ],
            })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            _, _, tracks = generate_events(path)

        self.assertEqual(len(tracks["ball"]), len(records))


if __name__ == "__main__":
    unittest.main()
