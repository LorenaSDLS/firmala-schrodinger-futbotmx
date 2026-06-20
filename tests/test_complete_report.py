from __future__ import annotations

import json
from pathlib import Path

from src.H_report.report_data import build_report_data
from src.H_report.charts import generate_charts


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _point(frame: int, x: float, y: float, *, team: str = "desconocido", visible: bool = True):
    return {
        "frame_index": frame,
        "timestamp_seconds": frame / 10.0,
        "visible": visible,
        "predicted": False,
        "confidence": 0.9,
        "display_name": "Robot",
        "team": team,
        "field_transform_valid": True,
        "field_x": x,
        "field_y": y,
    }


def test_score_uses_all_goals_not_only_featured_events(tmp_path: Path):
    events = [
        {"frame_index": 10, "timestamp_seconds": 1.0, "event_type": "goal", "description": "Gol 1", "data": {"scoring_team": "aliado", "confidence": 0.9}},
        {"frame_index": 20, "timestamp_seconds": 2.0, "event_type": "goal", "description": "Gol 2", "data": {"scoring_team": "aliado", "confidence": 0.9}},
        {"frame_index": 30, "timestamp_seconds": 3.0, "event_type": "robot_collision_candidate", "description": "Choque", "data": {"robot_a": "robot_0", "robot_b": "robot_1"}},
    ]
    tracks = {"robots": {}, "ball": [{"timestamp_seconds": 4.0, "visible": True}]}
    summary = {"possession_seconds": {}}
    data = build_report_data(
        _write(tmp_path / "events.json", events),
        _write(tmp_path / "summary.json", summary),
        _write(tmp_path / "tracks.json", tracks),
        max_featured_events=1,
        metadata_path=_write(tmp_path / "metadata.json", {"video_name": "test", "duration_seconds": 4.0}),
        output_directory=tmp_path,
    )
    assert data["marcador"] == {"aliado": 2, "rival": 0}


def test_report_uses_team_clustering_and_marks_stationary_robot(tmp_path: Path):
    points = [_point(index, 25.0, 40.0) for index in range(20)]
    for point in points:
        point["display_name"] = "Robot 1"
    tracks = {"robots": {"robot_0": points}, "ball": []}
    _write(tmp_path / "team_clustering.json", {"team_by_id": {"0": "aliado"}})
    data = build_report_data(
        _write(tmp_path / "events.json", []),
        _write(tmp_path / "summary.json", {"possession_seconds": {"robot_0": 0.5}}),
        _write(tmp_path / "tracks.json", tracks),
        metadata_path=_write(tmp_path / "metadata.json", {"video_name": "test", "duration_seconds": 2.0}),
        output_directory=tmp_path,
    )
    robot = data["robots"][0]
    assert robot["equipo"] == "aliado"
    assert robot["estado"] == "SIN MOVIMIENTO"
    assert robot["distancia"] == "0.0 m"
    assert data["equipos"]["aliado"]["posesion"] == "0.5 s"


def test_report_exposes_real_general_statistics_and_free_possession(tmp_path: Path):
    ally_points = [_point(index, 20 + index, 40, team="aliado") for index in range(20)]
    rival_points = [_point(index, 200 - index, 130, team="rival") for index in range(20)]
    ball = [
        {"timestamp_seconds": index / 10.0, "visible": True, "owner_robot_id": "robot_0" if 5 <= index <= 8 else None}
        for index in range(20)
    ]
    tracks = {"robots": {"robot_0": ally_points, "robot_1": rival_points}, "ball": ball}
    data = build_report_data(
        _write(tmp_path / "events.json", []),
        _write(tmp_path / "summary.json", {"possession_seconds": {"robot_0": 0.4}}),
        _write(tmp_path / "tracks.json", tracks),
        metadata_path=_write(tmp_path / "metadata.json", {"video_name": "test", "duration_seconds": 2.0}),
        output_directory=tmp_path,
    )
    names = {item["nombre"] for item in data["estadisticas"]}
    assert {"Posesión confirmada", "Tiempo con balón", "Distancia total", "Velocidad media", "Control de zonas"}.issubset(names)
    assert data["posesion"]["free_seconds"] == 1.6
    assert data["calidad_datos"]["field_coordinate_coverage_percent"] == 100.0


def test_charts_generate_real_files_with_team_assignments(tmp_path: Path):
    tracks = {
        "robots": {
            "robot_0": [_point(index, 20 + index, 50) for index in range(20)],
            "robot_1": [_point(index, 200 - index, 130) for index in range(20)],
        },
        "ball": [
            {"timestamp_seconds": index / 10.0, "visible": True, "owner_robot_id": "robot_0" if index < 5 else None}
            for index in range(20)
        ],
    }
    tracks_path = _write(tmp_path / "tracks.json", tracks)
    output = tmp_path / "assets"
    result = generate_charts(tracks_path, output, team_assignments={"robot_0": "aliado", "robot_1": "rival"})
    for path in result.values():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000
