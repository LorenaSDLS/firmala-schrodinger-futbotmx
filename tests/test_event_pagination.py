from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

from src.H_report.report_data import build_report_data
from src.H_report.run import REPORT_VERSION, _load_template_source


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_includes_every_event_and_chunks_after_summary_page(tmp_path: Path):
    events = [
        {
            "frame_index": index,
            "timestamp_seconds": index / 10.0,
            "event_type": "robot_inactive_candidate",
            "description": f"Evento {index + 1}",
            "data": {"robot_id": "robot_0"},
        }
        for index in range(41)
    ]
    data = build_report_data(
        _write(tmp_path / "events.json", events),
        _write(tmp_path / "summary.json", {"possession_seconds": {}}),
        _write(tmp_path / "tracks.json", {"robots": {}, "ball": []}),
        max_featured_events=1,  # Legacy option must no longer discard PDF events.
        metadata_path=_write(
            tmp_path / "metadata.json",
            {"video_name": "many-events", "duration_seconds": 5.0},
        ),
        output_directory=tmp_path,
    )

    # 41 analyzed events + the final match marker.
    assert data["event_count"] == 42
    assert len(data["eventos"]) == 42
    assert len(data["event_pages"]) == 4
    assert data["total_page_count"] == 5
    assert data["event_pages"][0]["page_number"] == 2
    assert data["event_pages"][-1]["events"][-1]["event_type"] == "match_end"


def test_template_places_events_only_on_dedicated_pages():
    template_path = (
        Path(__file__).parents[1]
        / "src"
        / "H_report"
        / "templates"
        / "infographic.html"
    )
    source = _load_template_source(template_path)
    assert REPORT_VERSION == "11.5.2"
    assert 'class="report-page summary-report-page"' in source
    assert 'class="report-page events-report-page"' in source
    assert "{% for event_page in event_pages %}" in source
    assert "Todos los eventos analizados" in source
    assert "prefer_css_page_size" not in source
    Environment().parse(source)
