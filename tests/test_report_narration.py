from pathlib import Path
import json
import wave

from src.G_narration.dialogue import generate_dialogue
from src.G_narration.tts import synthesize
from src.H_report.report_data import build_report_data


def test_dialogue_templates_do_not_require_api_key():
    lines, source = generate_dialogue({"event_type": "goal", "description": "Gol", "data": {"scoring_team": "aliado"}}, script_engine="template")
    assert source == "template"
    assert [line.speaker for line in lines] == ["MARTINOLI", "DOCTOR"]
    assert all(line.text for line in lines)


def test_silent_tts_produces_real_wav(tmp_path):
    output = tmp_path / "voice.wav"
    synthesize("Prueba de audio", output, engine="silent")
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 48000
        assert audio.getnframes() > 0


def test_report_data_uses_real_goal_score(tmp_path):
    events = [{"frame_index": 10, "timestamp_seconds": 1.0, "event_type": "goal", "description": "Gol", "data": {"scoring_team": "rival", "confidence": 0.9}}]
    tracks = {"robots": {}, "ball": [{"timestamp_seconds": 2.0, "visible": True}]}
    summary = {"possession_seconds": {}}
    ep, tp, sp, mp = [tmp_path / name for name in ("events.json", "tracks.json", "summary.json", "metadata.json")]
    ep.write_text(json.dumps(events), encoding="utf-8")
    tp.write_text(json.dumps(tracks), encoding="utf-8")
    sp.write_text(json.dumps(summary), encoding="utf-8")
    mp.write_text(json.dumps({"video_name": "test", "duration_seconds": 2.0}), encoding="utf-8")
    data = build_report_data(ep, sp, tp, metadata_path=mp, output_directory=tmp_path)
    assert data["marcador"] == {"aliado": 0, "rival": 1}
    assert any(event["titulo"] == "¡GOL!" for event in data["eventos"])
