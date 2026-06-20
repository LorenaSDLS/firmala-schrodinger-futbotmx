from pathlib import Path

import pytest

from src.H_report.run import _load_template_source, REPORT_VERSION


def test_packaged_template_has_expected_version_marker():
    template = Path(__file__).parents[1] / "src" / "H_report" / "templates" / "infographic.html"
    source = _load_template_source(template)
    assert f'content="{REPORT_VERSION}"' in source
    assert "Estrategia: Ofensiva" not in source
    assert "robot.actividad_text" in source
    assert "assets/zona_control_magenta.png" in source
    assert "events-report-page" in source


def test_old_template_is_rejected(tmp_path: Path):
    old = tmp_path / "infographic.html"
    old.write_text("<html><div>Estrategia: Ofensiva</div></html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="plantilla antigua"):
        _load_template_source(old)
