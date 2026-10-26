"""Setup checks: each missing piece gives a plain fix and what still works without it (no Docker or Ollama needed)."""
from pathlib import Path

from nasal import setup_check


def test_without_docker_airflow_is_off_but_the_rest_works(monkeypatch):
    monkeypatch.setattr(setup_check.shutil, "which", lambda name: None)
    docker, image = setup_check.check_docker()
    assert docker["status"] == image["status"] == "missing"
    assert "docker.com" in docker["fix"] and "stored OpenFOAM results still show" in docker["without"]


def test_docker_installed_but_not_running(monkeypatch):
    class Done:
        returncode, stdout = 1, ""

    monkeypatch.setattr(setup_check.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(setup_check.subprocess, "run", lambda *a, **k: Done())
    docker, image = setup_check.check_docker()
    assert docker["detail"] == "installed but not running" and "Start Docker Desktop" in docker["fix"]


def test_unreachable_ollama_is_optional():
    ollama, model = setup_check.check_ollama("http://127.0.0.1:9", timeout=0.5)
    assert ollama["status"] == model["status"] == "optional"
    assert "API key" in ollama["without"]


def test_segmentation_model_and_nasalseg_folders(tmp_path):
    missing = setup_check.check_segment_model(tmp_path / "model.joblib")
    assert missing["status"] == "missing" and "nasal-segment train" in missing["fix"]
    (tmp_path / "model.joblib").write_bytes(b"x")
    assert setup_check.check_segment_model(tmp_path / "model.joblib")["status"] == "ok"
    assert setup_check.check_nasalseg(None, None)["status"] == "optional"
    (tmp_path / "images").mkdir(), (tmp_path / "labels").mkdir()
    for pid in ("P001", "P002"):
        (tmp_path / "images" / f"{pid}_img.nrrd").write_bytes(b"")
    (tmp_path / "labels" / "P001_seg.nrrd").write_bytes(b"")
    found = setup_check.check_nasalseg(tmp_path / "images", tmp_path / "labels")
    assert found["status"] == "ok" and found["detail"] == "1 patients"


def test_markdown_and_summary_name_what_needs_attention():
    items = [setup_check._item("Docker", False, "not installed", fix="install it", without="no airflow runs"),
             setup_check._item("Python packages", True, "all installed"),
             setup_check._item("Ollama (local model)", False, "not reachable", fix="start it", without="no local model",
                               optional=True)]
    assert setup_check.summary(items) == "1 of 3 ready; needs attention: Docker"
    text = setup_check.markdown(items)
    assert "✗ **Docker**: not installed. **Fix:** install it. Without it: no airflow runs" in text
    assert "✓ **Python packages**: all installed" in text and "○ **Ollama (local model)**" in text


def test_checks_list_every_requirement(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_check, "check_docker", lambda: [setup_check._item("Docker", True, "ok"),
                                                               setup_check._item("OpenFOAM image", True, "ok")])
    monkeypatch.setattr(setup_check, "check_ollama", lambda: [setup_check._item("Ollama (local model)", True, "ok"),
                                                               setup_check._item("Local explainer model", True, "ok")])
    names = [i["name"] for i in setup_check.checks(tmp_path / "m.joblib", None, None, Path(tmp_path / "s.csv"))]
    assert names == ["Python packages", "Docker", "OpenFOAM image", "Segmentation model", "Ollama (local model)",
                     "Local explainer model", "NasalSeg patients", "Stored OpenFOAM results"]
