"""What the studio needs, whether this computer has it, and how to fix what is missing, in plain words.

`checks()` returns one entry per requirement: Python packages, Docker and the OpenFOAM image (airflow runs),
Ollama and its models (the local model, no API key), the segmentation model (uploaded scans), and the optional
NasalSeg folders and stored results. Each entry says what is wrong, how to fix it, and what still works
without it, so a missing piece turns off one feature instead of the app. Every check is quick (a few seconds
at most) and changes nothing.

    python -m nasal.setup_check          # print the checks
    python -m nasal.setup_check pull     # download the OpenFOAM image (about 1.1 GB)
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .simulate import IMAGE

OLLAMA = "http://127.0.0.1:11434"
STUDIO_PACKAGES = {"gradio": "gradio", "plotly": "plotly", "skimage": "scikit-image", "sklearn": "scikit-learn",
                   "nrrd": "pynrrd", "pydicom": "pydicom", "nibabel": "nibabel", "openai": "openai",
                   "scipy": "scipy", "joblib": "joblib"}
INSTALL = 'pip install -e ".[demo,segment,agents,patients]"'


def _item(name: str, ok: bool, detail: str, fix: str = "", without: str = "", optional: bool = False) -> dict:
    return {"name": name, "status": "ok" if ok else ("optional" if optional else "missing"), "detail": detail,
            "fix": "" if ok else fix, "without": "" if ok else without}


def check_packages() -> dict:
    missing = [pip for mod, pip in STUDIO_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    return _item("Python packages", not missing,
                 "all installed" if not missing else f"missing: {', '.join(missing)}",
                 fix=f"In the project folder run: {INSTALL}", without="the studio may not start")


def check_docker(image: str = IMAGE, timeout: float = 15) -> list[dict]:
    """Docker installed and running, and the OpenFOAM image present (two entries)."""
    docker = shutil.which("docker")
    without = "airflow runs are off; stored OpenFOAM results still show, and airways open and can be operated on"
    if not docker:
        return [_item("Docker", False, "not installed", fix="Install Docker Desktop: https://www.docker.com/products/docker-desktop/",
                      without=without),
                _item("OpenFOAM image", False, "needs Docker first", fix="Install Docker, then run: python -m nasal.setup_check pull",
                      without=without)]
    try:
        info = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                              timeout=timeout)
        running = info.returncode == 0
    except subprocess.TimeoutExpired:
        running = False
    if not running:
        return [_item("Docker", False, "installed but not running", fix="Start Docker Desktop and wait until it says it is running",
                      without=without),
                _item("OpenFOAM image", False, "unknown until Docker runs", fix="Start Docker Desktop, then press Recheck",
                      without=without)]
    try:
        have = subprocess.run([docker, "image", "inspect", image], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        have = False
    return [_item("Docker", True, f"version {info.stdout.strip()}, running"),
            _item("OpenFOAM image", have, image if have else f"{image} not downloaded yet",
                  fix="Run: python -m nasal.setup_check pull (about 1.1 GB; otherwise it downloads on the first airflow run)",
                  without="the first airflow run downloads it first, which takes a few minutes")]


def check_ollama(url: str = OLLAMA, timeout: float = 3) -> list[dict]:
    """Ollama reachable, and the distilled explainer installed in it (two entries; both optional)."""
    from .distill import STUDENT_MODEL

    without = "the local model is off; the cloud agents work with an API key in Settings"
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as reply:
            names = {m["name"].split(":")[0] for m in json.load(reply).get("models", [])}
    except Exception:
        return [_item("Ollama (local model)", False, f"not reachable at {url}", optional=True,
                      fix="Start the nasal-ollama container (see the README, Local explainer), or install Ollama",
                      without=without),
                _item("Local explainer model", False, "needs Ollama first", optional=True,
                      fix="Start Ollama, then create nasal-explainer (see the README, Local explainer)", without=without)]
    return [_item("Ollama (local model)", True, f"running at {url}"),
            _item("Local explainer model", STUDENT_MODEL in names,
                  STUDENT_MODEL if STUDENT_MODEL in names else f"{STUDENT_MODEL} not installed", optional=True,
                  fix=f"Create it: ollama create {STUDENT_MODEL} -q q4_K_M -f scripts/nasal-explainer.Modelfile "
                      "(needs the trained student, see the README)", without=without)]


def check_segment_model(path: Path) -> dict:
    path = Path(path)
    return _item("Segmentation model", path.exists(), str(path) if path.exists() else f"{path} not found",
                 fix="Train it once from NasalSeg: nasal-segment train (see the README, CT to airway)",
                 without="uploaded scans cannot be turned into an airway; NasalSeg expert labels still work")


def check_nasalseg(images_dir: Path | None, labels_dir: Path | None) -> dict:
    if not images_dir or not labels_dir:
        return _item("NasalSeg patients", False, "not given (optional)", optional=True,
                     fix="Start with --images and --labels pointing at NasalSeg, or set NASALSEG_ROOT for the launcher",
                     without="only uploaded scans can be opened")
    n = len(set(f.name.split("_")[0] for f in Path(images_dir).glob("*_img.nrrd"))
            & set(f.name.split("_")[0] for f in Path(labels_dir).glob("*_seg.nrrd")))
    return _item("NasalSeg patients", n > 0, f"{n} patients" if n else "no *_img.nrrd / *_seg.nrrd pairs found",
                 fix="Point --images and --labels at NasalSeg's images/ and labels/ folders", optional=True,
                 without="only uploaded scans can be opened")


def check_stored_results(path: Path) -> dict:
    path = Path(path)
    return _item("Stored OpenFOAM results", path.exists(), str(path) if path.exists() else f"{path} not found",
                 optional=True, fix="Run nasal-patients simulate, or copy dataset/patients/openfoam_results.csv there",
                 without="every airway needs its own airflow run (a few minutes each)")


def checks(segment_model: Path = Path("data/segment_model.joblib"), images_dir: Path | None = None,
           labels_dir: Path | None = None, stored: Path = Path("data/patient_sim.csv")) -> list[dict]:
    return [check_packages(), *check_docker(), check_segment_model(segment_model), *check_ollama(),
            check_nasalseg(images_dir, labels_dir), check_stored_results(stored)]


def summary(items: list[dict]) -> str:
    ready = sum(i["status"] == "ok" for i in items)
    missing = [i["name"] for i in items if i["status"] == "missing"]
    return f"{ready} of {len(items)} ready" + (f"; needs attention: {', '.join(missing)}" if missing else "")


def markdown(items: list[dict]) -> str:
    """The checks as a short list for the studio: what is fine, and for the rest how to fix it."""
    mark = {"ok": "✓", "missing": "✗", "optional": "○"}
    lines = [f"**Setup:** {summary(items)}"]
    for i in items:
        line = f"- {mark[i['status']]} **{i['name']}**: {i['detail']}"
        if i["status"] != "ok":
            line += f". **Fix:** {i['fix']}. Without it: {i['without']}"
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["pull"]:
        docker = shutil.which("docker")
        if not docker:
            sys.exit("Docker is not installed: https://www.docker.com/products/docker-desktop/")
        sys.exit(subprocess.call([docker, "pull", IMAGE]))
    items = checks()
    mark = {"ok": "ok ", "missing": "!! ", "optional": "-- "}
    print(summary(items))
    for i in items:
        print(f"  {mark[i['status']]}{i['name']}: {i['detail']}")
        if i["status"] != "ok":
            print(f"       fix: {i['fix']}\n       without it: {i['without']}")


if __name__ == "__main__":
    main()
