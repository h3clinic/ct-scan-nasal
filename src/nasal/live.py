"""Live view of a running OpenFOAM patient case: what the solver is doing while it runs, not only its result.

`sampling(tags, spacing, origin, every)` adds a function object that writes, every `every` iterations, the wall
pressure on the whole airway and the air speed on three cutting planes (a sagittal plane through each nasal
cavity and an axial plane through both). `LiveCase(case_dir, max_iterations).poll()` reads what the running case
has written so far: the iteration, the solver residuals, the flow through each nostril (and so the resistance at
this iteration), and the latest sampled fields. These are intermediate states of the steady solver on its way to
the result, not a time-accurate breath.
"""

import re
import shutil
from pathlib import Path

import numpy as np

from .anatomy import INLETS, LABELS
from .resistance import AIR_DENSITY, CM3, REFERENCE_PRESSURE
from .simulate import RAMP_ITERATIONS

LIVE_EVERY = 25  # iterations between sampled fields


def planes(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> list[tuple[str, tuple, tuple]]:
    """(name, point in m, normal) for a sagittal plane through the middle of each cavity and an axial plane at
    the cavities' middle height. Each plane sits a quarter voxel off the cell centres so it cuts cells cleanly."""
    out = []
    cavities = np.argwhere(np.isin(tags, [LABELS["left"], LABELS["right"]]))
    for side in ("left", "right"):
        idx = np.argwhere(tags == LABELS[side])
        if len(idx):
            x = origin[0] + (np.median(idx[:, 0]) + 0.25) * spacing[0]
            out.append((f"sagittal_{side}", (x * 1e-3, 0.0, 0.0), (1, 0, 0)))
    if len(cavities):
        z = origin[2] + (np.median(cavities[:, 2]) + 0.25) * spacing[2]
        out.append(("axial", (0.0, 0.0, z * 1e-3), (0, 0, 1)))
    return out


def sampling(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, every: int = LIVE_EVERY) -> str:
    """controlDict function object writing wall pressure and plane speeds every `every` iterations."""
    cuts = "\n".join(
        f"            {name} {{ type cuttingPlane; point ({p[0]:.8g} {p[1]:.8g} {p[2]:.8g}); "
        f"normal ({n[0]} {n[1]} {n[2]}); interpolate false; }}" for name, p, n in planes(tags, spacing, origin))
    return f"""    live
    {{
        type surfaces;
        libs (sampling);
        writeControl timeStep;
        writeInterval {every};
        surfaceFormat raw;
        interpolationScheme cell;
        fields (p U);
        surfaces
        {{
            wall {{ type patch; patches (wall); interpolate false; }}
{cuts}
        }}
    }}"""


# ---- reading a running case

_TIME = re.compile(r"^Time = (\d+)", re.M)
_RESIDUAL = re.compile(r"Solving for (\w+), Initial residual = ([\d.eE+-]+)")


def parse_log(text: str) -> tuple[int, dict[str, list[tuple[int, float]]]]:
    """(last iteration, {field: [(iteration, initial residual)]}) from simpleFoam's log; the first solve of a
    field in each iteration counts (p is solved twice with a non-orthogonal corrector)."""
    residuals: dict[str, list[tuple[int, float]]] = {}
    iteration = 0
    for block in re.split(r"^(?=Time = \d+)", text, flags=re.M):
        m = _TIME.match(block)
        if not m:
            continue
        iteration = int(m.group(1))
        seen = set()
        for field, value in _RESIDUAL.findall(block):
            if field not in seen:
                seen.add(field)
                residuals.setdefault(field, []).append((iteration, float(value)))
    return iteration, residuals


def read_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(points in mm, values) from an OpenFOAM raw surface file; ValueError while it is still being written."""
    with open(path, encoding="utf-8") as f:
        expected = int(f.readline().split()[-1])
        f.readline()
        data = np.loadtxt(f, ndmin=2)
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} of {expected} rows (still being written)")
    return data[:, :3] * 1e3, data[:, 3:]


def inlet_pressure(iteration: int, max_iterations: int) -> float:
    """Total pressure at the nostrils (Pa) at an iteration: it ramps up from 0 as in `simulate.case_files`."""
    ramp = min(RAMP_ITERATIONS, max_iterations // 3)
    return REFERENCE_PRESSURE * min(1.0, iteration / ramp) if ramp else REFERENCE_PRESSURE


def ramp_end(max_iterations: int) -> int:
    return min(RAMP_ITERATIONS, max_iterations // 3)


class LiveCase:
    """Reads a running (or finished) patient case. `poll()` returns what it has written so far. Sampled field
    folders older than the newest complete one are deleted (unless `keep_samples`), so a run keeps one on disk."""

    def __init__(self, case_dir: Path, max_iterations: int, keep_samples: bool = False):
        self.case_dir, self.max_iterations, self.keep_samples = Path(case_dir), max_iterations, keep_samples
        self.fields: dict | None = None

    def _flows(self) -> dict[str, np.ndarray]:
        out = {}
        for side, inlet in INLETS.items():
            files = list((self.case_dir / "postProcessing" / f"{inlet}Flow").glob("*/surfaceFieldValue.dat"))
            if files:
                rows = [r for r in (line.split() for line in files[0].read_text().splitlines()
                                    if line and not line.startswith("#")) if len(r) == 2]
                out[side] = np.array(rows, dtype=float).reshape(-1, 2)
        return out

    def _latest_fields(self) -> dict | None:
        live = self.case_dir / "postProcessing" / "live"
        if not live.exists():
            return self.fields
        dirs = sorted((d for d in live.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        for d in reversed(dirs):
            if self.fields and self.fields["iteration"] >= int(d.name):
                break
            try:
                points, p = read_raw(d / "p_wall.raw")
                planes_ = {}
                # OpenFOAM writes every surface's p before any U: a sample is complete when each p has its U
                for f in sorted(d.glob("p_*.raw")):
                    if f.stem != "p_wall":
                        pts, u = read_raw(d / f"U_{f.stem[2:]}.raw")
                        planes_[f.stem[2:]] = (pts, np.linalg.norm(u, axis=1))
            except (OSError, ValueError, IndexError):
                continue  # still being written: try the one before
            self.fields = {"iteration": int(d.name), "wall_points": points, "wall_pressure": p[:, 0] * AIR_DENSITY,
                           "planes": planes_}
            break
        if not self.keep_samples and self.fields:
            for d in dirs:
                if int(d.name) < self.fields["iteration"]:
                    shutil.rmtree(d, ignore_errors=True)
        return self.fields

    def poll(self) -> dict:
        log = self.case_dir / "log.simpleFoam"
        iteration, residuals = parse_log(log.read_text(errors="ignore") if log.exists() else "")
        flows, history = self._flows(), None
        if len(flows) == 2 and min(len(flows["left"]), len(flows["right"])):
            n = min(len(flows["left"]), len(flows["right"]))
            it = flows["left"][:n, 0]
            left, right = -flows["left"][:n, 1] / CM3, -flows["right"][:n, 1] / CM3
            p_in = np.array([inlet_pressure(int(i), self.max_iterations) for i in it])
            total = left + right
            with np.errstate(divide="ignore", invalid="ignore"):
                resistance = np.where(total > 0, p_in / total, np.nan)
            history = {"iteration": it, "left_flow": left, "right_flow": right, "total_resistance": resistance}
        return {"iteration": iteration, "max_iterations": self.max_iterations,
                "stage": "solving" if iteration else "meshing", "residuals": residuals, "history": history,
                "fields": self._latest_fields()}


# ---- figures


def wall_figure(surface: dict, snap: dict, title: str):
    """3D airway colored by the latest sampled wall pressure (Pa), with the plane speeds as points inside."""
    import plotly.graph_objects as go
    from scipy.spatial import cKDTree

    v, f = surface["verts"], surface["faces"]
    fields = snap.get("fields")
    mesh = {"x": v[:, 0], "y": v[:, 1], "z": v[:, 2], "i": f[:, 0], "j": f[:, 1], "k": f[:, 2]}
    if not fields:
        fig = go.Figure(go.Mesh3d(**mesh, color="#5a6b78", opacity=0.5, hoverinfo="skip"))
    else:
        _, nearest = cKDTree(fields["wall_points"]).query(v)
        fig = go.Figure(go.Mesh3d(**mesh, intensity=fields["wall_pressure"][nearest], colorscale="Viridis", cmin=0,
                                  cmax=REFERENCE_PRESSURE, opacity=0.35,
                                  colorbar={"title": {"text": "wall p, Pa"}, "len": 0.45, "y": 0.77, "thickness": 10},
                                  hovertemplate="wall pressure %{intensity:.1f} Pa<extra></extra>"))
        speeds = [s for _, s in fields["planes"].values()]
        top = float(np.percentile(np.concatenate(speeds), 99)) if speeds else 1.0
        for k, (name, (pts, speed)) in enumerate(fields["planes"].items()):
            bar = {"title": {"text": "speed, m/s"}, "len": 0.45, "y": 0.25, "thickness": 10} if k == 0 else None
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name.replace("_", " "),
                                       marker={"size": 2, "color": speed, "colorscale": "Inferno", "cmin": 0,
                                               "cmax": top, "colorbar": bar},
                                       hovertemplate="speed %{marker.color:.2f} m/s<extra>" + name + "</extra>"))
    fig.update_layout(title={"text": title, "font": {"size": 14}}, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                      showlegend=False, uirevision="live",
                      scene={"aspectmode": "data", "xaxis": {"visible": False}, "yaxis": {"visible": False},
                             "zaxis": {"visible": False},
                             "camera": {"eye": {"x": 1.8, "y": -0.6, "z": 0.3}, "up": {"x": 0, "y": 0, "z": 1}}})
    return fig


def progress_figure(snap: dict):
    """Resistance at each iteration (from the nostril flows) and the solver residuals, side by side."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Resistance, Pa·s/cm³", "Residuals"))
    h, ramp = snap.get("history"), ramp_end(snap["max_iterations"])
    if h is not None:
        ok = h["iteration"] >= ramp  # before that the inlet pressure is still ramping up from still air
        fig.add_trace(go.Scatter(x=h["iteration"][ok], y=h["total_resistance"][ok], mode="lines", name="resistance",
                                 line={"color": "#e0e0e0"}), row=1, col=1)
    for field, rows in sorted(snap.get("residuals", {}).items()):
        it, value = zip(*rows)
        fig.add_trace(go.Scatter(x=it, y=value, mode="lines", name=field), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(range=[0, snap["max_iterations"]], title_text="iteration")
    fig.update_layout(height=250, margin={"l": 45, "r": 10, "t": 30, "b": 30}, uirevision="live",
                      legend={"orientation": "h", "y": -0.35})
    return fig


def status_text(snap: dict) -> str:
    """'OpenFOAM iteration 340/900 · flow 812 cm³/s · resistance 0.1840 Pa·s/cm³ at this iteration'."""
    if snap["stage"] != "solving":
        return "OpenFOAM: building the mesh and starting the solver…"
    text = f"OpenFOAM iteration {snap['iteration']}/{snap['max_iterations']}"
    h = snap.get("history")
    if h is not None and len(h["iteration"]):
        text += f" · flow {h['left_flow'][-1] + h['right_flow'][-1]:.0f} cm³/s"
        if h["iteration"][-1] >= ramp_end(snap["max_iterations"]) and np.isfinite(h["total_resistance"][-1]):
            text += f" · resistance {h['total_resistance'][-1]:.4f} Pa·s/cm³ at this iteration"
        else:
            text += " · inlet pressure still ramping up"
    return text
