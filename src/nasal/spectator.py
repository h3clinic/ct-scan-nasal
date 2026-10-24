"""Spectator mode: a timeline of every change made to an airway, by you or by the surgeon agent, that can be
replayed step by step.

Each event records who acted, what they did and why (the agent's tool call and its arguments), the airway after
it as a sparse difference from the original (only the changed voxels), and the OpenFOAM result when the step ran
one. `tags_at(step)` rebuilds the airway at any step, `changes_at(step)` gives the voxels that step added and
removed, and `save`/`load` keep a whole session in one .npz file so it can be watched again later.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

SAVE_DIR = Path("data/spectator")
RESULT_KEYS = ("total_resistance", "left_resistance", "right_resistance")


class Timeline:
    def __init__(self, name: str, original: np.ndarray, spacing: np.ndarray, origin: np.ndarray, how: str = ""):
        self.name, self.how = name, how
        self.original = np.asarray(original)
        self.spacing, self.origin = np.asarray(spacing, dtype=float), np.asarray(origin, dtype=float)
        self.events: list[dict] = []
        self.start = time.time()
        self._lock = threading.Lock()  # the surgeon agent records from its own thread

    def add(self, actor: str, kind: str, text: str, tags: np.ndarray | None = None, result: dict | None = None,
            detail: dict | None = None) -> dict:
        """Record one step. `tags` is the airway after it (omit when the step changed nothing), `result` an
        OpenFOAM record and `detail` free JSON (tool arguments, reasons)."""
        event = {"seconds": round(time.time() - self.start, 1), "actor": actor, "kind": kind, "text": text,
                 "detail": detail or {}}
        if tags is not None:
            changed = np.flatnonzero(np.asarray(tags).ravel() != self.original.ravel())
            event["diff"] = (changed.astype(np.int64), np.asarray(tags).ravel()[changed].astype(self.original.dtype))
        if result is not None:
            event["result"] = {k: float(result[k]) for k in result if k.endswith("_resistance") or k == "total_flow_cm3_s"}
        with self._lock:
            event["step"] = len(self.events)
            self.events.append(event)
        return event

    def _diff_at(self, step: int):
        for e in reversed(self.events[:step + 1]):
            if "diff" in e:
                return e["diff"]
        return None

    def tags_at(self, step: int) -> np.ndarray:
        """The airway after `step` (the original before any change)."""
        tags = self.original.copy()
        diff = self._diff_at(step)
        if diff is not None:
            tags.ravel()[diff[0]] = diff[1]
        return tags

    def result_at(self, step: int) -> dict | None:
        """The OpenFOAM result of the airway shown at `step`: the latest one recorded since its last change."""
        for e in reversed(self.events[:step + 1]):
            if "result" in e:
                return e["result"]
            if "diff" in e:  # the airway changed here and was not simulated after
                return None
        return None

    def changes_at(self, step: int, since_original: bool = True) -> dict:
        """Points (mm) added to and removed from the airway by `step`: every change since the original airway, or
        only the step's own change with `since_original=False`."""
        before = self.original if since_original or step == 0 else self.tags_at(step - 1)
        after = self.tags_at(step)
        added = np.argwhere((after > 0) & (before == 0))
        removed = np.argwhere((after == 0) & (before > 0))
        return {"added": self.origin + added * self.spacing, "removed": self.origin + removed * self.spacing}

    def table(self) -> pd.DataFrame:
        rows = [{"step": e["step"], "time s": e["seconds"], "who": e["actor"], "what": e["text"],
                 "total resistance": round(e["result"]["total_resistance"], 4) if "result" in e else ""}
                for e in self.events]
        return pd.DataFrame(rows, columns=["step", "time s", "who", "what", "total resistance"])

    def resistance_series(self) -> tuple[list[int], list[float]]:
        steps = [e["step"] for e in self.events if "result" in e]
        return steps, [self.events[s]["result"]["total_resistance"] for s in steps]

    # ---- files

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or SAVE_DIR / f"{self.name}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.start))}.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        airway = np.flatnonzero(self.original.ravel())
        arrays = {"shape": np.array(self.original.shape), "airway_idx": airway.astype(np.int64),
                  "airway_val": self.original.ravel()[airway], "spacing": self.spacing, "origin": self.origin}
        meta = []
        for e in self.events:
            if "diff" in e:
                arrays[f"diff_idx_{e['step']}"], arrays[f"diff_val_{e['step']}"] = e["diff"]
            meta.append({k: v for k, v in e.items() if k != "diff"} | {"has_diff": "diff" in e})
        arrays["meta"] = np.array(json.dumps({"name": self.name, "how": self.how, "start": self.start, "events": meta}))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        original = np.zeros(int(np.prod(data["shape"])), dtype=data["airway_val"].dtype)
        original[data["airway_idx"]] = data["airway_val"]
        tl = cls(meta["name"], original.reshape(tuple(data["shape"])), data["spacing"], data["origin"], meta["how"])
        tl.start = meta["start"]
        for e in meta["events"]:
            has = e.pop("has_diff")
            if has:
                e["diff"] = (data[f"diff_idx_{e['step']}"], data[f"diff_val_{e['step']}"])
            tl.events.append(e)
        return tl


def saved_timelines(folder: Path = SAVE_DIR) -> list[Path]:
    return sorted(Path(folder).glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)


def detail_markdown(event: dict) -> str:
    """Who did what at a step, with the agent's arguments and the OpenFOAM result."""
    text = f"**Step {event['step']}** · {event['seconds']} s · **{event['actor']}** · {event['text']}"
    args = {k: v for k, v in event.get("detail", {}).items() if v not in (None, "", {}, [])}
    if args:
        text += "  \n" + " · ".join(f"{k}: `{json.dumps(v) if not isinstance(v, str) else v}`" for k, v in args.items())
    if "result" in event:
        r = event["result"]
        text += (f"  \n**OpenFOAM:** total {r['total_resistance']:.4f} · L {r['left_resistance']:.4f} · "
                 f"R {r['right_resistance']:.4f} Pa·s/cm³")
    return text


def series_figure(timeline: Timeline, step: int):
    """Total resistance at every simulated step, with the step being watched marked."""
    import plotly.graph_objects as go

    steps, values = timeline.resistance_series()
    fig = go.Figure(go.Scatter(x=steps, y=values, mode="lines+markers", line={"color": "#e0e0e0"},
                               hovertemplate="step %{x}: %{y:.4f} Pa·s/cm³<extra></extra>"))
    fig.add_vline(x=step, line={"color": "#f0a500"})
    fig.update_layout(height=200, margin={"l": 45, "r": 10, "t": 26, "b": 30},
                      title={"text": "OpenFOAM total resistance by step, Pa·s/cm³", "font": {"size": 11}},
                      xaxis={"title": "step", "range": [-0.5, max(len(timeline.events) - 0.5, 0.5)]})
    return fig
