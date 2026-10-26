"""Drawing and agent helpers for the studio: the airway surface with each vertex's region, the 3D resistance map,
model text as Markdown, and the four agents run with their steps shown as they happen."""

import queue
import threading

import numpy as np

from . import agents, anatomy
from .params import REGIONS, SIDES

DEFAULT_PATIENT = "P022"  # one side has three times the other's resistance


def tags_surface(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, step: int = 1) -> dict:
    """Surface (mm) of a cut airway tag volume, with each vertex's '<side>_<region>' name."""
    from skimage import measure

    names = anatomy.region_names(tags)
    padded = np.pad(tags > 0, 1).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, 0.5, step_size=step)
    vox = np.clip(np.rint(verts - 1).astype(int), 0, np.array(tags.shape) - 1)
    # a surface vertex sits between voxels: take the name of the nearest airway voxel around it
    region = names[tuple(vox.T)]
    airway_idx = np.argwhere(tags > 0)
    missing = region == ""
    if missing.any():
        from scipy.spatial import cKDTree

        tree = cKDTree(airway_idx)
        _, nearest = tree.query(verts[missing] - 1)
        region[missing] = names[tuple(airway_idx[nearest].T)]
    return {"verts": origin + (verts - 1) * spacing, "faces": faces, "region": region}


def region_values(record: dict) -> dict[str, float]:
    return {f"{s}_{r}": float(record[f"{s}_{r}_resistance"]) for s in SIDES for r in REGIONS}


def resistance_figure(surface: dict, values: dict[str, float], title: str):
    """Plotly 3D airway colored by region resistance (Pa*s/cm^3)."""
    import plotly.graph_objects as go

    v, f = surface["verts"], surface["faces"]
    intensity = np.array([values.get(name, np.nan) for name in surface["region"]])
    text = [f"{name.replace('_', ' ')}: {values.get(name, float('nan')):.3f} Pa·s/cm³" for name in surface["region"]]
    mesh = go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     intensity=intensity, intensitymode="vertex", colorscale="YlOrRd",
                     cmin=0, cmax=max(values.values()), hovertext=text, hoverinfo="text",
                     colorbar={"title": {"text": "Pa·s/cm³"}, "thickness": 14},
                     lighting={"ambient": 0.55, "diffuse": 0.7, "specular": 0.15}, flatshading=False)
    fig = go.Figure(mesh)
    fig.update_layout(title={"text": title, "font": {"size": 14}}, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                      scene={"aspectmode": "data", "xaxis": {"visible": False}, "yaxis": {"visible": False},
                             "zaxis": {"visible": False},
                             "camera": {"eye": {"x": 1.8, "y": -0.6, "z": 0.3}, "up": {"x": 0, "y": 0, "z": 1}}})
    return fig


def as_markdown(text: str) -> str:
    """Model text shown as Markdown: write units properly and escape stray asterisks
    and underscores, which would otherwise turn "Pa*s/cm^3" into italics."""
    for raw, nice in (("Pa*s/cm^3", "Pa·s/cm³"), ("Pa·s/cm^3", "Pa·s/cm³"), ("cm^3", "cm³"),
                      (" — ", ", "), ("—", ", "), (" – ", ", ")):
        text = text.replace(raw, nice)
    return text.replace("*", "\\*").replace("_", "\\_")


def describe_step(step: dict) -> str | None:
    """One markdown line for an agent log step, or None to skip it."""
    who = step["agent"].capitalize()
    if step["kind"] == "model":
        calls = ", ".join(f"`{c['name']}`" for c in step["tool_calls"]) or (
            "writes the summary" if step["agent"] == "explainer" else "replies without a tool call")
        return f"- **{who}** → {calls} ({step['seconds']:.2f} s)"
    if step["kind"] == "tool" and isinstance(step.get("result"), dict) and "error" in step["result"]:
        return f"  - error sent back to the model: {step['result']['error']}"
    if step["kind"] == "number_check":
        ok = not step["unsupported"] and not step.get("missing")
        return f"  - code number check: {'passed' if ok else 'failed, retrying'}"
    if step["kind"] == "fallback":
        return f"  - code fallback: {step['reason']}"
    return None


def ask_agents(pid, question, patients, flow, history: list[dict] | None = None, **options):
    """Run the four agents on one patient, yielding (steps markdown, answer markdown) as they go.
    With `history` (this session's earlier turns), the new turn is appended to it on success."""
    steps, q = [], queue.Queue()
    result = {}

    def work():
        try:
            result["out"] = agents.run_pipeline(f"Patient {pid}. {question}", patients=patients, patient_flow=flow,
                                                listener=q.put, history=history, **options)
        except Exception as err:  # shown in the UI instead of crashing the app
            result["error"] = f"{type(err).__name__}: {err}"
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    yield "_Running…_", ""
    while True:
        step = q.get()
        if step is None:
            break
        line = describe_step(step)
        if line:
            steps.append(line)
            yield "\n".join(steps), ""
    if "error" in result:
        yield "\n".join(steps), f"**Agents failed:** {agents.explain_error(result['error'], options.get('provider', 'OpenAI'))}"
        return
    out = result["out"]
    if history is not None:
        history.append({"question": f"Patient {pid}. {question}", "answer": out["summary"]})
    flags = "".join(f"\n- {f['text']}" for f in out["facts"]["flags"])
    yield ("\n".join(steps) + f"\n\n_{out['seconds']} s · {out['summary_source']}_",
           f"{as_markdown(out['summary'])}\n\n**Flags**{as_markdown(flags)}")
