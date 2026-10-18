"""Nasal0 demo (step 6): a patient's 3D airway colored by resistance, instant what-ifs, and the
Nemotron agents running live.

Run `nasal-demo --labels <NasalSeg labels folder>` after `nasal-patients simulate` and
`nasal-surrogate --patients`. The agents need the vLLM endpoint (NASAL_LLM_URL); the rest
works without it.
"""

import json
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import agents, anatomy
from .params import REGIONS, SIDES

DEFAULT_PATIENT = "P022"  # one side has three times the other's resistance
DEFAULT_QUESTION = "Which side is harder to breathe through, and what if the narrower side's narrowest section were widened by 30%?"


@lru_cache(maxsize=16)
def airway_surface(label_path: str, step: int = 1) -> dict:
    """Airway surface (mm) and each vertex's '<side>_<region>' name for one patient."""
    from skimage import measure

    labels, spacing, origin = anatomy.load_labels(Path(label_path))
    tags, _ = anatomy.airway(labels)
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


def with_widening(row: dict, widen: dict[str, float]) -> dict:
    """Patient geometry with each side's narrowest cross-section changed by a percentage."""
    out = dict(row)
    for side, pct in widen.items():
        out[f"{side}_min_area_mm2"] = row[f"{side}_min_area_mm2"] * (1 + pct / 100)
        out[f"{side}_mean_area_mm2"] = max(row[f"{side}_mean_area_mm2"], out[f"{side}_min_area_mm2"])
    return out


def comparison_table(row: dict, predicted: dict) -> pd.DataFrame:
    """OpenFOAM vs surrogate per side and region, with the surrogate's error."""
    rows = []
    for side in SIDES:
        for key in [f"{side}_{r}_resistance" for r in REGIONS] + [f"{side}_resistance"]:
            cfd, sur = float(row[key]), float(predicted[key])
            rows.append({"where": key.replace("_resistance", "").replace("_", " ") or side,
                         "OpenFOAM": round(cfd, 4), "surrogate": round(sur, 4),
                         "error %": round(100 * (sur / cfd - 1), 1)})
    cfd, sur = float(row["total_resistance"]), float(predicted["total_resistance"])
    rows.append({"where": "total", "OpenFOAM": round(cfd, 4), "surrogate": round(sur, 4),
                 "error %": round(100 * (sur / cfd - 1), 1)})
    return pd.DataFrame(rows)


def as_markdown(text: str) -> str:
    """Model text shown as Markdown: write units properly and escape stray asterisks
    and underscores, which would otherwise turn "Pa*s/cm^3" into italics."""
    for raw, nice in (("Pa*s/cm^3", "Pa·s/cm³"), ("Pa·s/cm^3", "Pa·s/cm³"), ("cm^3", "cm³")):
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


def build_app(labels_dir: Path, sim_path: Path = agents.PATIENTS_PATH, model_path: Path = agents.PATIENT_SURROGATE_PATH):
    import gradio as gr

    patients, surrogate = agents.load_patients(sim_path, model_path)
    surgery = agents.load_surgery()
    if patients is None:
        raise SystemExit(f"need {sim_path} and {model_path}: run nasal-patients simulate and nasal-surrogate --patients")
    ids = list(patients.patient)
    label_file = {f.name.split("_")[0]: str(f) for f in Path(labels_dir).glob("*_seg.nrrd")}
    by_id = {r["patient"]: r for r in patients.to_dict("records")}
    default = DEFAULT_PATIENT if DEFAULT_PATIENT in by_id else ids[0]

    def show(pid, widen_left, widen_right, source):
        row = by_id[pid]
        start = time.perf_counter()
        predicted = surrogate.predict(with_widening(row, {"left": widen_left, "right": widen_right}))
        ms = (time.perf_counter() - start) * 1000
        base = surrogate.predict(row)
        changed = bool(widen_left or widen_right)
        shown = predicted if (source == "Surrogate" or changed) else row
        label = "what-if" if changed else source
        fig = resistance_figure(airway_surface(label_file[pid]), region_values(shown),
                                f"{pid} · {label} · {shown['total_resistance']:.3f} Pa·s/cm³")
        info = (f"**OpenFOAM** {row['total_resistance']:.3f} · L {row['left_resistance']:.3f} · R {row['right_resistance']:.3f}  \n"
                f"**Surrogate** {base['total_resistance']:.3f} "
                f"({100 * (base['total_resistance'] / row['total_resistance'] - 1):+.0f}%) · **{ms:.2f} ms**")
        if changed:
            info += (f"  \n**What-if** L {widen_left:+.0f}% · R {widen_right:+.0f}% → "
                     f"**{predicted['total_resistance']:.3f}** "
                     f"({100 * (predicted['total_resistance'] / base['total_resistance'] - 1):+.1f}%)")
        return fig, comparison_table(row, base), info

    def ask(pid, question):
        steps, q = [], queue.Queue()
        result = {}

        def work():
            try:
                result["out"] = agents.run_pipeline(f"Patient {pid}. {question}", patients=patients,
                                                    patient_surrogate=surrogate, listener=q.put,
                                                    labels_dir=Path(labels_dir), surgery=surgery)
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
            yield "\n".join(steps), f"**Agents failed:** {result['error']} ({agents.DEFAULT_URL})"
            return
        out = result["out"]
        flags = "".join(f"\n- {f['text']}" for f in out["facts"]["flags"])
        yield ("\n".join(steps) + f"\n\n_{out['seconds']} s · {out['summary_source']}_",
               f"{as_markdown(out['summary'])}\n\n**Flags**{as_markdown(flags)}")

    with gr.Blocks(title="Nasal0") as app:
        gr.Markdown("# Nasal0")
        with gr.Row():
            pid = gr.Dropdown(ids, value=default, label="Patient")
            source = gr.Radio(["OpenFOAM", "Surrogate"], value="OpenFOAM", label="Color")
            widen_left = gr.Slider(-30, 60, value=0, step=5, label="Left narrowest (%)")
            widen_right = gr.Slider(-30, 60, value=0, step=5, label="Right narrowest (%)")
        with gr.Row():
            plot = gr.Plot(label="Resistance map")
            with gr.Column():
                info = gr.Markdown()
                table = gr.Dataframe(label="Pa·s/cm³", interactive=False)
        gr.Markdown("### Agents · Nemotron 3.5 Lightning")
        with gr.Row():
            question = gr.Textbox(value=DEFAULT_QUESTION, label="Ask", scale=4)
            go_btn = gr.Button("Run agents", variant="primary", scale=1)
        with gr.Row():
            steps = gr.Markdown(label="Steps")
            answer = gr.Markdown(label="Answer")
        inputs = [pid, widen_left, widen_right, source]
        for comp in inputs:
            comp.change(show, inputs, [plot, table, info])
        app.load(show, inputs, [plot, table, info])
        go_btn.click(ask, [pid, question], [steps, answer])
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Nasal0 demo (Gradio).")
    parser.add_argument("--labels", type=Path, required=True, help="NasalSeg labels folder (*_seg.nrrd)")
    parser.add_argument("--sim", type=Path, default=agents.PATIENTS_PATH, help="patient_sim.csv from nasal-surrogate --patients")
    parser.add_argument("--model", type=Path, default=agents.PATIENT_SURROGATE_PATH)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="also serve a public gradio.live link (72 h)")
    args = parser.parse_args(argv)
    build_app(args.labels, args.sim, args.model).launch(server_port=args.port, server_name="127.0.0.1", share=args.share)


if __name__ == "__main__":
    main()
