"""Nasal0 studio: CT scan to 3D airway, airflow in OpenFOAM, virtual surgery and the agents in one app.

Run `nasal-studio --images <NasalSeg images> --labels <NasalSeg labels>`. Every resistance shown
comes from OpenFOAM: the stored run where one exists (`data/patient_sim.csv`, `data/patient_surgery.csv`),
otherwise a run started from the app in Docker (several minutes each, cached for the session).
The patient surrogate is not used: on some airways it is off by more than 30%.
Automatic segmentation needs `data/segment_model.joblib` from `nasal-segment train`. The agents
need the vLLM endpoint (NASAL_LLM_URL); the rest works without it.
"""

import itertools
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import agents, anatomy, demo
from .decision import patient as ps
from .params import REGIONS, SIDES

WORK_DIR = Path("data/studio")  # OpenFOAM cases run from the app
PROCEDURE_NAMES = {"Turbinate reduction": "turbinate_reduction", "Valve repair": "valve_repair",
                   "Septoplasty": "septoplasty"}
# App runs stop at 900 iterations instead of 3000: in the 13 stored 3000-iteration patient runs, total flow
# averaged up to iteration 600 was already within 0.11% of the final value, and P022 run at 900 iterations
# on 8 cores matched its stored run within 0.22% in every region (237 s, against about 25 minutes before).
FAST_ITERATIONS = 900
_RUN_ID = itertools.count()


def run_settings(cores: int | None = None, procs: int | None = None) -> tuple[int, int]:
    """(processes per case, cases run at once) for this machine; leaves two cores free."""
    cores = max(1, (cores or os.cpu_count() or 4) - 2)
    procs = procs or min(8, cores)
    return procs, max(1, cores // procs)


def plan_from_choices(choices: list[str]) -> tuple[str, ...]:
    """Checkbox labels -> a plan in `ps.candidate_plans` order (so it matches the OpenFOAM table)."""
    chosen = {PROCEDURE_NAMES[c] for c in choices}
    return tuple(p for p in ps.PATIENT_PROCEDURES if p in chosen)


def changed_voxels(before: np.ndarray, after: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> dict:
    """Points (mm) where surgery added airway and where it removed airway."""
    added = np.argwhere((after > 0) & (before == 0))
    removed = np.argwhere((after == 0) & (before > 0))
    return {"added": origin + added * spacing, "removed": origin + removed * spacing}


def region_table(before: dict, after: dict | None = None) -> pd.DataFrame:
    """OpenFOAM resistance per side and region (Pa*s/cm^3), and the change when an operated result is given."""
    keys = [f"{s}_{r}_resistance" for s in SIDES for r in REGIONS] + [f"{s}_resistance" for s in SIDES] + ["total_resistance"]
    rows = []
    for key in keys:
        row = {"where": key.replace("_resistance", "").replace("_", " "), "before": round(float(before[key]), 4)}
        if after is not None:
            row["after"] = round(float(after[key]), 4)
            row["change %"] = round(100 * (after[key] / before[key] - 1), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plan_ranking(results: dict[str, dict], standard: tuple) -> pd.DataFrame:
    """Every plan with OpenFOAM's resistance drop where it has been run, best first, and the standard rule's pick.

    `results` maps plan labels ('none', 'turbinate_reduction+valve_repair', ...) to OpenFOAM records."""
    before = results.get("none")
    rows = []
    for plan in ps.candidate_plans():
        name = ps.label(plan)
        r = results.get(name)
        rows.append({"plan": name.replace("_", " ").replace("+", " + "),
                     "OpenFOAM drop %": round(100 * (1 - r["total_resistance"] / before["total_resistance"]), 1)
                     if (r is not None and before is not None) else np.nan,
                     "standard rule": "picks" if plan == standard else ""})
    table = pd.DataFrame(rows).sort_values("OpenFOAM drop %", ascending=False, na_position="last", ignore_index=True)
    table["OpenFOAM drop %"] = table["OpenFOAM drop %"].astype(object).where(table["OpenFOAM drop %"].notna(), "not run")
    return table


def airway_figure(surface: dict, title: str, result: dict | None = None, changes: dict | None = None):
    """3D airway colored by OpenFOAM region resistance when a result is given, else plain; surgery changes as markers."""
    import plotly.graph_objects as go

    if result is not None:
        fig = demo.resistance_figure(surface, demo.region_values(result), title)
    else:
        v, f = surface["verts"], surface["faces"]
        fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#9ecae1",
                                  hovertext=[n.replace("_", " ") for n in surface["region"]], hoverinfo="text",
                                  lighting={"ambient": 0.55, "diffuse": 0.7, "specular": 0.15}))
        fig.update_layout(title={"text": title, "font": {"size": 14}}, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                          scene={"aspectmode": "data", "xaxis": {"visible": False}, "yaxis": {"visible": False},
                                 "zaxis": {"visible": False},
                                 "camera": {"eye": {"x": 1.8, "y": -0.6, "z": 0.3}, "up": {"x": 0, "y": 0, "z": 1}}})
    for key, color in (("added", "#1a9850"), ("removed", "#d73027")):
        pts = (changes or {}).get(key, [])
        if len(pts):
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=f"{key} ({len(pts)} voxels)",
                                       marker={"size": 2, "color": color}, hoverinfo="name"))
    if changes:
        fig.update_layout(showlegend=True, legend={"x": 0, "y": 1})
    return fig


def run_openfoam(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict, case_dir: Path,
                 procs: int = 8, max_iterations: int = FAST_ITERATIONS) -> dict:
    """Mesh the airway, run simpleFoam in Docker and read the result (a few minutes)."""
    from .simulate import run_case

    geometry = anatomy.write_airway_case(case_dir, tags, spacing, origin, cuts, procs, max_iterations=max_iterations)
    run_case(case_dir)
    return {**geometry, **anatomy.read_patient(case_dir, max_iterations=max_iterations)}


def summary(result: dict) -> str:
    return (f"total {result['total_resistance']:.4f} · L {result['left_resistance']:.4f} · "
            f"R {result['right_resistance']:.4f} Pa·s/cm³")


# Layout and look follow the VOCO desktop app (github.com/h3clinic/VOCO): a neutral dark workbench with a title
# bar, a patient sidebar, the airway in the main pane, commands on the right and the agents as the console.
STUDIO_CSS = """
:root, .dark { --canvas:#1b1b1b; --chrome:#171717; --panel:#202020; --hover:#292929; --line:#303030;
  --line-soft:#292929; --text:#d0d0d0; --muted:#9a9a9a; --faint:#707070; }
body, gradio-app, .gradio-container { background:var(--canvas) !important; color:var(--text);
  font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; font-size:13px; }
.gradio-container, .gradio-container main, .gradio-container .contain, .gradio-container .fillable,
.gradio-container .app { max-width:none !important; padding:0 !important; margin:0 !important; }
.gradio-container > .main, .gradio-container .wrap { gap:0 !important; }
footer { display:none !important; }
.block, .form, .panel, .gr-group { background:transparent !important; border:none !important; box-shadow:none !important; border-radius:0 !important; }
#window-bar { height:38px; display:flex; align-items:center; gap:8px; padding:0 12px; background:var(--chrome);
  border-bottom:1px solid var(--line); font-size:12px; }
#window-bar .mark { display:grid; place-items:center; width:22px; height:22px; border:1px solid #555; border-radius:3px;
  font-size:9px; font-weight:700; color:#b8b8b8; }
#window-bar strong { color:#eee; letter-spacing:.02em; }
#window-bar .sub { color:var(--faint); font-size:10px; margin-left:6px; }
#shell { gap:0 !important; min-height:calc(100vh - 38px); flex-wrap:nowrap !important; }
#sidebar { background:var(--chrome) !important; border-right:1px solid var(--line) !important; padding:0 8px 10px !important;
  gap:6px !important; flex:0 0 258px !important; max-width:258px; min-width:258px !important; }
.section-head { display:flex; justify-content:space-between; align-items:center; height:38px; padding:0 4px 0 6px;
  color:#a0a0a0; font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.section-head .count { color:var(--muted); font-weight:500; letter-spacing:0; }
[data-testid="block-info"] { color:var(--muted) !important; font-size:10px !important; font-weight:650 !important;
  text-transform:uppercase; letter-spacing:.05em; }
label, .wrap label span, .prose, .info, td, th { font-size:11px !important; }
#workspace { gap:0 !important; min-width:0; }
#topbar { min-height:44px; display:flex; justify-content:flex-end; align-items:center; gap:8px; padding:0 12px;
  background:var(--chrome); border-bottom:1px solid var(--line); color:var(--muted); font-size:10px; }
#topbar p, #topbar * { margin:0; font-size:10px !important; color:var(--muted) !important; }
.pane-title { display:flex; align-items:center; justify-content:space-between; min-height:36px; padding:0 12px;
  background:var(--chrome); border-bottom:1px solid var(--line-soft); color:#d0d0d0; font-size:10px; font-weight:650; }
#session { gap:0 !important; flex-wrap:nowrap !important; }
#monitor { min-width:0; gap:0 !important; }
#controls { background:var(--panel) !important; border-left:1px solid var(--line) !important; padding:0 10px 12px !important;
  gap:6px !important; flex:0 0 244px !important; max-width:244px; min-width:244px !important; }
#console { border-top:1px solid var(--line) !important; gap:0 !important; }
.info { padding:8px 12px !important; font-size:11px; color:var(--text); }
.info p { margin:0 0 4px; }
button.primary, button.secondary, button.signal { border-radius:3px !important; min-height:32px; font-size:11px !important;
  font-weight:550 !important; box-shadow:none !important; text-align:left; justify-content:flex-start !important; padding:5px 9px !important; }
button.primary { background:#dedede !important; border:1px solid #dedede !important; color:#181818 !important; font-weight:650 !important; }
button.primary:hover { background:#fff !important; border-color:#fff !important; }
button.secondary, button.signal { background:#252525 !important; border:1px solid #464646 !important; color:#d0d0d0 !important; }
button.secondary:hover, button.signal:hover { background:#303030 !important; border-color:#666 !important; }
input:not([type=radio]):not([type=checkbox]), textarea, select, .wrap-inner, .secondary-wrap { background:#1b1b1b !important; border-color:#464646 !important;
  color:#d8d8d8 !important; border-radius:3px !important; font-size:11px !important; }
.table-wrap, table { background:#1b1b1b !important; font-size:10px !important; }
table th { background:#171717 !important; color:var(--muted) !important; font-weight:650 !important; }
table td, table th { border-color:#292929 !important; color:#d0d0d0 !important; padding:4px 8px !important; }
.messages { min-height:150px; max-height:300px; overflow-y:auto; background:#1b1b1b; }
.msg { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:18px; padding:7px 12px; border-bottom:1px solid #292929;
  font-size:11px; line-height:1.45; color:#d0d0d0; }
.msg.you, .msg.agents { border-left:2px solid #777; background:#202020; }
.msg.system { color:var(--muted); }
.msg .meta { color:var(--muted); font-size:9px; white-space:nowrap; }
.msg p { margin:0 0 4px; } .msg ul { margin:2px 0 2px 16px; padding:0; }
#composer { background:var(--chrome) !important; border-top:1px solid var(--line) !important; padding:6px 10px !important;
  gap:6px !important; align-items:center; }
.note { color:var(--faint); font-size:9px; padding:0 12px 6px; }
.html-container { padding:0 !important; }
@media (max-width: 1000px) {
  #session { flex-wrap:wrap !important; }
  #controls { flex:1 1 100% !important; max-width:none; min-width:0 !important; border-left:none !important;
    border-top:1px solid var(--line) !important; }
}
@media (max-width: 760px) {
  #shell { flex-wrap:wrap !important; }
  #sidebar { flex:1 1 100% !important; max-width:none; min-width:0 !important; border-right:none !important;
    border-bottom:1px solid var(--line) !important; }
  #composer { flex-wrap:wrap !important; }
  #window-bar .sub { display:none; }
}
"""


def plot_theme(fig):
    """Dark Plotly layout that matches the studio's workbench colors."""
    fig.update_layout(paper_bgcolor="#1b1b1b", plot_bgcolor="#1b1b1b", font={"color": "#d0d0d0", "size": 11},
                      title={"font": {"size": 12, "color": "#d0d0d0"}}, legend={"font": {"color": "#d0d0d0"}})
    return fig


def console_html(entries: list[dict]) -> str:
    """Console log in VOCO's message style: your question, agent steps and the agents' answer."""
    import html as html_lib

    from markdown_it import MarkdownIt

    md = MarkdownIt()
    rows = [f'<div class="msg {e["role"]}"><div>{md.render(e["text"]) if e["role"] != "system" else html_lib.escape(e["text"])}</div>'
            f'<span class="meta">{html_lib.escape(e.get("meta", ""))}</span></div>' for e in entries]
    return '<div class="messages">' + ("".join(rows) or '<div class="msg system"><div>Ask the agents about the airway or a surgery plan.</div></div>') + "</div>"


def build_app(images_dir: Path, labels_dir: Path, sim_path: Path = agents.PATIENTS_PATH,
              model_path: Path = agents.PATIENT_SURROGATE_PATH, segment_model: Path | None = None,
              procs: int | None = None, iterations: int = FAST_ITERATIONS):
    import gradio as gr

    from . import segment as sg

    patients, surrogate = agents.load_patients(sim_path, model_path)  # the surrogate is only passed on to the agents
    surgery = agents.load_surgery()
    segment_model = Path(segment_model or sg.MODEL_PATH)
    stored_none = {r["patient"]: r for r in patients.to_dict("records")} if patients is not None else {}
    images = {f.name.split("_")[0]: f for f in Path(images_dir).glob("*_img.nrrd")}
    labels = {f.name.split("_")[0]: f for f in Path(labels_dir).glob("*_seg.nrrd")}
    ids = sorted(set(images) & set(labels))
    default = demo.DEFAULT_PATIENT if demo.DEFAULT_PATIENT in ids else ids[0]
    procs, parallel = run_settings(procs=procs)
    session_runs: dict[tuple[str, str], dict] = {}  # (airway key, plan label) -> OpenFOAM record run from the app
    ready = f"OpenFOAM local · {procs} cores × {parallel} cases · {iterations} iterations"

    def openfoam_results(st) -> dict[str, dict]:
        """Plan label -> OpenFOAM record for this airway: stored runs (expert labels only) plus runs made here."""
        out = {}
        if st["expert"]:
            if st["name"] in stored_none:
                out["none"] = stored_none[st["name"]]
            if surgery is not None:
                out.update({r["plan"]: r for r in surgery[surgery.patient == st["name"]].to_dict("records")})
        out.update({plan: r for (key, plan), r in session_runs.items() if key == st["key"]})
        return out

    def load_airway(pid, upload, source):
        """-> state dict with the cut airway, or raises gr.Error."""
        if upload is not None:
            if not segment_model.exists():
                raise gr.Error(f"an uploaded CT needs the segmentation model: run nasal-segment train (writes {segment_model})")
            import joblib

            ct, spacing, origin = sg.load_ct(Path(upload))
            lab = sg.segment(ct, spacing, joblib.load(segment_model))
            name, how = Path(upload).name.split(".")[0], "segmented from the uploaded CT"
        elif source == "Automatic (from the CT)":
            if not segment_model.exists():
                raise gr.Error(f"automatic segmentation needs {segment_model}: run nasal-segment train")
            import joblib

            ct, spacing, origin = sg.load_ct(images[pid])
            lab = sg.segment(ct, spacing, joblib.load(segment_model))
            name, how = pid, "segmented automatically from the CT (no expert labels)"
        else:
            lab, spacing, origin = anatomy.load_labels(labels[pid], like=images[pid])
            name, how = pid, "NasalSeg expert labels"
        try:
            tags, cuts = anatomy.airway(lab)
        except ValueError as err:
            raise gr.Error(f"{name}: {err}") from err
        return {"name": name, "how": how, "key": f"{name}:{how}", "tags": tags, "cuts": cuts, "spacing": spacing,
                "origin": origin, "expert": how == "NasalSeg expert labels"}

    def original_view(st):
        """(plot, region table, ranking, info) for the untouched airway."""
        geometry = anatomy.measure(st["tags"], st["spacing"])
        results = openfoam_results(st)
        result = results.get("none")
        surface = demo.tags_surface(st["tags"], st["spacing"], st["origin"])
        title = f"{st['name']} · OpenFOAM · {result['total_resistance']:.3f} Pa·s/cm³" if result else f"{st['name']} · not simulated yet"
        info = (f"**{st['name']}**, {st['how']}: {int((st['tags'] > 0).sum()):,} airway voxels, "
                f"left {geometry['left_volume_cm3']:.1f} cm³, right {geometry['right_volume_cm3']:.1f} cm³.  \n")
        info += (f"**OpenFOAM** (air at 150 Pa): {summary(result)}" if result else
                 "No OpenFOAM run for this airway yet: **Run airflow**.")
        standard = ps.standard_rule_patient(geometry)
        return (plot_theme(airway_figure(surface, title, result)), region_table(result) if result else None,
                plan_ranking(results, standard), info)

    def reconstruct(pid, upload, source):
        st = load_airway(pid, upload, source)
        return (st, *original_view(st), ready)

    def run_cases(st, jobs: list[tuple[str, np.ndarray]]):
        """Generator: runs (plan label, tags) cases in parallel, yielding progress text; the last item
        is None when every case finished, or an error string."""
        from concurrent.futures import ThreadPoolExecutor

        errors = {}

        def work(label, tags):
            case_dir = WORK_DIR / st["name"] / f"{label}_{next(_RUN_ID)}"
            try:
                session_runs[(st["key"], label)] = run_openfoam(tags, st["spacing"], st["origin"], st["cuts"], case_dir,
                                                                procs, iterations)
            except Exception as err:  # shown in the UI
                errors[label] = f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"

        start = time.time()
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(work, label, tags) for label, tags in jobs]
            while not all(f.done() for f in futures):
                done = sum(f.done() for f in futures)
                yield f"OpenFOAM running · {done}/{len(jobs)} cases done · {int(time.time() - start)} s"
                time.sleep(3)
        if errors:
            yield "OpenFOAM failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()) + " (is Docker Desktop running?)"
            return
        yield None

    def simulate_original(st):
        if not st:
            raise gr.Error("open an airway first")
        msg = ready
        if "none" not in openfoam_results(st):
            for msg in run_cases(st, [("none", st["tags"])]):
                if msg is not None:
                    yield (*([gr.update()] * 4), msg)
            msg = msg or ready
        yield (*original_view(st), msg)

    def surgery_view(st, plan):
        tags, cuts, spacing = st["tags"], st["cuts"], st["spacing"]
        operated = ps.apply_patient_plan(tags, cuts, spacing, plan)
        standard = ps.standard_rule_patient(anatomy.measure(tags, spacing))
        changes = changed_voxels(tags, operated, spacing, st["origin"])
        results = openfoam_results(st)
        before, after = results.get("none"), results.get(ps.label(plan))
        surface = demo.tags_surface(operated, spacing, st["origin"])
        name = ps.label(plan).replace("_", " ").replace("+", " + ")
        if before and after:
            drop = 100 * (1 - after["total_resistance"] / before["total_resistance"])
            title = f"{st['name']} · {name} · OpenFOAM {after['total_resistance']:.3f} Pa·s/cm³ ({abs(drop):.1f}% {'lower' if drop >= 0 else 'higher'})"
        else:
            title = f"{st['name']} · {name} · not simulated yet"
        info = (f"**Plan:** {name}: {len(changes['added']):,} voxels added, {len(changes['removed']):,} removed "
                f"(one layer, about 0.6 mm).  \n**Standard rule** picks {ps.label(standard).replace('_', ' ')} "
                f"(placeholder threshold, not ENT-reviewed).  \n")
        if before and after:
            info += f"**OpenFOAM:** {before['total_resistance']:.4f} → {after['total_resistance']:.4f} Pa·s/cm³."
        else:
            missing = [n for n, r in (("original airway", before), ("this plan", after)) if r is None]
            info += f"Not simulated yet: {' and '.join(missing)}. **Run airflow on plan**."
        table = region_table(before, after) if (before and after) else None
        return ({**st, "operated": operated, "plan": plan},
                plot_theme(airway_figure(surface, title, after, changes)), table, plan_ranking(results, standard), info)

    def operate(st, choices):
        if not st:
            raise gr.Error("open an airway first")
        return surgery_view(st, plan_from_choices(choices))

    def simulate_plans(st, choices, every: bool):
        """OpenFOAM on the original airway and the chosen plan (or every plan), skipping what has results."""
        if not st:
            raise gr.Error("open an airway first")
        plan = plan_from_choices(choices)
        view = surgery_view(st, plan)
        yield (*view, ready)
        st = view[0]
        plans = ps.candidate_plans() if every else [plan]
        jobs = [("none", st["tags"])] + [(ps.label(p), st["operated"] if p == plan else
                                          ps.apply_patient_plan(st["tags"], st["cuts"], st["spacing"], p)) for p in plans if p]
        have = openfoam_results(st)
        jobs = [(label, tags) for label, tags in jobs if label not in have]
        msg = ready
        if jobs:
            for msg in run_cases(st, jobs):
                if msg is not None:
                    yield (*([gr.update()] * 5), msg)
            msg = msg or ready
        yield (*surgery_view(st, plan), msg)

    def plan_once(st, choices):
        yield from simulate_plans(st, choices, False)

    def plan_all(st, choices):
        yield from simulate_plans(st, choices, True)

    def ask(pid, question, provider, memory, log):
        backend = "openai" if provider.startswith("OpenAI") else "local"
        if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise gr.Error("set the OPENAI_API_KEY environment variable before starting nasal-studio")
        if not question.strip():
            raise gr.Error("type a question first")
        log = [*log, {"role": "you", "text": f"{pid} · {question}", "meta": "you"}]
        steps = {"role": "system", "text": "Running…", "meta": provider}
        for steps_md, answer_md in demo.ask_agents(pid, question, patients, surrogate, labels_dir, surgery,
                                                   provider=backend, history=memory):
            steps["text"] = " · ".join(line.strip("-_ ").replace("**", "").replace("`", "")
                                       for line in steps_md.splitlines() if line.strip()) or "Running…"
            shown = [*log, steps] + ([{"role": "agents", "text": answer_md, "meta": "agents"}] if answer_md else [])
            yield console_html(shown), shown, memory, "", f"{len(memory)} earlier question(s) in memory"
        # the final yield above already holds the answer; keep it as the new log

    with gr.Blocks(title="Nasal0", fill_width=True) as app:
        gr.HTML('<div id="window-bar"><span class="mark">N0</span><strong>NASAL0</strong>'
                '<span class="sub">CT → airway → OpenFOAM → virtual surgery → agents</span></div>')
        state = gr.State({})
        memory = gr.State([])
        log = gr.State([])
        with gr.Row(elem_id="shell"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=258):
                gr.HTML(f'<div class="section-head"><span>Patients</span><span class="count">{len(ids)}</span></div>')
                pid = gr.Dropdown(ids, value=default, label="NasalSeg CT", container=True)
                source = gr.Radio(["Expert labels", "Automatic (from the CT)"], value="Expert labels", label="Airway from")
                build = gr.Button("Open airway", variant="primary", elem_classes="signal")
                gr.HTML('<div class="section-head"><span>Upload</span></div>')
                upload = gr.File(label="CT volume (.nrrd, HU)", file_types=[".nrrd"], height=90)
                with gr.Accordion("ABOUT THE MODEL", open=False):
                    gr.Markdown("Rigid walls, one steady breath at 150 Pa, voxel walls. Every resistance comes "
                                "from OpenFOAM. Decision thresholds are placeholders until an ENT reviews them.",
                                elem_classes="note")
            with gr.Column(elem_id="workspace", scale=1):
                status = gr.Markdown(ready, elem_id="topbar")
                with gr.Row(elem_id="session"):
                    with gr.Column(elem_id="monitor", scale=1):
                        gr.HTML('<div class="pane-title"><span>Airway</span><span>OpenFOAM resistance, Pa·s/cm³</span></div>')
                        plot = gr.Plot(show_label=False)
                        info = gr.Markdown(elem_classes="info")
                        with gr.Row():
                            table = gr.Dataframe(label="Regions (OpenFOAM)", interactive=False)
                            ranking = gr.Dataframe(label="All 7 plans (OpenFOAM)", interactive=False)
                    with gr.Column(elem_id="controls", scale=0, min_width=244):
                        gr.HTML('<div class="section-head"><span>Commands</span></div>')
                        air_btn = gr.Button("Run airflow", elem_classes="signal")
                        procedures = gr.CheckboxGroup(list(PROCEDURE_NAMES), value=["Turbinate reduction", "Valve repair"],
                                                      label="Procedures")
                        apply_btn = gr.Button("Apply plan", elem_classes="signal")
                        sim_btn = gr.Button("Run airflow on plan", elem_classes="signal")
                        all_btn = gr.Button("Run all 7 plans", elem_classes="signal")
                        reset_btn = gr.Button("Show original airway", elem_classes="signal")
                with gr.Column(elem_id="console"):
                    gr.HTML('<div class="pane-title"><span>Console · agents</span><span>memory stays in this session</span></div>')
                    messages = gr.HTML(console_html([]))
                    with gr.Row(elem_id="composer"):
                        question = gr.Textbox(placeholder="Ask the agents, e.g. which plan lowers resistance most and why?",
                                              show_label=False, scale=5, container=False)
                        provider = gr.Dropdown(["Local Nemotron (vLLM)", f"OpenAI API ({agents.OPENAI_MODEL})"],
                                               value=f"OpenAI API ({agents.OPENAI_MODEL})" if os.environ.get("OPENAI_API_KEY")
                                               else "Local Nemotron (vLLM)", show_label=False, scale=2, container=False)
                        send = gr.Button("Send", variant="primary", scale=0, min_width=70)
                        clear = gr.Button("Clear", scale=0, min_width=70, elem_classes="signal")
                    memory_info = gr.Markdown("Only the patient id, your question and computed numbers are sent to the "
                                              "model, never the CT. OpenAI requests use store=False; OpenAI may still keep "
                                              "them up to 30 days for abuse monitoring unless your organization has zero "
                                              "data retention.", elem_classes="note")

        view_out = [plot, table, ranking, info]
        build.click(reconstruct, [pid, upload, source], [state, *view_out, status])
        app.load(reconstruct, [pid, upload, source], [state, *view_out, status])
        air_btn.click(simulate_original, [state], [*view_out, status])
        reset_btn.click(lambda st: original_view(st) if st else [gr.update()] * 4, [state], view_out)
        apply_btn.click(operate, [state, procedures], [state, *view_out])
        sim_btn.click(plan_once, [state, procedures], [state, *view_out, status])
        all_btn.click(plan_all, [state, procedures], [state, *view_out, status])
        for trigger in (send.click, question.submit):
            trigger(ask, [pid, question, provider, memory, log], [messages, log, memory, question, memory_info])
        clear.click(lambda: (console_html([]), [], [], "Session memory cleared."), None, [messages, log, memory, memory_info])
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import gradio as gr

    parser = argparse.ArgumentParser(description="Nasal0 studio (Gradio): CT to airway, OpenFOAM airflow, virtual surgery, agents.")
    parser.add_argument("--images", type=Path, required=True, help="NasalSeg images folder (*_img.nrrd)")
    parser.add_argument("--labels", type=Path, required=True, help="NasalSeg labels folder (*_seg.nrrd)")
    parser.add_argument("--sim", type=Path, default=agents.PATIENTS_PATH, help="stored OpenFOAM runs of the untouched airways")
    parser.add_argument("--model", type=Path, default=agents.PATIENT_SURROGATE_PATH, help="patient surrogate, used only by the agents")
    parser.add_argument("--segment-model", type=Path, default=None, help="default: data/segment_model.joblib")
    parser.add_argument("--procs", type=int, default=None, help="cores per OpenFOAM case (default: up to 8)")
    parser.add_argument("--iterations", type=int, default=FAST_ITERATIONS, help="solver iterations per app run")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    app = build_app(args.images, args.labels, args.sim, args.model, args.segment_model, args.procs, args.iterations)
    app.launch(server_port=args.port, server_name="127.0.0.1", css=STUDIO_CSS,
               theme=gr.themes.Base(neutral_hue="neutral", text_size="sm", spacing_size="sm", radius_size="sm"), js="() => document.body.classList.add('dark')")


if __name__ == "__main__":
    main()
