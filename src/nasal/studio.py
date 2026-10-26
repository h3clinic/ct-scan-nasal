"""Nasal0 studio: CT scan to 3D airway, airflow in OpenFOAM, virtual surgery and the agents in one app.

Run `nasal-studio --images <NasalSeg images> --labels <NasalSeg labels>`. Every resistance shown
comes from OpenFOAM: the stored run where one exists (`data/patient_sim.csv`, `data/patient_surgery.csv`),
otherwise a run started from the app in Docker (a few minutes each, kept for the session and shared with
the agents through `nasal.patient_flow`). Runs started here are drawn live while OpenFOAM works (`nasal.live`),
and the Spectator tab replays every change made to the airway (`nasal.spectator`). The patient surrogate is not
used: on some airways it is off by more than 30%. Automatic segmentation needs `data/segment_model.joblib` from `nasal-segment train`. The
agents run on OpenAI's API with the key added in Settings (or OPENAI_API_KEY); the rest works without it.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import agents, anatomy, distill, live, providers, settings, setup_check, spectator, surgeon, view
from .decision import patient as ps
from .params import REGIONS, SIDES
from .patient_flow import _RUN_ID, FAST_ITERATIONS, WORK_DIR, PatientFlow, run_openfoam

PROCEDURE_NAMES = {"Turbinate reduction": "turbinate_reduction", "Valve repair": "valve_repair",
                   "Septoplasty": "septoplasty"}


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


def plan_ranking(results: dict[str, dict]) -> pd.DataFrame:
    """Every plan with OpenFOAM's resistance drop where it has been run, best first.

    `results` maps plan labels ('none', 'turbinate_reduction+valve_repair', ...) to OpenFOAM records."""
    before = results.get("none")
    rows = []
    for plan in ps.candidate_plans():
        name = ps.label(plan)
        r = results.get(name)
        rows.append({"plan": name.replace("_", " ").replace("+", " + "),
                     "OpenFOAM drop %": round(100 * (1 - r["total_resistance"] / before["total_resistance"]), 1)
                     if (r is not None and before is not None) else np.nan})
    table = pd.DataFrame(rows).sort_values("OpenFOAM drop %", ascending=False, na_position="last", ignore_index=True)
    table["OpenFOAM drop %"] = table["OpenFOAM drop %"].astype(object).where(table["OpenFOAM drop %"].notna(), "not run")
    return table


def airway_figure(surface: dict, title: str, result: dict | None = None, changes: dict | None = None):
    """3D airway colored by OpenFOAM region resistance when a result is given, else plain; surgery changes as markers."""
    import plotly.graph_objects as go

    if result is not None:
        fig = view.resistance_figure(surface, view.region_values(result), title)
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


def import_text(report: dict) -> str:
    """One line on how an uploaded scan was read: format, slices, voxel size and the crop to the nose."""
    sp = report["spacing_mm"]
    text = (f"**Scan:** {report['format']}, {report['slices']} slices, voxels {sp[0]:g} × {sp[1]:g} × {sp[2]:g} mm")
    if report.get("series_found", 1) > 1:
        text += f" (the largest of {report['series_found']} series)"
    if report.get("cropped"):
        w, d, h = report["size_mm"]
        text += (f"; full-head scan cropped to {w:.0f} × {d:.0f} × {h:.0f} mm around the nose (from the nose tip "
                 f"and the hard palate, found {report['landmarks']['palate_search']}). Check that the airway looks complete")
    return text + "."


def surgeon_label(op: dict) -> str:
    """'valve repair (both, 1.2 mm)' for an operation the surgeon agent applied (or undid); depths given in
    voxel layers read '(both, 2 layers)'."""
    if "undo" in op:
        return "undo " + surgeon_label(op["undo"])
    where = op.get("region", "")
    extra = f" {where} {op.get('direction', '')}".rstrip() if where else ""
    if "depth_mm" in op:
        size = f"{op['depth_mm']:g} mm"
    else:
        depth = op.get("depth", 1)
        size = f"{depth} layer{'s' if depth > 1 else ''}"
    side = "" if op["technique"] == "septoplasty" else f"{op.get('side', 'both')}, "
    return f"{op['technique'].replace('_', ' ')}{extra} ({side}{size})"


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
#status, #status * { margin:0; font-size:10px !important; color:var(--muted) !important; text-align:right; }
#topbar { flex-wrap:nowrap !important; }
#setup-banner { background:#2a2414 !important; border-bottom:1px solid #6b5a2a !important; padding:8px 12px !important;
  color:#e8d9a8 !important; font-size:11px; }
#setup-banner p { margin:0; }
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
#settings, #settings > div { background:var(--chrome) !important; border-left:1px solid var(--line) !important; }
#settings .section-head { padding-left:0; }
#model-label { flex:0 0 auto !important; min-width:0; padding:0 4px !important; white-space:nowrap; }
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


def make_actions(images_dir: Path | None = None, labels_dir: Path | None = None, sim_path: Path = agents.PATIENTS_PATH,
                 segment_model: Path | None = None, procs: int | None = None, iterations: int = FAST_ITERATIONS) -> dict:
    """Everything the studio can do, as functions shared by the Gradio layout and other front ends.

    Returns a dict of handlers, plus `ids` (NasalSeg patients), `default` (the patient to open first) and `flow`
    (the OpenFOAM source shared with the agents). Gradio state the handlers expect: `st` (the open airway, a dict;
    start with {}), `memory` and `log` (lists; the console), `prefs` (dict; API key and model), and for the
    spectator `sp` = {"step": 0, "playing": False, "loaded": None, "drawn": None}. What each one was wired to:

    - reconstruct(pid, upload) -> (st, plot, region table, plan ranking, info, status): open a NasalSeg patient, or
      an uploaded CT (DICOM files/zip, NIfTI, NRRD; segmented automatically, full-head scans cropped to the nose)
    - simulate_original(st) -> generator of (plot, region table, plan ranking, info, status, live chart): OpenFOAM
      on the open airway, streamed live (skipped when a stored run exists)
    - show_original(st) -> (plot, region table, plan ranking, info)
    - plan_once(st, procedure labels) / plan_all(st, procedure labels) -> generator of
      (st, plot, region table, plan ranking, info, status, live chart): OpenFOAM on the ticked plan, or all 7
    - agent_operate(st, goal, prefs, log) -> generator of
      (plot, region table, plan ranking, info, status, console html, log, live chart): the surgeon agent
    - ask(pid, question, memory, log, prefs) -> generator of (console html, log, memory, question box, memory note):
      the local model when installed, otherwise the cloud agents with an API key
    - spectate(st, sp, step=None, follow=False, advance=0, force=False), tick(st, sp, follow), toggle_play(st, sp)
      -> (sp, plot, detail, step slider, resistance chart, timeline table, play button label);
      save_timeline(st, sp) -> (note, saved list); load_timeline(name, st, sp) and watch_current(st, sp)
      -> (the seven above, follow checkbox, note)
    - load_prefs() -> (prefs, key status, model label, provider, model, API address); set_key(key, remember,
      provider, model, address, prefs) -> (prefs, key status, key box, model label, provider, model);
      check_key(prefs) -> key status; drop_key(prefs) -> (prefs, key status, model label);
      pick_provider(name, prefs) -> (prefs, model, model label, key status); set_pref(name) -> f(value, prefs) -> prefs;
      model_text(prefs) -> model label
    - run_setup_checks() -> (setup list markdown, banner update): `nasal.setup_check` with plain fixes
    - console_html(entries) (module level) draws the console; PROCEDURE_NAMES lists the procedure labels
    """
    import gradio as gr  # handlers use gr.update and gr.Error

    from . import segment as sg

    patients = agents.load_patient_table(sim_path)
    surgery = agents.load_surgery()
    # one OpenFOAM source for the studio and the agents: stored runs plus live runs made in this app
    flow = PatientFlow(patients, surgery, labels_dir, live=True, procs=procs, iterations=iterations)
    segment_model = Path(segment_model or sg.MODEL_PATH)
    # the NasalSeg folders are optional: without them the studio works from uploaded scans only
    images = {f.name.split("_")[0]: f for f in Path(images_dir).glob("*_img.nrrd")} if images_dir else {}
    labels = {f.name.split("_")[0]: f for f in Path(labels_dir).glob("*_seg.nrrd")} if labels_dir else {}
    ids = sorted(set(images) & set(labels))
    default = view.DEFAULT_PATIENT if view.DEFAULT_PATIENT in ids else (ids[0] if ids else None)
    session_runs: dict[tuple[str, str], dict] = {}  # runs on airways that are not NasalSeg expert labels

    def ready_text() -> str:
        return f"OpenFOAM local · {flow.procs} cores × {flow.parallel} cases · {flow.iterations} iterations"

    def openfoam_results(st) -> dict[str, dict]:
        """Plan label -> OpenFOAM record for this airway: stored runs (expert labels only) plus runs made here."""
        if st["expert"]:
            labels_ = ["none", *(ps.label(p) for p in ps.candidate_plans())]
            return {label: r for label in labels_ if (r := flow.known(st["name"], label)) is not None}
        return {plan: r for (key, plan), r in session_runs.items() if key == st["key"]}

    def load_airway(pid, upload):
        """-> state dict with the cut airway, or raises gr.Error."""
        imported = None
        if upload:
            if not segment_model.exists():
                raise gr.Error(f"an uploaded CT needs the segmentation model: run nasal-segment train (writes {segment_model})")
            import joblib

            from . import ct_import

            paths = [Path(f) for f in (upload if isinstance(upload, list) else [upload])]
            try:
                ct, spacing, origin, imported = ct_import.prepare(paths)
            except ValueError as err:
                raise gr.Error(f"could not read the scan: {err}") from err
            lab = sg.segment(ct, spacing, joblib.load(segment_model))
            first = paths[0].name
            name = (first[:-7] if first.lower().endswith(".nii.gz") else paths[0].stem) if len(paths) == 1 else "uploaded CT"
            how = f"segmented from the uploaded CT ({imported['format']}, {imported['slices']} slices)"
        elif pid not in images:
            raise gr.Error("upload a CT scan (DICOM, NIfTI or NRRD), or start the studio with --images and --labels")
        else:
            lab, spacing, origin = anatomy.load_labels(labels[pid], like=images[pid])
            name, how = pid, "NasalSeg expert labels"
        try:
            tags, cuts = anatomy.airway(lab)
        except ValueError as err:
            raise gr.Error(f"{name}: {err}") from err
        return {"name": name, "how": how, "key": f"{name}:{how}", "tags": tags, "cuts": cuts, "spacing": spacing,
                "origin": origin, "expert": how == "NasalSeg expert labels", "imported": imported,
                "timeline": spectator.Timeline(name, tags, spacing, origin, how)}

    def original_view(st):
        """(plot, region table, ranking, info) for the untouched airway."""
        geometry = anatomy.measure(st["tags"], st["spacing"])
        results = openfoam_results(st)
        result = results.get("none")
        surface = view.tags_surface(st["tags"], st["spacing"], st["origin"])
        title = f"{st['name']} · OpenFOAM · {result['total_resistance']:.3f} Pa·s/cm³" if result else f"{st['name']} · not simulated yet"
        info = (f"**{st['name']}**, {st['how']}: {int((st['tags'] > 0).sum()):,} airway voxels, "
                f"left {geometry['left_volume_cm3']:.1f} cm³, right {geometry['right_volume_cm3']:.1f} cm³.  \n")
        info += (f"**OpenFOAM** (air at 150 Pa): {summary(result)}" if result else
                 "No OpenFOAM run for this airway yet: **Run airflow**.")
        if st.get("imported"):
            info += "  \n" + import_text(st["imported"])
        return (plot_theme(airway_figure(surface, title, result)), region_table(result) if result else None,
                plan_ranking(results), info)

    def reconstruct(pid, upload):
        if not upload and pid not in images:  # no NasalSeg folders and nothing uploaded yet
            return ({}, gr.update(), gr.update(), gr.update(),
                    "**Upload a CT scan to start:** DICOM (a .zip of the series or the slice files), NIfTI or NRRD.",
                    ready_text())
        st = load_airway(pid, upload)
        st["timeline"].add("you", "open", f"opened {st['name']} ({st['how']})", result=openfoam_results(st).get("none"))
        return (st, *original_view(st), ready_text())

    live_surfaces: dict[tuple, dict] = {}  # coarse surfaces for the live view, per airway and plan

    def live_view(st, label, tags, snap, shown: dict):
        """(airway figure or no update, progress chart, status) for a live snapshot. The 3D figure is resent
        only when a new field sample arrived (every live.LIVE_EVERY iterations); `shown` remembers the last."""
        key = (st["key"], label)
        if key not in live_surfaces:
            live_surfaces[key] = view.tags_surface(tags, st["spacing"], st["origin"], step=2)
        sample = (label, (snap.get("fields") or {}).get("iteration"))
        name = label.replace("_", " ").replace("+", " + ")
        figure = gr.update()
        if sample != shown.get("sample"):
            shown["sample"] = sample
            title = f"{st['name']} · {name} · live OpenFOAM, iteration {sample[1] or 0}: wall pressure and air speed"
            figure = plot_theme(live.wall_figure(live_surfaces[key], snap, title))
        return figure, plot_theme(live.progress_figure(snap)), f"{name}: {live.status_text(snap)}"

    def run_cases(st, jobs: list[tuple[str, np.ndarray]], actor: str = "you"):
        """Generator: runs (plan label, tags) cases in parallel. Yields (status text, label, live snapshot) while
        they run, the case shown being the first one still running; the last item is (None, None, None) when
        every case finished, or (error text, None, None). Each start and result goes on the airway's timeline."""
        from concurrent.futures import ThreadPoolExecutor

        errors, snaps, finished = {}, {}, set()
        timeline = st["timeline"]

        def work(label, tags):
            case_dir = WORK_DIR / st["name"] / f"{label}_{next(_RUN_ID)}"
            name = label.replace("_", " ").replace("+", " + ")
            timeline.add(actor, "simulate", f"OpenFOAM started: {name}", tags=tags)
            try:
                record = run_openfoam(tags, st["spacing"], st["origin"], st["cuts"], case_dir, flow.procs, flow.iterations,
                                      on_progress=lambda snap: snaps.__setitem__(label, snap))
                if st["expert"]:  # shared with the agents
                    flow.runs[(st["name"], label)] = {**record, "source": f"OpenFOAM run now ({flow.iterations} iterations)"}
                else:
                    session_runs[(st["key"], label)] = record
                timeline.add(actor, "result", f"OpenFOAM result: {name}", tags=tags, result=record)
            except Exception as err:  # shown in the UI
                errors[label] = f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"
                timeline.add(actor, "result", f"OpenFOAM failed: {name}", detail={"error": errors[label]})
            finished.add(label)

        start = time.time()
        with ThreadPoolExecutor(max_workers=flow.parallel) as pool:
            futures = [pool.submit(work, label, tags) for label, tags in jobs]
            while not all(f.done() for f in futures):
                focus = next((label for label, _ in jobs if label in snaps and label not in finished), None)
                msg = f"OpenFOAM running · {len(finished)}/{len(jobs)} cases done · {int(time.time() - start)} s"
                yield msg, focus, snaps.get(focus)
                time.sleep(2)
        if errors:
            yield "OpenFOAM failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()) + " (is Docker Desktop running?)", None, None
            return
        yield None, None, None

    def run_live(st, jobs, wrap):
        """Runs `jobs` and yields wrap(airway figure, status, live chart) frames while they run (no update where
        nothing changed). Returns the last status text: None when every case finished, else the error."""
        shown, tags_of = {}, dict(jobs)
        for msg, label, snap in run_cases(st, jobs):
            if msg is None:
                return None
            if snap is not None:
                figure, chart, text = live_view(st, label, tags_of[label], snap, shown)
                yield wrap(figure, f"{msg} · {text}", chart)
            else:
                yield wrap(gr.update(), msg, gr.update())
            if msg.startswith("OpenFOAM failed"):
                return msg
        return None

    def simulate_original(st):
        if not st:
            raise gr.Error("open an airway first")
        msg = ready_text()
        if "none" not in openfoam_results(st):
            msg = yield from run_live(st, [("none", st["tags"])],
                                      lambda fig, text, chart: (fig, gr.update(), gr.update(), gr.update(), text, chart))
            msg = msg or ready_text()
        yield (*original_view(st), msg, gr.update())

    def surgery_view(st, plan):
        tags, cuts, spacing = st["tags"], st["cuts"], st["spacing"]
        operated = ps.apply_patient_plan(tags, cuts, spacing, plan)
        changes = changed_voxels(tags, operated, spacing, st["origin"])
        results = openfoam_results(st)
        before, after = results.get("none"), results.get(ps.label(plan))
        surface = view.tags_surface(operated, spacing, st["origin"])
        name = ps.label(plan).replace("_", " ").replace("+", " + ")
        if before and after:
            drop = 100 * (1 - after["total_resistance"] / before["total_resistance"])
            title = f"{st['name']} · {name} · OpenFOAM {after['total_resistance']:.3f} Pa·s/cm³ ({abs(drop):.1f}% {'lower' if drop >= 0 else 'higher'})"
        else:
            title = f"{st['name']} · {name} · not simulated yet"
        vol = ps.edited_volume(tags, operated, spacing)
        info = (f"**Plan:** {name}: airway added {vol['added_mm3']:,.0f} mm³, removed {vol['removed_mm3']:,.0f} mm³ "
                f"(one voxel layer per procedure).  \n")
        if before and after:
            info += f"**OpenFOAM:** {before['total_resistance']:.4f} → {after['total_resistance']:.4f} Pa·s/cm³."
        else:
            missing = [n for n, r in (("original airway", before), ("this plan", after)) if r is None]
            info += f"Not simulated yet: {' and '.join(missing)}. **Run airflow on plan**."
        table = region_table(before, after) if (before and after) else None
        return ({**st, "operated": operated, "plan": plan},
                plot_theme(airway_figure(surface, title, after, changes)), table, plan_ranking(results), info)

    def show_original(st):
        if not st:
            return [gr.update()] * 4
        st["timeline"].add("you", "view", "back to the original airway", tags=st["tags"],
                           result=openfoam_results(st).get("none"))
        return original_view(st)

    def simulate_plans(st, choices, every: bool):
        """OpenFOAM on the original airway and the chosen plan (or every plan), skipping what has results."""
        if not st:
            raise gr.Error("open an airway first")
        plan = plan_from_choices(choices)
        plan_view = surgery_view(st, plan)
        yield (*plan_view, ready_text(), gr.update())
        st = plan_view[0]
        plans = ps.candidate_plans() if every else [plan]
        jobs = [("none", st["tags"])] + [(ps.label(p), st["operated"] if p == plan else
                                          ps.apply_patient_plan(st["tags"], st["cuts"], st["spacing"], p)) for p in plans if p]
        have = openfoam_results(st)
        jobs = [(label, tags) for label, tags in jobs if label not in have]
        msg = ready_text()
        if jobs:
            msg = yield from run_live(st, jobs, lambda fig, text, chart: (gr.update(), fig, *([gr.update()] * 3), text, chart))
            msg = msg or ready_text()
        yield (*surgery_view(st, plan), msg, gr.update())

    def plan_once(st, choices):
        yield from simulate_plans(st, choices, False)

    def plan_all(st, choices):
        yield from simulate_plans(st, choices, True)

    def llm(prefs) -> dict:
        """The language model this browser session uses: provider, key, model, address; raises if no key."""
        name = prefs.get("provider") or providers.DEFAULT
        spec = providers.PROVIDERS[name]
        key = prefs.get("api_key") or (os.environ.get("OPENAI_API_KEY") if name == "OpenAI" else None)
        if not key:
            raise gr.Error("add an API key in Settings (top right); the surgeon agent needs a language model")
        model = prefs.get("model") or (agents.OPENAI_MODEL if name == "OpenAI" else spec.default_model)
        base_url = prefs.get("base_url") or spec.base_url
        if not base_url:
            raise gr.Error("a custom provider needs its API address in Settings")
        return {"provider": name, "api_key": key, "model": model, "base_url": base_url}

    def local_ready() -> bool:
        return setup_check.check_ollama(timeout=1.5)[1]["status"] == "ok"

    def has_key(prefs) -> bool:
        return bool(prefs.get("api_key") or ((prefs.get("provider") or providers.DEFAULT) == "OpenAI"
                                             and os.environ.get("OPENAI_API_KEY")))

    def ask(pid, question, memory, log, prefs):
        """The local distilled model answers when it is installed (nothing leaves this computer); otherwise the
        cloud agents answer with the API key from Settings."""
        if not question.strip():
            raise gr.Error("type a question first")
        log = [*log, {"role": "you", "text": f"{pid} · {question}", "meta": "you"}]
        if local_ready() or not has_key(prefs):
            yield from ask_local(pid, question, memory, log)
            return
        model = llm(prefs)
        steps = {"role": "system", "text": "Running…", "meta": f"{model['provider']} · {model['model']}"}
        history = memory
        for steps_md, answer_md in view.ask_agents(pid, question, patients, flow, history=history, **model):
            steps["text"] = " · ".join(line.strip("-_ ").replace("**", "").replace("`", "")
                                       for line in steps_md.splitlines() if line.strip()) or "Running…"
            shown = [*log, steps] + ([{"role": "agents", "text": answer_md, "meta": "agents"}] if answer_md else [])
            yield console_html(shown), shown, memory, "", f"{len(memory)} earlier question(s) in memory"
        # the final yield above already holds the answer; keep it as the new log

    def agent_operate(st, goal, prefs, log):
        """The surgeon agent operates on the open airway; every cut and every OpenFOAM result is drawn as it happens."""
        import queue
        import threading

        from openai import OpenAI

        if not st:
            raise gr.Error("open an airway first")
        model = llm(prefs)
        events = queue.Queue()
        baseline = openfoam_results(st).get("none")
        timeline, actor = st["timeline"], "surgeon agent"
        timeline.add(actor, "start", f"surgeon agent started ({model['provider']} · {model['model']})",
                     tags=st["tags"], result=baseline, detail={"goal": goal})
        # saved from the agent's thread, so the run can be replayed (Watch saved) even if the page reconnects
        save_to = spectator.SAVE_DIR / f"{st['name']}_agent_{time.strftime('%Y%m%d-%H%M%S')}.npz"

        def simulate_fn(tags):
            case_dir = WORK_DIR / st["name"] / f"agent_{next(_RUN_ID)}"
            return run_openfoam(tags, st["spacing"], st["origin"], st["cuts"], case_dir, flow.procs, flow.iterations,
                                on_progress=lambda snap: events.put(("live", None, snap)))

        def on_session(event, data):
            """Every change the agent makes, for the spectator timeline, then for the live view."""
            if event == "operate":
                op = data["operation"]
                timeline.add(actor, "operate", surgeon_label(op), tags=session.tags.copy(), detail={"operation": op})
            elif event == "simulating":
                timeline.add(actor, "simulate", "OpenFOAM started on the agent's airway", tags=session.tags.copy())
            elif event == "simulated":
                timeline.add(actor, "result", "OpenFOAM result on the agent's airway", tags=session.tags.copy(),
                             result=data["record"])
                timeline.save(save_to)
            elif event == "finished":
                timeline.add(actor, "finish", "finished surgery", tags=session.tags.copy(),
                             result=session.simulated.get(session.key()), detail={"kept": data["result"]["operations"]})
                timeline.save(save_to)
            events.put(("session", event, data))

        def on_step(step):
            """The agent's words and its tool calls that change nothing (or were rejected) on the timeline."""
            if step["kind"] == "model" and (step.get("content") or "").strip():
                timeline.add(actor, "says", step["content"].strip()[:400])
            elif step["kind"] == "tool":
                rejected = isinstance(step.get("result"), dict) and "error" in step["result"]
                if rejected or step["name"] not in ("operate", "undo", "simulate", "finish_surgery"):
                    args = step.get("arguments")
                    timeline.add(actor, "tool", step["name"] + (f" (rejected: {step['result']['error']})" if rejected else ""),
                                 detail={"arguments": json.loads(args) if isinstance(args, str) and args.strip() else args})
            events.put(("step", step["kind"], step))

        session = surgeon.SurgerySession(st["name"], st["tags"], st["cuts"], st["spacing"], st["origin"], simulate_fn,
                                         baseline=baseline, listener=on_session)
        client = OpenAI(api_key=model["api_key"], base_url=model["base_url"])
        out = {}

        def work():
            try:
                out["result"] = surgeon.run_surgeon(session, goal or "lower the resistance with the least surgery", client,
                                                    model["model"], model["provider"], listener=on_step)
            except Exception as err:  # shown in the console
                out["error"] = agents.explain_error(f"{type(err).__name__}: {err}", model["provider"])
                timeline.add(actor, "error", f"surgeon agent failed: {out['error']}")
            else:
                timeline.add(actor, "summary", out["result"]["summary"][:400], detail={"source": out["result"]["summary_source"]})
            timeline.save(save_to)
            events.put(("done", None, None))

        threading.Thread(target=work, daemon=True).start()
        log = [*log, {"role": "you", "text": f"{st['name']} · surgeon agent · {goal}", "meta": "you"}]
        lines = []
        steps = {"role": "system", "text": "The surgeon agent is starting…", "meta": f"{model['provider']} · {model['model']}"}

        def frame(status_text):
            before = session.simulated.get(session.key([]))
            after = session.simulated.get(session.key())
            ops = ", ".join(surgeon_label(h["operation"]) for h in session.history) or "no operation yet"
            changes = changed_voxels(session.original, session.tags, st["spacing"], st["origin"])
            surface = view.tags_surface(session.tags, st["spacing"], st["origin"])
            title = f"{st['name']} · agent: {ops}" + (f" · OpenFOAM {after['total_resistance']:.3f} Pa·s/cm³" if after else " · not simulated")
            table = region_table(before, after) if (before and after and session.history) else (region_table(after) if after else None)
            vol = ps.edited_volume(session.original, session.tags, st["spacing"])
            info = (f"**Surgeon agent:** {ops}. Airway added {vol['added_mm3']:,.0f} mm³, removed "
                    f"{vol['removed_mm3']:,.0f} mm³. "
                    f"OpenFOAM runs: {session.runs}.")
            steps["text"] = " · ".join(lines[-8:]) or steps["text"]
            shown = [*log, steps]
            return (plot_theme(airway_figure(surface, title, after, changes)), table, gr.update(), info, status_text,
                    console_html(shown), shown, gr.update())

        yield frame("Surgeon agent working…")
        live_shown, n_sim = {}, 0
        while True:
            source, kind, data = events.get()
            if source == "done":
                break
            if source == "step" and kind == "tool":
                lines.append(f"{data['name']}" + (" (rejected)" if isinstance(data.get("result"), dict) and "error" in data["result"] else ""))
                continue
            if source == "live":
                figure, chart, text = live_view(st, f"agent_check_{n_sim}", session.tags, data, live_shown)
                yield (figure, gr.update(), gr.update(), gr.update(), f"Surgeon agent's airway · {text}",
                       gr.update(), gr.update(), chart)
            elif source == "session" and kind == "simulating":
                n_sim += 1
                yield frame(f"OpenFOAM running on the agent's airway ({flow.procs} cores, {flow.iterations} iterations)…")
            elif source == "session" and kind in ("operate", "simulated", "finished"):
                yield frame("Surgeon agent working…")
        if "error" in out:
            final = {"role": "agents", "text": f"**Surgeon agent failed:** {out['error']}", "meta": "surgeon"}
        else:
            res = out["result"]
            final = {"role": "agents", "text": view.as_markdown(res["summary"]) + f"\n\n_{res['seconds']} s · {res['summary_source']} · "
                                                f"timeline saved as {save_to.name} (Spectator → Watch saved)_", "meta": "surgeon"}
        *outputs, _, _, shown, _ = frame(ready_text())
        shown = [*shown, final]
        yield (*outputs, ready_text(), console_html(shown), shown, gr.update())

    def ask_local(pid, question, memory, log):
        """Code picks the tools, the distilled student writes the answer; nothing leaves this computer."""
        steps = {"role": "system", "text": "Running the local model…", "meta": distill.STUDENT_MODEL}
        yield console_html([*log, steps]), [*log, steps], memory, "", "Local model: nothing is sent off this computer."
        try:
            out = distill.answer_locally(question, pid, patients, flow)
        except Exception as err:  # Ollama not running or the model not installed
            answer = {"role": "agents", "text": f"**No model to answer:** the local model is not reachable "
                                                f"({type(err).__name__}) and no API key is set. See ⚙ Settings → Setup "
                                                f"for Ollama, or add a key there.", "meta": "local"}
        else:
            steps["text"] = " · ".join(out["steps"]) + f" · {out['seconds']} s · {out['summary_source']}"
            answer = {"role": "agents", "text": view.as_markdown(out["summary"]), "meta": "local"}
        shown = [*log, steps, answer]
        yield console_html(shown), shown, memory, "", "Local model: nothing is sent off this computer."


    def watched(st, sp):
        return sp.get("loaded") or (st or {}).get("timeline")

    def spectate(st, sp, step=None, follow=False, advance=0, force=False):
        """Draw step `step` of the watched timeline; only what changed is sent to the browser."""
        tl = watched(st, sp)
        if tl is None or not tl.events:
            return (sp, *([gr.update()] * 6))
        last = len(tl.events) - 1
        if follow and not sp.get("loaded"):
            step = last
        step = min(max(0, int(sp["step"] if step is None else step) + advance), last)
        playing = sp["playing"] and step < last
        key = (id(tl), step, len(tl.events))
        sp = {**sp, "step": step, "playing": playing}
        if key == sp.get("drawn") and not force:
            return (sp, *([gr.update()] * 6))
        redraw = (id(tl), step) != (sp.get("drawn") or (None, None, None))[:2] or force
        sp["drawn"] = key
        event = tl.events[step]
        figure = gr.update()
        if redraw:
            tags, result = tl.tags_at(step), tl.result_at(step)
            surface = view.tags_surface(tags, tl.spacing, tl.origin)
            title = (f"{tl.name} · step {step} · {event['actor']}: {event['text'][:60]}"
                     + (f" · OpenFOAM {result['total_resistance']:.3f} Pa·s/cm³" if result else ""))
            figure = plot_theme(airway_figure(surface, title, result, tl.changes_at(step)))
        return (sp, figure, spectator.detail_markdown(event), gr.update(maximum=max(last, 1), value=step),
                plot_theme(spectator.series_figure(tl, step)), tl.table(), "❚❚ Pause" if playing else "▶ Play")

    def tick(st, sp, follow):
        if sp.get("playing"):
            return spectate(st, sp, advance=1)
        return spectate(st, sp, follow=follow)

    def toggle_play(st, sp):
        tl = watched(st, sp)
        at_end = tl is not None and sp["step"] >= len(tl.events) - 1
        sp = {**sp, "playing": not sp["playing"]}
        return spectate(st, sp, step=0 if (sp["playing"] and at_end) else None, force=True)

    def save_timeline(st, sp):
        tl = watched(st, sp)
        if tl is None or not tl.events:
            raise gr.Error("nothing to save yet")
        path = tl.save()
        return f"Saved {path}.", gr.update(choices=[p.name for p in spectator.saved_timelines()], value=path.name)

    def load_timeline(name, st, sp):
        if not name:
            raise gr.Error("pick a saved session")
        sp = {**sp, "loaded": spectator.Timeline.load(spectator.SAVE_DIR / name), "playing": False, "drawn": None}
        return (*spectate(st, sp, step=0, force=True), False, f"Watching {name} (saved).")

    def watch_current(st, sp):
        sp = {**sp, "loaded": None, "drawn": None}
        return (*spectate(st, sp, follow=True, force=True), True, "Watching this airway's session.")

    def default_model(name: str) -> str:
        return agents.OPENAI_MODEL if name == "OpenAI" else providers.PROVIDERS[name].default_model

    def key_text(prefs) -> str:
        name = prefs.get("provider") or providers.DEFAULT
        if prefs.get("api_key"):
            where = "saved on this computer" if prefs.get("saved") else "for this session"
            return f"{name} key {settings.mask(prefs['api_key'])}, {where}."
        if name == "OpenAI" and os.environ.get("OPENAI_API_KEY"):
            return f"OpenAI key {settings.mask(os.environ['OPENAI_API_KEY'])} from the OPENAI_API_KEY variable."
        return "No API key yet: paste one below and press Use key."

    def model_text(prefs) -> str:
        if local_ready():
            return f"Answers: local {distill.STUDENT_MODEL}"
        name = prefs.get("provider") or providers.DEFAULT
        if prefs.get("api_key") or (name == "OpenAI" and os.environ.get("OPENAI_API_KEY")):
            return f"{name} · {prefs.get('model') or default_model(name)}"
        return "No key: open ⚙ Settings"

    def load_prefs():
        saved = settings.load_saved() or {}
        name = saved.get("provider") or providers.DEFAULT
        prefs = {"api_key": saved.get("key"), "saved": bool(saved), "provider": name, "base_url": saved.get("base_url"),
                 "model": default_model(name)}
        return prefs, key_text(prefs), model_text(prefs), name, prefs["model"], saved.get("base_url") or ""

    def set_key(text, remember, chosen, model, base_url, prefs):
        key = (text or "").strip()
        custom = chosen == "Custom (OpenAI-compatible)"
        if not key or any(c.isspace() for c in key) or (not custom and not settings.looks_like_key(key)):
            raise gr.Error("That does not look like an API key.")  # a local custom server may take any word
        detected = providers.detect(key)
        name = chosen if detected == "Custom (OpenAI-compatible)" else detected
        if model in ("", default_model(prefs.get("provider") or providers.DEFAULT)):
            model = default_model(name)  # follow the provider unless the user typed a model
        saved = remember and settings.save(key, name, base_url or None)
        prefs = {**prefs, "api_key": key, "saved": bool(saved), "provider": name, "model": model,
                 "base_url": base_url or None}
        note = "" if saved or not remember else " Could not save it on this computer; it is kept for this session."
        return prefs, key_text(prefs) + note, "", model_text(prefs), name, model

    def check_key(prefs):
        key = prefs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise gr.Error("add a key first")
        return key_text(prefs) + " " + settings.test_key(key, prefs.get("provider") or providers.DEFAULT,
                                                         prefs.get("base_url"))[1]

    def drop_key(prefs):
        settings.forget_saved()
        prefs = {**prefs, "api_key": None, "saved": False}
        return prefs, key_text(prefs), model_text(prefs)

    def pick_provider(name, prefs):
        prefs = {**prefs, "provider": name, "model": default_model(name)}
        return prefs, prefs["model"], model_text(prefs), key_text(prefs)

    def set_pref(name):
        return lambda value, prefs: {**prefs, name: value}

    def run_setup_checks():
        """The setup list for Settings, and a banner when something the studio needs is missing."""
        items = setup_check.checks(segment_model, images_dir, labels_dir, sim_path)
        missing = [i for i in items if i["status"] == "missing"]
        banner = ("**Setup needs attention:** " + "; ".join(f"{i['name']} ({i['detail']}): {i['without']}"
                                                          for i in missing)
                  + ". Open ⚙ Settings → Setup for the fix.") if missing else ""
        return setup_check.markdown(items), gr.update(value=banner, visible=bool(missing))

    return {
        "ids": ids, "default": default, "flow": flow,
        "ready_text": ready_text,
        "openfoam_results": openfoam_results,
        "load_airway": load_airway,
        "original_view": original_view,
        "reconstruct": reconstruct,
        "live_view": live_view,
        "run_cases": run_cases,
        "run_live": run_live,
        "simulate_original": simulate_original,
        "surgery_view": surgery_view,
        "show_original": show_original,
        "simulate_plans": simulate_plans,
        "plan_once": plan_once,
        "plan_all": plan_all,
        "llm": llm,
        "local_ready": local_ready,
        "has_key": has_key,
        "ask": ask,
        "agent_operate": agent_operate,
        "ask_local": ask_local,
        "watched": watched,
        "spectate": spectate,
        "tick": tick,
        "toggle_play": toggle_play,
        "save_timeline": save_timeline,
        "load_timeline": load_timeline,
        "watch_current": watch_current,
        "default_model": default_model,
        "key_text": key_text,
        "model_text": model_text,
        "load_prefs": load_prefs,
        "set_key": set_key,
        "check_key": check_key,
        "drop_key": drop_key,
        "pick_provider": pick_provider,
        "set_pref": set_pref,
        "run_setup_checks": run_setup_checks,
    }


def build_app(images_dir: Path | None = None, labels_dir: Path | None = None, sim_path: Path = agents.PATIENTS_PATH,
              segment_model: Path | None = None, procs: int | None = None, iterations: int = FAST_ITERATIONS):
    """Build the original clinician-facing studio around the current action layer."""
    import gradio as gr

    actions = make_actions(images_dir, labels_dir, sim_path, segment_model, procs, iterations)
    ids, default = actions["ids"], actions["default"]

    with gr.Blocks(title="Nasal0", fill_width=True) as app:
        gr.HTML('<div id="window-bar"><span class="mark">N0</span><strong>NASAL0</strong>'
                '<span class="sub">CT → airway → OpenFOAM → virtual surgery → agents</span></div>')
        state = gr.State({})
        memory = gr.State([])
        log = gr.State([])
        prefs = gr.State({})  # this browser session's agent settings, including the OpenAI key
        with gr.Sidebar(label="Settings", open=False, position="right", width=330, elem_id="settings") as settings_bar:
            gr.HTML('<div class="section-head"><span>Language model</span></div>')
            key_status = gr.Markdown(elem_classes="note")
            key_box = gr.Textbox(label="API key (OpenAI, Claude, Gemini, OpenRouter, …)", type="password",
                                 placeholder="Paste your key", max_lines=1)
            provider_pick = gr.Dropdown(list(providers.PROVIDERS), value=providers.DEFAULT, label="Provider (set from the key)")
            model_box = gr.Textbox(value=agents.OPENAI_MODEL, label="Model", max_lines=1)
            base_url_box = gr.Textbox(label="API address (custom provider only)", placeholder="https://…/v1", max_lines=1)
            remember_key = gr.Checkbox(value=False, interactive=settings.can_remember(),
                                       label="Remember on this computer (Windows Credential Manager)"
                                       if settings.can_remember() else "Remember on this computer (needs keyring)")
            with gr.Row():
                use_key = gr.Button("Use key", variant="primary", min_width=60)
                test_key = gr.Button("Test", elem_classes="signal", min_width=60)
                forget_key = gr.Button("Forget", elem_classes="signal", min_width=60)
            gr.Markdown("Used by the surgeon agent, and by the console when the local model is not installed. The "
                        "key stays in this app's memory for your browser session, and in the credential store only if "
                        "you tick Remember. It is sent only to that provider and never shown in full.",
                        elem_classes="note")
            gr.HTML('<div class="section-head"><span>Setup</span></div>')
            setup_info = gr.Markdown("Checking what this computer has…", elem_classes="note")
            recheck = gr.Button("Recheck setup", elem_classes="signal")
        with gr.Row(elem_id="shell"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=258):
                gr.HTML(f'<div class="section-head"><span>Patients</span><span class="count">{len(ids)}</span></div>')
                pid = gr.Dropdown(ids, value=default, label="NasalSeg CT", container=True)
                build = gr.Button("Open airway", variant="primary", elem_classes="signal")
                gr.HTML('<div class="section-head"><span>Upload</span></div>')
                upload = gr.File(label="CT scan: DICOM (.zip or slice files), NIfTI or NRRD", file_count="multiple",
                                 height=110)
                with gr.Accordion("ABOUT THE MODEL", open=False):
                    gr.Markdown("Rigid walls, one steady breath at 150 Pa, voxel walls. Every resistance comes "
                                "from OpenFOAM. Decision thresholds are placeholders until an ENT reviews them.",
                                elem_classes="note")
            with gr.Column(elem_id="workspace", scale=1):
                with gr.Row(elem_id="topbar"):
                    status = gr.Markdown(actions["ready_text"](), elem_id="status")
                    open_settings = gr.Button("⚙ Settings", elem_classes="signal", scale=0, min_width=110)
                setup_banner = gr.Markdown(visible=False, elem_id="setup-banner")
                with gr.Row(elem_id="session"):
                    with gr.Column(elem_id="monitor", scale=1):
                        with gr.Tabs():
                            with gr.Tab("Airway"):
                                gr.HTML('<div class="pane-title"><span>Airway</span><span>OpenFOAM resistance, Pa·s/cm³ · '
                                        'live while OpenFOAM runs</span></div>')
                                plot = gr.Plot(show_label=False)
                                live_chart = gr.Plot(show_label=False)
                                info = gr.Markdown(elem_classes="info")
                                with gr.Row():
                                    table = gr.Dataframe(label="Regions (OpenFOAM)", interactive=False)
                                    ranking = gr.Dataframe(label="All 7 plans (OpenFOAM)", interactive=False)
                            with gr.Tab("Spectator"):
                                gr.HTML('<div class="pane-title"><span>Spectator</span><span>every change, by you or the '
                                        'surgeon agent, step by step</span></div>')
                                spec_plot = gr.Plot(show_label=False)
                                spec_detail = gr.Markdown(elem_classes="info")
                                with gr.Row():
                                    spec_prev = gr.Button("◀ Step", elem_classes="signal", min_width=70)
                                    spec_play = gr.Button("▶ Play", elem_classes="signal", min_width=70)
                                    spec_next = gr.Button("Step ▶", elem_classes="signal", min_width=70)
                                    spec_follow = gr.Checkbox(value=True, label="Follow live")
                                spec_step = gr.Slider(0, 1, value=0, step=1, label="Step")
                                spec_series = gr.Plot(show_label=False)
                                spec_table = gr.Dataframe(label="Timeline", interactive=False, max_height=260)
                                with gr.Row():
                                    spec_saved = gr.Dropdown([], label="Saved sessions", scale=3)
                                    spec_load = gr.Button("Watch saved", elem_classes="signal", scale=1)
                                    spec_live = gr.Button("Watch this airway", elem_classes="signal", scale=1)
                                    spec_save = gr.Button("Save session", elem_classes="signal", scale=1)
                                spec_note = gr.Markdown(elem_classes="note")
                                spec_timer = gr.Timer(1.5)
                    with gr.Column(elem_id="controls", scale=0, min_width=244):
                        gr.HTML('<div class="section-head"><span>Commands</span></div>')
                        air_btn = gr.Button("Run airflow", elem_classes="signal")
                        procedures = gr.CheckboxGroup(list(PROCEDURE_NAMES), value=["Turbinate reduction", "Valve repair"],
                                                      label="Procedures")
                        sim_btn = gr.Button("Run airflow on plan", elem_classes="signal")
                        all_btn = gr.Button("Run all 7 plans", elem_classes="signal")
                        reset_btn = gr.Button("Show original airway", elem_classes="signal")
                        gr.HTML('<div class="section-head"><span>Surgeon agent</span></div>')
                        goal = gr.Textbox(value="Lower the resistance with the least surgery", label="Goal", lines=2)
                        surgeon_btn = gr.Button("Let the agent operate", variant="primary", elem_classes="signal")
                with gr.Column(elem_id="console"):
                    gr.HTML('<div class="pane-title"><span>Console · agents</span><span>memory stays in this session</span></div>')
                    messages = gr.HTML(console_html([]))
                    with gr.Row(elem_id="composer"):
                        question = gr.Textbox(placeholder="Ask the agents, e.g. which plan lowers resistance most and why?",
                                              show_label=False, scale=5, container=False)
                        model_label = gr.Markdown(elem_classes="note", elem_id="model-label")
                        send = gr.Button("Send", variant="primary", scale=0, min_width=70)
                        clear = gr.Button("Clear", scale=0, min_width=70, elem_classes="signal")
                    memory_info = gr.Markdown("Only the patient id, your question and computed numbers are sent to the "
                                              "model, never the CT. OpenAI requests use store=False; what any provider keeps "
                                              "is set by its own data policy.", elem_classes="note")

        view_out = [plot, table, ranking, info]
        build.click(actions["reconstruct"], [pid, upload], [state, *view_out, status])
        app.load(actions["reconstruct"], [pid, upload], [state, *view_out, status])
        air_btn.click(actions["simulate_original"], [state], [*view_out, status, live_chart])
        reset_btn.click(actions["show_original"], [state], view_out)
        sim_btn.click(actions["plan_once"], [state, procedures], [state, *view_out, status, live_chart])
        all_btn.click(actions["plan_all"], [state, procedures], [state, *view_out, status, live_chart])
        for trigger in (send.click, question.submit):
            trigger(actions["ask"], [pid, question, memory, log, prefs], [messages, log, memory, question, memory_info])
        clear.click(lambda: (console_html([]), [], [], "Session memory cleared."), None, [messages, log, memory, memory_info])
        surgeon_btn.click(actions["agent_operate"], [state, goal, prefs, log],
                          [*view_out, status, messages, log, live_chart])

        # ---- spectator: replay the airway's timeline (or a saved one) step by step
        spec = gr.State({"step": 0, "playing": False, "loaded": None, "drawn": None})
        spec_out = [spec, spec_plot, spec_detail, spec_step, spec_series, spec_table, spec_play]
        spec_timer.tick(actions["tick"], [state, spec, spec_follow], spec_out, show_progress="hidden")
        # moving by hand stops following the live end of the timeline
        spec_step.release(lambda st, sp, v: (*actions["spectate"](st, sp, step=v), False), [state, spec, spec_step],
                          [*spec_out, spec_follow])
        spec_prev.click(lambda st, sp: (*actions["spectate"](st, {**sp, "playing": False}, advance=-1), False),
                        [state, spec], [*spec_out, spec_follow])
        spec_next.click(lambda st, sp: (*actions["spectate"](st, {**sp, "playing": False}, advance=1), False),
                        [state, spec], [*spec_out, spec_follow])
        spec_play.click(actions["toggle_play"], [state, spec], spec_out)
        spec_save.click(actions["save_timeline"], [state, spec], [spec_note, spec_saved])
        spec_load.click(actions["load_timeline"], [spec_saved, state, spec], [*spec_out, spec_follow, spec_note])
        spec_live.click(actions["watch_current"], [state, spec], [*spec_out, spec_follow, spec_note])
        app.load(lambda: gr.update(choices=[p.name for p in spectator.saved_timelines()]), None, [spec_saved])
        spec_saved.focus(lambda: gr.update(choices=[p.name for p in spectator.saved_timelines()]), None, [spec_saved])

        # ---- settings
        app.load(actions["load_prefs"], None, [prefs, key_status, model_label, provider_pick, model_box, base_url_box])
        open_settings.click(lambda: gr.Sidebar(open=True), None, [settings_bar])
        key_inputs = [key_box, remember_key, provider_pick, model_box, base_url_box, prefs]
        key_outputs = [prefs, key_status, key_box, model_label, provider_pick, model_box]
        use_key.click(actions["set_key"], key_inputs, key_outputs)
        key_box.submit(actions["set_key"], key_inputs, key_outputs)
        test_key.click(actions["check_key"], [prefs], [key_status])
        forget_key.click(actions["drop_key"], [prefs], [prefs, key_status, model_label])
        provider_pick.input(actions["pick_provider"], [provider_pick, prefs], [prefs, model_box, model_label, key_status])
        for comp, name in ((model_box, "model"), (base_url_box, "base_url")):
            comp.change(actions["set_pref"](name), [comp, prefs], [prefs])
        model_box.change(lambda p: actions["model_text"](p), [prefs], [model_label])

        app.load(actions["run_setup_checks"], None, [setup_info, setup_banner])
        recheck.click(actions["run_setup_checks"], None, [setup_info, setup_banner])

    app.actions = actions
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import gradio as gr

    parser = argparse.ArgumentParser(description="Nasal0 studio (Gradio): CT to airway, OpenFOAM airflow, virtual surgery, agents.")
    parser.add_argument("--images", type=Path, default=None, help="NasalSeg images folder (*_img.nrrd); optional")
    parser.add_argument("--labels", type=Path, default=None, help="NasalSeg labels folder (*_seg.nrrd); optional")
    parser.add_argument("--sim", type=Path, default=agents.PATIENTS_PATH, help="stored OpenFOAM runs of the untouched airways")
    parser.add_argument("--segment-model", type=Path, default=None, help="default: data/segment_model.joblib")
    parser.add_argument("--procs", type=int, default=None, help="cores per OpenFOAM case (default: up to 8)")
    parser.add_argument("--iterations", type=int, default=FAST_ITERATIONS, help="solver iterations per app run")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--open", action="store_true", help="open the studio in the default browser")
    args = parser.parse_args(argv)
    app = build_app(args.images, args.labels, args.sim, args.segment_model, args.procs, args.iterations)
    app.launch(server_port=args.port, server_name="127.0.0.1", css=STUDIO_CSS, inbrowser=args.open,
               theme=gr.themes.Base(neutral_hue="neutral", text_size="sm", spacing_size="sm", radius_size="sm"), js="() => document.body.classList.add('dark')")


if __name__ == "__main__":
    main()
