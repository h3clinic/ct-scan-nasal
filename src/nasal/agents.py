"""Four small-model agents that call plain-code tools.

Every agent is the same served model (NVIDIA Nemotron 3.5 Lightning 30B-A3B by
default) with its own instructions and its own tools:

- Case: turns a description into case parameters by calling `set_case`. Code
  validates the arguments against PARAM_SPECS and sends errors back to the model.
- Planner: decides which physics tools to run (surrogate, formula baseline,
  what-if changes). The tools compute every number.
- Critic: runs the code checks (`review_results`) and reports flags, citing
  only flag ids the checks produced.
- Explainer: writes the plain-language summary. Code rejects any number in the
  text that is not in the tool results; after one retry it falls back to a
  template written by code.

The language model never produces or alters a number. There are no clinical
decision rules here: those stay placeholders until an ENT reviews them.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .params import PARAM_SPECS, REGIONS, SIDES, CaseParams
from .resistance import REFERENCE_PRESSURE, evaluate

DEFAULT_URL = os.environ.get("NASAL_LLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("NASAL_LLM_MODEL", "nemotron")
MODEL_INFO = {"name": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "total_params": "30B", "active_params": "3B"}
SURROGATE_PATH = Path("data/surrogate.pt")
MAX_TURNS = 6
PLANNER_MAX_TURNS = 10  # surgery questions ask for several what-ifs
DISAGREEMENT_FLAG_PCT = 25.0  # surrogate vs formula, a model-agreement check, not a clinical threshold
EXTRAPOLATION_SD = 3.0  # patient geometry this many std devs from the training mean is flagged
PATIENTS_PATH = Path("data/patient_sim.csv")
PATIENT_SURROGATE_PATH = Path("data/patient_surrogate.json")
SURGERY_PATH = Path("data/patient_surgery.csv")  # OpenFOAM on every plan, from nasal-patients surgery

# ---------------------------------------------------------------- tools (code)


def _param_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "number", "minimum": s.low, "maximum": s.high, "description": f"{s.description} ({s.unit})"}
            for name, s in PARAM_SPECS.items()
        },
        "required": list(PARAM_SPECS),
    }


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


class ToolError(Exception):
    """A tool call the model can fix: the message goes back to the model."""


def _params(args: dict) -> CaseParams:
    missing = [k for k in PARAM_SPECS if k not in args]
    if missing:
        raise ToolError(f"missing arguments: {', '.join(missing)}")
    try:
        params = CaseParams(**{k: float(args[k]) for k in PARAM_SPECS})
    except (TypeError, ValueError) as err:
        raise ToolError(f"arguments must be numbers: {err}") from err
    bad = params.out_of_range()
    if bad:
        ranges = "; ".join(f"{k} must be {PARAM_SPECS[k].low:g} to {PARAM_SPECS[k].high:g}" for k in bad)
        raise ToolError(f"out of range: {ranges}")
    return params


def _summary(record: dict) -> dict:
    keep = ["total_resistance", "total_flow_cm3_s"] + [f"{s}_resistance" for s in SIDES] + [
        f"{s}_{r}_resistance" for s in SIDES for r in REGIONS
    ]
    return {k: round(float(record[k]), 4) for k in keep}


@dataclass
class Tools:
    """The plain-code tools agents can call, plus the state they produce."""

    surrogate: object | None = None
    case: CaseParams | None = None
    results: dict = field(default_factory=dict)
    patients: object | None = None  # DataFrame of NasalSeg patients (geometry, and OpenFOAM results if run)
    patient_surrogate: object | None = None
    patient: dict | None = None
    labels_dir: Path | None = None  # NasalSeg *_seg.nrrd files, needed for virtual surgery
    surgery: object | None = None  # DataFrame of OpenFOAM surgery runs, if any

    # ---- real patients (NasalSeg) ----

    def _patient_row(self, args: dict) -> dict:
        if self.patients is None:
            raise ToolError("no patient data loaded")
        pid = str(args.get("patient_id", "")).strip().upper()
        match = self.patients[self.patients.patient == pid]
        if match.empty:
            raise ToolError(f"unknown patient_id {pid!r}; ids look like P001 to P130")
        return match.iloc[0].to_dict()

    def set_patient(self, args: dict) -> dict:
        self.patient = self._patient_row(args)
        geometry = {k: round(float(v), 1) for k, v in self.patient.items()
                    if any(k.startswith(f"{s}_") for s in SIDES) and k.endswith(("_mm2", "_mm", "_cm3"))}
        return {"patient_id": self.patient["patient"], "geometry": geometry, "valid": True}

    def _patient_predict(self, geometry: dict) -> tuple[str, dict]:
        if self.patient_surrogate is None:
            raise ToolError("no trained patient surrogate; run nasal-surrogate --patients first")
        return "patient surrogate (power law fitted to OpenFOAM on real airways)", self.patient_surrogate.predict(geometry)

    def patient_resistance(self, args: dict) -> dict:
        row = self._patient_row(args)
        source, record = self._patient_predict(row)
        out = {"source": source, "patient_id": row["patient"], **_summary(record)}
        self.results["prediction"] = out
        if not np.isnan(row.get("total_resistance", np.nan)):
            self.results["openfoam"] = {"source": "OpenFOAM on this patient's CT airway", **_summary(row)}
            out["openfoam_total_resistance"] = self.results["openfoam"]["total_resistance"]
        return out

    def patient_what_if(self, args: dict) -> dict:
        row = self._patient_row(args)
        side = str(args.get("side", "")).lower()
        if side not in SIDES:
            raise ToolError("side must be 'left' or 'right'")
        try:
            pct = float(args.get("narrowest_area_change_pct"))
        except (TypeError, ValueError) as err:
            raise ToolError("narrowest_area_change_pct must be a number") from err
        if not -50 <= pct <= 100:
            raise ToolError("narrowest_area_change_pct must be between -50 and 100")
        changed = dict(row)
        changed[f"{side}_min_area_mm2"] = row[f"{side}_min_area_mm2"] * (1 + pct / 100)
        changed[f"{side}_mean_area_mm2"] = max(row[f"{side}_mean_area_mm2"], changed[f"{side}_min_area_mm2"])
        source, before = self._patient_predict(row)
        _, after = self._patient_predict(changed)
        out = {
            "source": source, "patient_id": row["patient"],
            "label": f"{side} narrowest cross-section {'widened' if pct >= 0 else 'narrowed'} by {abs(pct):g}%",
            "total_resistance_before": round(before["total_resistance"], 4),
            "total_resistance_after": round(after["total_resistance"], 4),
            "total_change_pct": round(100 * (after["total_resistance"] / before["total_resistance"] - 1), 1),
            **{f"{s}_resistance_after": round(after[f"{s}_resistance"], 4) for s in SIDES},
        }
        self.results.setdefault("what_if", []).append(out)
        return out

    def patient_surgery_options(self, args: dict) -> dict:
        row = self._patient_row(args)
        pid = row["patient"]
        if self.patient_surrogate is None:
            raise ToolError("no trained patient surrogate; run nasal-surrogate --patients first")
        files = sorted(Path(self.labels_dir).glob(f"{pid}_*seg.nrrd")) if self.labels_dir else []
        if not files:
            raise ToolError("virtual surgery needs the patient's CT labels, which are not loaded")
        out = dict(surgery_options(pid, str(files[0]), self.patient_surrogate))
        out.update(_technique_notes(row if not np.isnan(row.get("total_resistance", np.nan))
                                    else self.patient_surrogate.predict(row), out["recommended_plan"]))
        if self.surgery is not None:
            runs = self.surgery[self.surgery.patient == pid]
            cfd = dict(zip(runs.plan, runs.total_resistance))
            if "none" in cfd and len(cfd) > 1:
                drops = {plan_name(p): round(100 * (1 - r / cfd["none"]), 1) for p, r in cfd.items() if p != "none"}
                best = max(drops, key=drops.get)
                check = {"source": "OpenFOAM on each operated airway", "best_plan": best, "best_drop_pct": drops[best]}
                for key in ("recommended", "look_at_scan"):
                    plan = out[f"{key}_plan"]
                    if plan in drops:
                        check[f"{key}_plan_drop_pct"] = drops[plan]
                        check[f"{key}_plan_result"] = _drop_text(plan, drops[plan], "in OpenFOAM")
                out["openfoam_check"] = check
        self.results["surgery"] = out
        return out

    def recommend_surgery(self, args: dict) -> dict:
        """Pankaj's final-stage recommender (nasal.decision.recommend) on the current synthetic case."""
        from .decision.recommend import recommend

        if self.case is None:
            raise ToolError("set_case first")
        source, predict = self.predictor()
        r = recommend(self.case, predict, source)
        plan, std = plan_name(r["suggested"]["plan"]), plan_name(r["standard"]["plan"])
        out = {
            "source": f"virtual surgery on the case, scored by the {r['evaluator']}",
            "recommended_plan": plan, "recommended_drop_pct": r["suggested"]["drop_pct"],
            "recommended_result": _drop_text(plan, r["suggested"]["drop_pct"], "predicted"),
            "look_at_scan_plan": std, "look_at_scan_drop_pct": r["standard"]["drop_pct"],
            "look_at_scan_result": _drop_text(std, r["standard"]["drop_pct"], "predicted"),
            "why": r["why"],
            "dominant_region": f"{r['before']['worse_side']} {r['dominant_region']}: "
                               f"{r['before']['region_share_pct_worse_side'][r['dominant_region']]}% of that side's resistance",
            "not_modeled": r["not_modeled"],
        }
        self.results["surgery"] = out
        return out

    def predictor(self) -> tuple[str, Callable[[CaseParams], dict]]:
        if self.surrogate is not None:
            return "surrogate (PyTorch, trained on OpenFOAM)", self.surrogate.predict
        return "formula baseline (no trained surrogate found)", evaluate

    def set_case(self, args: dict) -> dict:
        self.case = _params(args)
        return {"case": asdict(self.case), "valid": True}

    def predict_resistance(self, args: dict) -> dict:
        source, predict = self.predictor()
        out = {"source": source, **_summary(predict(_params(args)))}
        self.results["prediction"] = out
        return out

    def formula_resistance(self, args: dict) -> dict:
        out = {"source": "formula baseline (uncalibrated)", **_summary(evaluate(_params(args)))}
        self.results["formula"] = out
        return out

    def what_if(self, args: dict) -> dict:
        if self.case is None:
            raise ToolError("no case set yet")
        changes = {k: v for k, v in args.items() if k in PARAM_SPECS}
        if not changes:
            raise ToolError(f"give at least one of: {', '.join(PARAM_SPECS)}")
        source, predict = self.predictor()
        before = predict(self.case)
        after_params = _params({**asdict(self.case), **changes})
        after = predict(after_params)
        out = {
            "source": source,
            "label": str(args.get("label", "what-if")),
            "changes": {k: float(v) for k, v in changes.items()},
            "total_resistance_before": round(before["total_resistance"], 4),
            "total_resistance_after": round(after["total_resistance"], 4),
            "total_change_pct": round(100 * (after["total_resistance"] / before["total_resistance"] - 1), 1),
            **{f"{s}_resistance_after": round(after[f"{s}_resistance"], 4) for s in SIDES},
        }
        self.results.setdefault("what_if", []).append(out)
        return out

    def check_parameters(self, args: dict) -> dict:
        values = {k: args.get(k) for k in PARAM_SPECS}
        problems = []
        for k, v in values.items():
            s = PARAM_SPECS[k]
            if not isinstance(v, (int, float)):
                problems.append(f"{k} is missing or not a number")
            elif not s.low <= v <= s.high:
                problems.append(f"{k}={v:g} is outside {s.low:g} to {s.high:g}")
        return {"ok": not problems, "problems": problems}

    def review_results(self, args: dict) -> dict:
        """Deterministic checks. Each flag has an id the Critic must cite."""
        flags = []
        pred, form = self.results.get("prediction"), self.results.get("formula")
        if self.case is None and self.patient is None:
            flags.append({"id": "no_case", "severity": "error", "text": "No valid case or patient was set."})
        cfd = self.results.get("openfoam")
        if pred and cfd:
            diff = 100 * abs(pred["total_resistance"] / cfd["total_resistance"] - 1)
            if diff > DISAGREEMENT_FLAG_PCT:
                flags.append({"id": "surrogate_vs_openfoam", "severity": "warning",
                              "text": f"Surrogate and this patient's OpenFOAM run differ by {diff:.0f}%."})
        if self.patient is not None and self.patient_surrogate is not None:
            z = self.patient_surrogate.zscores(self.patient_surrogate.side_features(self.patient))
            if np.abs(z).max() > EXTRAPOLATION_SD:
                flags.append({"id": "unusual_anatomy", "severity": "warning",
                              "text": "This airway is far outside the training patients, so the surrogate is extrapolating."})
        if pred is None:
            flags.append({"id": "no_prediction", "severity": "error", "text": "No resistance prediction was run."})
        if pred and form:
            diff = 100 * abs(pred["total_resistance"] / form["total_resistance"] - 1)
            if diff > DISAGREEMENT_FLAG_PCT:
                flags.append({"id": "model_disagreement", "severity": "warning",
                              "text": f"Surrogate and formula differ by {diff:.0f}% in total resistance."})
        if pred and "formula" in pred["source"]:
            flags.append({"id": "no_surrogate", "severity": "info",
                          "text": "Using the uncalibrated formula because no trained surrogate was found."})
        if pred:
            for side in SIDES:
                parts = sum(pred[f"{side}_{r}_resistance"] for r in REGIONS)
                if abs(parts - pred[f"{side}_resistance"]) > 1e-3:
                    flags.append({"id": f"{side}_regions_do_not_sum", "severity": "error",
                                  "text": f"{side} region resistances do not add up to the side total."})
            hi, lo = max(SIDES, key=lambda s: pred[f"{s}_resistance"]), min(SIDES, key=lambda s: pred[f"{s}_resistance"])
            if pred[f"{hi}_resistance"] > 1.5 * pred[f"{lo}_resistance"]:
                flags.append({"id": "asymmetric", "severity": "info",
                              "text": f"The {hi} side has over 1.5 times the resistance of the {lo} side."})
        geometry = ("Real CT airway (NasalSeg), voxel-resolution model." if self.patient is not None
                    else "Synthetic, idealized geometry.")
        flags.append({"id": "model_scope", "severity": "info", "text": geometry})
        self.results["flags"] = flags
        return {"flags": flags}


SCHEMAS = {
    "set_case": _tool("set_case", "Record the case parameters extracted from the description.", _param_schema()),
    "predict_resistance": _tool(
        "predict_resistance", "Fast nasal resistance prediction per side and region (Pa*s/cm^3) at 150 Pa.",
        _param_schema()),
    "formula_resistance": _tool(
        "formula_resistance", "Uncalibrated friction-plus-narrowing formula, for comparison.", _param_schema()),
    "what_if": _tool("what_if", "Resistance before and after changing some parameters of the current case.", {
        "type": "object",
        "properties": {"label": {"type": "string", "description": "short name for the change"},
                       **{k: v for k, v in _param_schema()["properties"].items()}},
        "required": ["label"],
    }),
    "check_parameters": _tool("check_parameters", "Check parameter values against the allowed ranges.", {
        "type": "object", "properties": _param_schema()["properties"]}),
    "review_results": _tool("review_results", "Run the code checks on everything computed so far.",
                            {"type": "object", "properties": {}}),
    "finish": _tool("finish", "Call when you have run every tool the request needs.",
                    {"type": "object", "properties": {"reason": {"type": "string"}}}),
    "set_patient": _tool("set_patient", "Select a real NasalSeg patient by id (P001 to P130).", {
        "type": "object", "properties": {"patient_id": {"type": "string", "pattern": "^P[0-9]{3}$"}},
        "required": ["patient_id"]}),
    "patient_resistance": _tool(
        "patient_resistance",
        "Predicted nasal resistance per side and region for a real patient, plus their OpenFOAM result if available.",
        {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}),
    "patient_what_if": _tool(
        "patient_what_if",
        "Predicted resistance if one side's narrowest cross-section were widened or narrowed by a percentage.",
        {"type": "object", "properties": {
            "patient_id": {"type": "string"},
            "side": {"type": "string", "enum": ["left", "right"]},
            "narrowest_area_change_pct": {"type": "number", "minimum": -50, "maximum": 100}},
         "required": ["patient_id", "side", "narrowest_area_change_pct"]}),
    "patient_surgery_options": _tool(
        "patient_surgery_options",
        "Virtual surgery on the patient's CT airway: turbinate reduction, valve repair, septoplasty and their "
        "combinations, each scored for resistance drop; returns the recommended plan and the look-at-the-scan plan.",
        {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}),
    "recommend_surgery": _tool(
        "recommend_surgery",
        "Final-stage surgery suggestion for the current case: virtual turbinate reduction, valve repair and "
        "septoplasty in every combination, the suggested plan with reasons, and the look-at-the-scan plan.",
        {"type": "object", "properties": {}}),
    "report_flags": _tool("report_flags", "Report which flags from review_results matter for the reader.", {
        "type": "object",
        "properties": {"flag_ids": {"type": "array", "items": {"type": "string"}},
                       "note": {"type": "string", "description": "one sentence, no numbers"}},
        "required": ["flag_ids"],
    }),
}

# ---------------------------------------------------------------- agent loop

PROMPTS = {
    "case": (
        "You are the Case agent in a research prototype about nasal airflow. Turn the user's description into "
        "the three case parameters and call set_case exactly once. Valve angle is in degrees (normal about 15, "
        "narrow about 8 to 10). Turbinate scale 1.0 is nominal; larger means a bigger turbinate and less airway. "
        "Septum deviation is -1 to 1: positive bends toward and narrows the LEFT side, negative the RIGHT. "
        "If the description does not mention a parameter, use the nominal value (15, 1.0, 0.0). "
        "If set_case returns an error, fix the arguments and call it again."
    ),
    "case_patient": (
        "You are the Case agent in a research prototype about nasal airflow. If the request names a real "
        "NasalSeg patient (an id like P017), call set_patient with that id. Otherwise turn the description into "
        "case parameters and call set_case: valve angle in degrees (normal about 15, narrow 8 to 10), turbinate "
        "scale (1.0 nominal, larger means less airway), septum deviation -1 to 1 (positive narrows the LEFT side). "
        "Use nominal values (15, 1.0, 0.0) for anything not mentioned. If a tool returns an error, fix it and retry."
    ),
    "planner_patient": (
        "You are the Planner. A real patient has been selected. Always call patient_resistance for that patient. "
        "If the user asks about widening or narrowing one side, call patient_what_if with the side and the "
        "percentage change of its narrowest cross-section. If the user asks about surgery, treatment or what to do, "
        "call patient_surgery_options once. Never state numbers yourself. When done, call finish."
    ),
    "planner": (
        "You are the Planner. A case has been set. Decide which tools answer the user's request. Always call "
        "predict_resistance and formula_resistance for the case. If the user asks about a change (for example "
        "widening the valve or reducing the turbinate), call what_if with only the parameters that change. "
        "If the user asks which surgery or treatment to do, call recommend_surgery once. "
        "Never state numbers yourself. When done, call finish."
    ),
    "critic": (
        "You are the Critic. Call review_results, then call report_flags with the ids of the flags a reader must "
        "see. Use only ids that review_results returned. Do not write numbers."
    ),
    "explainer": (
        "You are the Explainer. Write a short plain-language summary (at most 150 words) of the results JSON for "
        "a non-specialist. Use only numbers that appear in the JSON, copied exactly; do not compute new ones. "
        "Resistance is in Pa*s/cm^3; higher means harder to breathe through. For every what_if entry, say what "
        "changed and give total_resistance_before, total_resistance_after and total_change_pct. "
        "If the JSON has an openfoam entry, give its total_resistance as this patient's simulation result and "
        "say the other numbers are fast surrogate predictions. Mention every flag listed. "
        "The prediction entry is the patient's current resistance, before any surgery. If the JSON has a surgery "
        "entry, copy its recommended_result and look_at_scan_result sentences, and if there is an openfoam_check, "
        "its recommended_plan_result and look_at_scan_plan_result sentences and its best_plan; keep the words "
        "lower and higher exactly as written. Give the why reasons and the dominant_region (and "
        "dominant_region_note if present). Say a surgeon makes the final decision."
    ),
}


@dataclass
class AgentLog:
    steps: list = field(default_factory=list)
    listener: Callable[[dict], None] | None = None  # called with each step as it happens (live demo)

    def add(self, agent: str, kind: str, **data) -> None:
        step = {"agent": agent, "kind": kind, "t": round(time.time(), 3), **data}
        self.steps.append(step)
        if self.listener is not None:
            self.listener(step)


def run_agent(client, model: str, agent: str, system: str, user: str, tools: Tools, tool_names: list[str],
              log: AgentLog, final_tool: str | tuple[str, ...] | None = None, max_turns: int = MAX_TURNS,
              extra_body: dict | None = None, stop_at_limit: bool = False):
    """Chat loop: execute tool calls in code, send errors back, stop at final_tool.
    Returns the final tool's arguments, or the model's text if no final tool. With stop_at_limit,
    running out of turns keeps the tool results so far and returns None instead of raising."""
    finals = (final_tool,) if isinstance(final_tool, str) else tuple(final_tool or ())
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    schemas = [SCHEMAS[n] for n in tool_names]
    for turn in range(max_turns):
        start = time.time()
        kwargs = {"model": model, "messages": messages, "temperature": 0.2}
        if schemas:
            kwargs.update(tools=schemas, tool_choice="auto")
        if extra_body:
            kwargs["extra_body"] = extra_body
        reply = client.chat.completions.create(**kwargs).choices[0].message
        calls = reply.tool_calls or []
        log.add(agent, "model", turn=turn, seconds=round(time.time() - start, 2),
                content=reply.content, tool_calls=[{"name": c.function.name, "arguments": c.function.arguments}
                                                   for c in calls])
        if not calls:
            if not finals:
                return reply.content or ""
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append({"role": "user", "content": f"Call {' or '.join(finals)} to finish."})
            continue
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls]})
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ToolError("arguments must be a JSON object")
                if name not in tool_names:
                    raise ToolError(f"unknown tool {name}; available: {', '.join(tool_names)}")
                if name in finals:
                    if name == "report_flags":
                        known = {f["id"] for f in tools.results.get("flags", [])}
                        unknown = set(args.get("flag_ids", [])) - known
                        if not known:
                            raise ToolError("call review_results first")
                        if unknown:
                            raise ToolError(f"unknown flag ids: {', '.join(sorted(unknown))}")
                    result = getattr(tools, name)(args) if hasattr(tools, name) else {"ok": True}
                    log.add(agent, "tool", name=name, arguments=args, result=result)
                    return args
                result = getattr(tools, name)(args)
            except json.JSONDecodeError as err:
                result = {"error": f"arguments are not valid JSON: {err}"}
            except ToolError as err:
                result = {"error": str(err)}
            log.add(agent, "tool", name=name, arguments=c.function.arguments, result=result)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
    if stop_at_limit:
        log.add(agent, "fallback", reason=f"stopped after {max_turns} turns; using the results so far")
        return None
    raise RuntimeError(f"{agent} agent did not finish in {max_turns} turns")


# ---------------------------------------------------------------- explainer check

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _facts_numbers(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _facts_numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _facts_numbers(v)]
    if isinstance(obj, str):
        return [float(m) for m in NUMBER.findall(obj)]
    return []


def unsupported_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in text that do not match any number in facts at the precision written."""
    allowed = _facts_numbers(facts) + [REFERENCE_PRESSURE] + list(range(0, 11))
    bad = []
    for token in NUMBER.findall(text):
        value = float(token)
        decimals = len(token.split(".")[1]) if "." in token else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(value - round(a, decimals)) <= tol or abs(abs(value) - abs(round(a, decimals))) <= tol
                   for a in allowed):
            bad.append(token)
    return bad


def missing_numbers(text: str, facts: dict) -> list[str]:
    """Key results the summary must quote: the total prediction and each what-if's after value."""
    required = [facts["prediction"]["total_resistance"]] + [w["total_resistance_after"] for w in facts.get("what_if", [])]
    if "openfoam" in facts:
        required.append(facts["openfoam"]["total_resistance"])
    if "surgery" in facts:
        required.append(facts["surgery"]["recommended_drop_pct"])
    found = [float(t) for t in NUMBER.findall(text)]
    return [f"{v:g}" for v in required if not any(abs(f - v) <= 0.5 * 10 ** -3 + 1e-9 for f in found)]


def template_summary(facts: dict) -> str:
    """Code-written fallback when the Explainer's text fails the number check."""
    p = facts["prediction"]
    lines = [f"Predicted total nasal resistance is {p['total_resistance']} Pa*s/cm^3 "
             f"(left {p['left_resistance']}, right {p['right_resistance']}), from the {p['source']}."]
    if "openfoam" in facts:
        lines.append(f"This patient's OpenFOAM simulation gives {facts['openfoam']['total_resistance']} Pa*s/cm^3.")
    for w in facts.get("what_if", []):
        lines.append(f"With the change '{w['label']}', total resistance goes from {w['total_resistance_before']} "
                     f"to {w['total_resistance_after']} ({w['total_change_pct']}%).")
    s = facts.get("surgery")
    if s:
        lines.append(f"Recommended surgery: {s['recommended_result']}. Look-at-the-scan plan: {s['look_at_scan_result']}.")
        lines += [f"Why: {w}." for w in s.get("why", [])]
        if s.get("dominant_region"):
            lines.append(f"Most resistance: {s['dominant_region']}.")
        if s.get("dominant_region_note"):
            lines.append(f"Note: {s['dominant_region_note']}.")
        check = s.get("openfoam_check")
        if check and check.get("recommended_plan_result"):
            lines.append(f"OpenFOAM check: {check['recommended_plan_result']}; best plan: {check['best_plan']}.")
    for f in facts.get("flags", []):
        lines.append(f["text"])
    return " ".join(lines)


def plan_name(plan) -> str:
    """'turbinate_reduction+valve_repair' or a tuple of procedures -> 'turbinate reduction + valve repair'."""
    parts = plan.split("+") if isinstance(plan, str) else list(plan)
    return " + ".join(p.replace("_", " ") for p in parts if p and p != "none") or "no surgery"


_SURGERY_CACHE: dict = {}


def surgery_options(pid: str, label_path: str, surrogate) -> dict:
    """Every procedure plan on the patient's voxel airway, scored by the patient surrogate (a few seconds,
    then cached)."""
    key = (pid, label_path, id(surrogate))
    if key not in _SURGERY_CACHE:
        _SURGERY_CACHE[key] = _surgery_options(pid, label_path, surrogate)
    return _SURGERY_CACHE[key]


def _surgery_options(pid: str, label_path: str, surrogate) -> dict:
    from . import anatomy
    from .decision import patient as ps

    labels, spacing, _ = anatomy.load_labels(Path(label_path))
    tags, cuts = anatomy.airway(labels)
    recommended, predicted = ps.plus_plus_rule_patient(tags, cuts, spacing, ps.surrogate_evaluator(surrogate, spacing))
    standard = ps.standard_rule_patient(anatomy.measure(tags, spacing))
    before = predicted[()]
    drop = {plan: round(100 * (1 - r / before), 1) for plan, r in predicted.items()}
    return {
        "source": "patient surrogate on the virtually operated CT airway", "patient_id": pid,
        "plans_compared": len(predicted) - 1,
        "recommended_plan": plan_name(recommended), "recommended_drop_pct": drop[recommended],
        "recommended_result": _drop_text(plan_name(recommended), drop[recommended], "predicted"),
        "look_at_scan_plan": plan_name(standard), "look_at_scan_drop_pct": drop[standard],
        "look_at_scan_result": _drop_text(plan_name(standard), drop[standard], "predicted"),
        "method": "smallest plan within 90% of the best predicted drop",
    }


def _technique_notes(record: dict, plan: str) -> dict:
    """Pankaj's recommender notes for a patient: why each procedure, where the resistance sits,
    and what the model leaves out."""
    from .decision.recommend import REASONS
    from .decision.rules import region_fractions

    shares = region_fractions(record)
    side = max(SIDES, key=lambda s: record[f"{s}_resistance"])
    top = max(shares, key=shares.get)
    targeted = {"turbinate reduction": "turbinate", "valve repair": "valve"}
    procedures = [p for p in plan.split(" + ") if p != "no surgery"]
    notes = {
        "why": [f"{p}: targets {REASONS[p.replace(' ', '_')]}" for p in procedures]
               or ["No procedure lowered modeled resistance."],
        "dominant_region": f"{side} {top}: {round(100 * shares[top], 1)}% of that side's resistance",
        "not_modeled": ["inferior/middle/superior flow split", "wall shear and ENS-risk patterns",
                        "mucosal state", "patient symptoms"],
    }
    if top not in targeted.values():
        notes["dominant_region_note"] = (f"none of the three procedures acts on the {top}, "
                                         f"where most of the {side} side's resistance sits")
    return notes


def _drop_text(plan: str, drop_pct: float, where: str) -> str:
    """A drop stated in words, so its direction can't be misread."""
    word = "lower" if drop_pct >= 0 else "higher"
    return f"{plan}: {abs(drop_pct)}% {word} resistance ({where})"


# ---------------------------------------------------------------- pipeline


def load_surrogate(path: Path = SURROGATE_PATH):
    if not Path(path).exists():
        return None
    try:
        from .surrogate import Surrogate
        return Surrogate.load(path)
    except ImportError:
        return None


def load_surgery(path: Path = SURGERY_PATH):
    """OpenFOAM runs of every surgery plan (nasal-patients surgery), or None."""
    if not Path(path).exists():
        return None
    import pandas as pd
    return pd.read_csv(path)


def load_patients(sim_path: Path = PATIENTS_PATH, model_path: Path = PATIENT_SURROGATE_PATH):
    """NasalSeg patient table and trained patient surrogate, or (None, None) when missing."""
    if not Path(sim_path).exists() or not Path(model_path).exists():
        return None, None
    import pandas as pd

    from .surrogate import PatientSurrogate
    return pd.read_csv(sim_path), PatientSurrogate.load(model_path)


def run_pipeline(request: str, client=None, model: str = DEFAULT_MODEL, surrogate=None,
                 thinking: bool = False, patients=None, patient_surrogate=None,
                 listener: Callable[[dict], None] | None = None, labels_dir: Path | None = None,
                 surgery=None) -> dict:
    """Case -> Planner -> Critic -> Explainer. Returns the facts, the summary and the full agent log.
    With `patients` (and their surrogate) the Case agent may pick a real NasalSeg patient."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=DEFAULT_URL, api_key=os.environ.get("NASAL_LLM_KEY", "none"))
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    tools = Tools(surrogate=surrogate, patients=patients, patient_surrogate=patient_surrogate,
                  labels_dir=labels_dir, surgery=surgery)
    log = AgentLog(listener=listener)
    start = time.time()

    if patients is not None:
        run_agent(client, model, "case", PROMPTS["case_patient"], request, tools, ["set_patient", "set_case"], log,
                  final_tool=("set_patient", "set_case"), extra_body=extra)
    else:
        run_agent(client, model, "case", PROMPTS["case"], request, tools, ["set_case"], log, final_tool="set_case",
                  extra_body=extra)

    if tools.patient is not None:
        pid = tools.patient["patient"]
        run_agent(client, model, "planner", PROMPTS["planner_patient"], f"Request: {request}\nPatient: {pid}", tools,
                  ["patient_resistance", "patient_what_if", "patient_surgery_options", "finish"], log, final_tool="finish", extra_body=extra,
                  max_turns=PLANNER_MAX_TURNS, stop_at_limit=True)
        if "prediction" not in tools.results:
            log.add("planner", "fallback", reason="planner skipped patient_resistance")
            tools.patient_resistance({"patient_id": pid})
        case_facts = {"patient_id": pid}
    else:
        case_json = json.dumps(asdict(tools.case))
        run_agent(client, model, "planner", PROMPTS["planner"], f"Request: {request}\nCase: {case_json}", tools,
                  ["predict_resistance", "formula_resistance", "what_if", "recommend_surgery", "finish"], log,
                  final_tool="finish",
                  extra_body=extra, max_turns=PLANNER_MAX_TURNS, stop_at_limit=True)
        if "prediction" not in tools.results:  # the plan must include a prediction; code enforces it
            log.add("planner", "fallback", reason="planner skipped predict_resistance")
            tools.predict_resistance(asdict(tools.case))
        case_facts = asdict(tools.case)

    flags_chosen = run_agent(client, model, "critic", PROMPTS["critic"], "Review the results.", tools,
                             ["review_results", "check_parameters", "report_flags"], log, final_tool="report_flags",
                             extra_body=extra)
    chosen = set(flags_chosen.get("flag_ids", [])) | {"model_scope"}
    chosen |= {f["id"] for f in tools.results["flags"] if f["severity"] == "error"}  # errors are never dropped
    facts = {"case": case_facts, **{k: v for k, v in tools.results.items() if k != "flags"},
             "flags": [f for f in tools.results["flags"] if f["id"] in chosen]}

    summary, source = None, "explainer"
    for attempt in range(2):
        prompt = f"Results JSON:\n{json.dumps(facts, indent=1)}"
        if attempt:
            prompt += f"\nYour previous text had problems: {problem}. Rewrite it."
        text = run_agent(client, model, "explainer", PROMPTS["explainer"], prompt, tools, [], log,
                         extra_body=extra)
        bad, missing = unsupported_numbers(text, facts), missing_numbers(text, facts)
        log.add("explainer", "number_check", unsupported=bad, missing=missing)
        problem = "; ".join(filter(None, [
            bad and f"numbers not in the JSON: {', '.join(bad)}",
            missing and f"missing required results: {', '.join(missing)}",
        ]))
        if not problem:
            summary = text
            break
    if summary is None:
        summary, source = template_summary(facts), "template (explainer failed the number check)"
    return {"request": request, "model": MODEL_INFO, "facts": facts, "summary": summary,
            "summary_source": source, "seconds": round(time.time() - start, 2), "log": log.steps}
