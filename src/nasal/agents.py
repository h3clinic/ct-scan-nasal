"""Four small-model agents that call plain-code tools.

Every agent is the same served model (NVIDIA Nemotron 3.5 Lightning 30B-A3B by
default) with its own instructions and its own tools:

- Case: turns a description into case parameters by calling `set_case`. Code
  validates the arguments against PARAM_SPECS and sends errors back to the model.
- Planner: decides which physics tools to run: OpenFOAM results and virtual
  surgery for real patients; surrogate, formula baseline and what-if changes
  for synthetic cases. The tools compute every number.
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

from .decision.patient import PATIENT_PROCEDURES
from .params import PARAM_SPECS, REGIONS, SIDES, CaseParams
from .resistance import REFERENCE_PRESSURE, evaluate

DEFAULT_URL = os.environ.get("NASAL_LLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("NASAL_LLM_MODEL", "nemotron")
MODEL_INFO = {"name": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "total_params": "30B", "active_params": "3B"}
# Optional hosted backend: OpenAI's API with OPENAI_API_KEY. Requests are sent with store=False, so OpenAI
# keeps no stored completion; under OpenAI's API policy, API data is not used for training by default but
# may be retained for abuse monitoring (up to 30 days) unless the organization has zero data retention.
# OpenAI does not publish its models' sizes, so this backend is outside the project's 33B limit.
OPENAI_MODEL = os.environ.get("NASAL_OPENAI_MODEL", "gpt-4.1-mini")
PROVIDERS = ("local", "openai")
SURROGATE_PATH = Path("data/surrogate.pt")
MAX_TURNS = 6
PLANNER_MAX_TURNS = 10  # surgery questions ask for several what-ifs
DISAGREEMENT_FLAG_PCT = 25.0  # surrogate vs formula, a model-agreement check, not a clinical threshold
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
    patient: dict | None = None
    flow: object | None = None  # nasal.patient_flow.PatientFlow: every patient resistance comes from OpenFOAM

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

    def _openfoam(self, pid: str, plans: list[tuple[str, ...]]) -> dict[tuple[str, ...], dict]:
        """OpenFOAM records for the patient's plans; () is the untouched airway."""
        if self.flow is None:
            raise ToolError("no OpenFOAM results are loaded for real patients")
        try:
            return self.flow.records(pid, plans)
        except LookupError as err:
            raise ToolError(str(err)) from err

    def patient_resistance(self, args: dict) -> dict:
        row = self._patient_row(args)
        record = self._openfoam(row["patient"], [()])[()]
        out = {"source": f"OpenFOAM on this patient's CT airway, {record['source']}", "patient_id": row["patient"],
               **_summary(record)}
        self.results["prediction"] = out
        return out

    def patient_what_if(self, args: dict) -> dict:
        row = self._patient_row(args)
        procedures = args.get("procedures")
        if not isinstance(procedures, list) or not procedures:
            raise ToolError(f"procedures must be a non-empty list from {list(PATIENT_PROCEDURES)}")
        unknown = [p for p in procedures if p not in PATIENT_PROCEDURES]
        if unknown:
            raise ToolError(f"unknown procedures {unknown}; choose from {list(PATIENT_PROCEDURES)}")
        plan = tuple(p for p in PATIENT_PROCEDURES if p in procedures)
        recs = self._openfoam(row["patient"], [(), plan])
        before, after = recs[()], recs[plan]
        out = {
            "source": f"OpenFOAM on the virtually operated CT airway, {after['source']}", "patient_id": row["patient"],
            "label": plan_name(plan),
            "total_resistance_before": round(before["total_resistance"], 4),
            "total_resistance_after": round(after["total_resistance"], 4),
            "total_change_pct": round(100 * (after["total_resistance"] / before["total_resistance"] - 1), 1),
            **{f"{s}_resistance_after": round(after[f"{s}_resistance"], 4) for s in SIDES},
        }
        self.results.setdefault("what_if", []).append(out)
        return out

    def patient_surgery_options(self, args: dict) -> dict:
        from .decision import patient as ps
        from .patient_flow import plus_plus_from_totals

        row = self._patient_row(args)
        pid = row["patient"]
        recs = self._openfoam(pid, [(), *ps.candidate_plans()])
        totals = {plan: r["total_resistance"] for plan, r in recs.items()}
        drop = {plan: round(100 * (1 - r / totals[()]), 1) for plan, r in totals.items()}
        recommended = plus_plus_from_totals(totals)
        standard = ps.standard_rule_patient(recs[()])
        best = max((p for p in totals if p), key=lambda p: drop[p])
        out = {
            "source": "OpenFOAM on the CT airway after each virtual surgery"
                      + ("; some plans run now" if any("run now" in r["source"] for r in recs.values()) else ""),
            "patient_id": pid,
            "plans_compared": len(totals) - 1,
            "recommended_plan": plan_name(recommended), "recommended_drop_pct": drop[recommended],
            "recommended_result": _drop_text(plan_name(recommended), drop[recommended], "in OpenFOAM"),
            "look_at_scan_plan": plan_name(standard), "look_at_scan_drop_pct": drop[standard],
            "look_at_scan_result": _drop_text(plan_name(standard), drop[standard], "in OpenFOAM"),
            "best_plan": plan_name(best), "best_drop_pct": drop[best],
            "method": "smallest plan within 90% of the best OpenFOAM drop",
        }
        out.update(_technique_notes(recs[()], out["recommended_plan"]))
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
        if self.patient is not None and any("run now" in str(r.get("source", "")) for r in
                                            [pred or {}, self.results.get("surgery") or {}, *self.results.get("what_if", [])]):
            flags.append({"id": "fast_openfoam_run", "severity": "info",
                          "text": "Some results come from OpenFOAM runs made now with fewer iterations than the stored runs."})
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
        "This real patient's nasal resistance per side and region from OpenFOAM on their CT airway (Pa*s/cm^3).",
        {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}),
    "patient_what_if": _tool(
        "patient_what_if",
        "OpenFOAM resistance before and after one virtual surgery plan on the patient's CT airway.",
        {"type": "object", "properties": {
            "patient_id": {"type": "string"},
            "procedures": {"type": "array", "minItems": 1,
                           "items": {"type": "string", "enum": list(PATIENT_PROCEDURES)}}},
         "required": ["patient_id", "procedures"]}),
    "patient_surgery_options": _tool(
        "patient_surgery_options",
        "Virtual surgery on the patient's CT airway: turbinate reduction, valve repair, septoplasty and their "
        "combinations, each run in OpenFOAM; returns the recommended plan, the best plan and the look-at-the-scan plan.",
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
        "If the user asks what a specific procedure would do (turbinate reduction, valve repair, septoplasty, or "
        "widening the nasal valve, which is valve repair), call patient_what_if with those procedures. If the user "
        "asks which surgery or treatment to do, call patient_surgery_options once. Never state numbers yourself. "
        "When done, call finish."
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
        "You are the Explainer. Write a short plain-language summary (at most 120 words) of the results for a "
        "non-specialist. Use only numbers that appear in the results, copied exactly; do not compute new ones. "
        "Never mention field names or the JSON. Resistance is in Pa*s/cm^3; higher means harder to breathe "
        "through. Mention every flag listed."
    ),
}

# Added to the Explainer's prompt only when the results have that entry, so it never reads out
# instructions that don't apply.
EXPLAINER_PARTS = {
    "what_if": "For each what-if, say what changed and give the resistance before, after, and the percent change.",
    "surgery": "Lead with the recommended surgery: copy the recommended_result sentence, name the best_plan, "
               "then compare with the look_at_scan_result sentence. Keep the words lower and higher exactly as written. Give the why "
               "reasons, and the dominant_region_note if present. End by saying a surgeon makes the final decision.",
}


def explainer_prompt(facts: dict) -> str:
    return " ".join([PROMPTS["explainer"], *(part for key, part in EXPLAINER_PARTS.items() if facts.get(key))])


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
              extra_body: dict | None = None, stop_at_limit: bool = False, options: dict | None = None):
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
        kwargs.update(options or {})
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
    if "surgery" in facts:
        required.append(facts["surgery"]["recommended_drop_pct"])
    found = [float(t) for t in NUMBER.findall(text)]
    return [f"{v:g}" for v in required if not any(abs(f - v) <= 0.5 * 10 ** -3 + 1e-9 for f in found)]


def template_summary(facts: dict) -> str:
    """Code-written fallback when the Explainer's text fails the number check."""
    p = facts["prediction"]
    lines = [f"Total nasal resistance is {p['total_resistance']} Pa*s/cm^3 "
             f"(left {p['left_resistance']}, right {p['right_resistance']}), from the {p['source']}."]
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
        if s.get("best_plan"):
            lines.append(f"Best plan in OpenFOAM: {s['best_plan']}.")
    for f in facts.get("flags", []):
        lines.append(f["text"])
    return " ".join(lines)


def plan_name(plan) -> str:
    """'turbinate_reduction+valve_repair' or a tuple of procedures -> 'turbinate reduction + valve repair'."""
    parts = plan.split("+") if isinstance(plan, str) else list(plan)
    return " + ".join(p.replace("_", " ") for p in parts if p and p != "none") or "no surgery"


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


def load_patient_table(sim_path: Path = PATIENTS_PATH):
    """NasalSeg patient table (geometry and stored OpenFOAM results), or None when missing."""
    import pandas as pd

    return pd.read_csv(sim_path) if Path(sim_path).exists() else None


def load_patients(sim_path: Path = PATIENTS_PATH, model_path: Path = PATIENT_SURROGATE_PATH):
    """NasalSeg patient table and trained patient surrogate, or (None, None) when missing."""
    if not Path(sim_path).exists() or not Path(model_path).exists():
        return None, None
    import pandas as pd

    from .surrogate import PatientSurrogate
    return pd.read_csv(sim_path), PatientSurrogate.load(model_path)


def with_history(request: str, history: list[dict] | None, keep: int = 4) -> str:
    """The request with the last few turns of this session prepended as context."""
    if not history:
        return request
    lines = [f"Q: {h['question']}\nA: {h['answer']}" for h in history[-keep:]]
    return "Earlier in this session:\n" + "\n\n".join(lines) + f"\n\nNew request: {request}"


def run_pipeline(request: str, client=None, model: str = DEFAULT_MODEL, surrogate=None,
                 thinking: bool = False, patients=None, listener: Callable[[dict], None] | None = None,
                 labels_dir: Path | None = None, surgery=None, patient_flow=None, provider: str = "local",
                 history: list[dict] | None = None, api_key: str | None = None, openai_model: str | None = None,
                 local_url: str | None = None) -> dict:
    """Case -> Planner -> Critic -> Explainer. Returns the facts, the summary and the full agent log.
    With `patients` the Case agent may pick a real NasalSeg patient; that patient's resistance, what-ifs and
    surgery options come from OpenFOAM through `patient_flow` (by default the stored runs in `patients` and
    `surgery`, with `labels_dir` for the airways; no live runs).

    provider "local" uses the vLLM endpoint (`local_url`, else NASAL_LLM_URL); "openai" uses OpenAI's API with
    store=False, with `api_key` (else OPENAI_API_KEY) and `openai_model` (else NASAL_OPENAI_MODEL). `history` holds earlier {"question", "answer"} turns of this
    session; it lives only in the caller's memory and is sent as context with the new request."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    request = with_history(request, history)
    options, model_info = None, MODEL_INFO
    if provider == "openai":
        if client is None:
            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("add an OpenAI API key in Settings, or set OPENAI_API_KEY")
            client = OpenAI(api_key=key, base_url="https://api.openai.com/v1")  # never NASAL_LLM_URL
        model = model if model != DEFAULT_MODEL else (openai_model or OPENAI_MODEL)
        extra, options = None, {"store": False}
        model_info = {"name": model, "provider": "OpenAI API (store=False)", "total_params": "not published"}
    else:
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=local_url or DEFAULT_URL, api_key=os.environ.get("NASAL_LLM_KEY", "none"))
        extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if patient_flow is None and patients is not None:
        from .patient_flow import PatientFlow
        patient_flow = PatientFlow(patients, surgery, labels_dir)
    tools = Tools(surrogate=surrogate, patients=patients, flow=patient_flow)
    log = AgentLog(listener=listener)
    start = time.time()

    if patients is not None:
        run_agent(client, model, "case", PROMPTS["case_patient"], request, tools, ["set_patient", "set_case"], log,
                  final_tool=("set_patient", "set_case"), extra_body=extra, options=options)
    else:
        run_agent(client, model, "case", PROMPTS["case"], request, tools, ["set_case"], log, final_tool="set_case",
                  extra_body=extra, options=options)

    if tools.patient is not None:
        pid = tools.patient["patient"]
        run_agent(client, model, "planner", PROMPTS["planner_patient"], f"Request: {request}\nPatient: {pid}", tools,
                  ["patient_resistance", "patient_what_if", "patient_surgery_options", "finish"], log, final_tool="finish", extra_body=extra, options=options,
                  max_turns=PLANNER_MAX_TURNS, stop_at_limit=True)
        if "prediction" not in tools.results:
            log.add("planner", "fallback", reason="planner skipped patient_resistance")
            try:
                tools.patient_resistance({"patient_id": pid})
            except ToolError as err:  # no OpenFOAM result and no live runs: say so instead of guessing
                flag = {"id": "no_openfoam", "severity": "error", "text": str(err)}
                return {"request": request, "model": model_info, "facts": {"case": {"patient_id": pid}, "flags": [flag]},
                        "summary": f"No OpenFOAM result is available for patient {pid}: {err}.",
                        "summary_source": "code (no OpenFOAM result)", "seconds": round(time.time() - start, 2),
                        "log": log.steps}
        case_facts = {"patient_id": pid}
    else:
        case_json = json.dumps(asdict(tools.case))
        run_agent(client, model, "planner", PROMPTS["planner"], f"Request: {request}\nCase: {case_json}", tools,
                  ["predict_resistance", "formula_resistance", "what_if", "recommend_surgery", "finish"], log,
                  final_tool="finish",
                  extra_body=extra, options=options, max_turns=PLANNER_MAX_TURNS, stop_at_limit=True)
        if "prediction" not in tools.results:  # the plan must include a prediction; code enforces it
            log.add("planner", "fallback", reason="planner skipped predict_resistance")
            tools.predict_resistance(asdict(tools.case))
        case_facts = asdict(tools.case)

    flags_chosen = run_agent(client, model, "critic", PROMPTS["critic"], "Review the results.", tools,
                             ["review_results", "check_parameters", "report_flags"], log, final_tool="report_flags",
                             extra_body=extra, options=options)
    chosen = set(flags_chosen.get("flag_ids", [])) | {"model_scope"}
    chosen |= {f["id"] for f in tools.results["flags"] if f["severity"] == "error"}  # errors are never dropped
    facts = {"case": case_facts, **{k: v for k, v in tools.results.items() if k != "flags"},
             "flags": [f for f in tools.results["flags"] if f["id"] in chosen]}

    summary, source = None, "explainer"
    for attempt in range(2):
        prompt = f"Results JSON:\n{json.dumps(facts, indent=1)}"
        if attempt:
            prompt += f"\nYour previous text had problems: {problem}. Rewrite it."
        text = run_agent(client, model, "explainer", explainer_prompt(facts), prompt, tools, [], log,
                         extra_body=extra, options=options)
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
    return {"request": request, "model": model_info, "facts": facts, "summary": summary,
            "summary_source": source, "seconds": round(time.time() - start, 2), "log": log.steps}
