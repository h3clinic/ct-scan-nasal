"""Four small-model agents that call plain-code tools. Research only, not medical advice.

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

from .params import PARAM_SPECS, REGIONS, SIDES, CaseParams
from .resistance import REFERENCE_PRESSURE, evaluate

DEFAULT_URL = os.environ.get("NASAL_LLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("NASAL_LLM_MODEL", "nemotron")
MODEL_INFO = {"name": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "total_params": "30B", "active_params": "3B"}
SURROGATE_PATH = Path("data/surrogate.pt")
MAX_TURNS = 6
DISAGREEMENT_FLAG_PCT = 25.0  # surrogate vs formula, a model-agreement check, not a clinical threshold

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
        if self.case is None:
            flags.append({"id": "no_case", "severity": "error", "text": "No valid case was set."})
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
        flags.append({"id": "research_only", "severity": "info",
                      "text": "Synthetic, idealized geometry. Research only, not medical advice."})
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
    "planner": (
        "You are the Planner. A case has been set. Decide which tools answer the user's request. Always call "
        "predict_resistance and formula_resistance for the case. If the user asks about a change (for example "
        "widening the valve or reducing the turbinate), call what_if with only the parameters that change. "
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
        "Mention every flag listed. "
        "Do not recommend treatment. End with: Research only, not medical advice."
    ),
}


@dataclass
class AgentLog:
    steps: list = field(default_factory=list)

    def add(self, agent: str, kind: str, **data) -> None:
        self.steps.append({"agent": agent, "kind": kind, "t": round(time.time(), 3), **data})


def run_agent(client, model: str, agent: str, system: str, user: str, tools: Tools, tool_names: list[str],
              log: AgentLog, final_tool: str | None = None, max_turns: int = MAX_TURNS,
              extra_body: dict | None = None):
    """Chat loop: execute tool calls in code, send errors back, stop at final_tool.
    Returns the final tool's arguments, or the model's text if no final tool."""
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
            if final_tool is None:
                return reply.content or ""
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append({"role": "user", "content": f"Call {final_tool} to finish."})
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
                if name == final_tool:
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
    found = [float(t) for t in NUMBER.findall(text)]
    return [f"{v:g}" for v in required if not any(abs(f - v) <= 0.5 * 10 ** -3 + 1e-9 for f in found)]


def template_summary(facts: dict) -> str:
    """Code-written fallback when the Explainer's text fails the number check."""
    p = facts["prediction"]
    lines = [f"Predicted total nasal resistance is {p['total_resistance']} Pa*s/cm^3 "
             f"(left {p['left_resistance']}, right {p['right_resistance']}), from the {p['source']}."]
    for w in facts.get("what_if", []):
        lines.append(f"With the change '{w['label']}', total resistance goes from {w['total_resistance_before']} "
                     f"to {w['total_resistance_after']} ({w['total_change_pct']}%).")
    for f in facts.get("flags", []):
        lines.append(f["text"])
    if not any(f["id"] == "research_only" for f in facts.get("flags", [])):
        lines.append("Research only, not medical advice.")
    return " ".join(lines)


# ---------------------------------------------------------------- pipeline


def load_surrogate(path: Path = SURROGATE_PATH):
    if not Path(path).exists():
        return None
    try:
        from .surrogate import Surrogate
        return Surrogate.load(path)
    except ImportError:
        return None


def run_pipeline(request: str, client=None, model: str = DEFAULT_MODEL, surrogate=None,
                 thinking: bool = False) -> dict:
    """Case -> Planner -> Critic -> Explainer. Returns the facts, the summary and the full agent log."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=DEFAULT_URL, api_key=os.environ.get("NASAL_LLM_KEY", "none"))
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    tools, log = Tools(surrogate=surrogate), AgentLog()
    start = time.time()

    run_agent(client, model, "case", PROMPTS["case"], request, tools, ["set_case"], log, final_tool="set_case",
              extra_body=extra)

    case_json = json.dumps(asdict(tools.case))
    run_agent(client, model, "planner", PROMPTS["planner"], f"Request: {request}\nCase: {case_json}", tools,
              ["predict_resistance", "formula_resistance", "what_if", "finish"], log, final_tool="finish",
              extra_body=extra)
    if "prediction" not in tools.results:  # the plan must include a prediction; code enforces it
        log.add("planner", "fallback", reason="planner skipped predict_resistance")
        tools.predict_resistance(asdict(tools.case))

    flags_chosen = run_agent(client, model, "critic", PROMPTS["critic"], "Review the results.", tools,
                             ["review_results", "check_parameters", "report_flags"], log, final_tool="report_flags",
                             extra_body=extra)
    chosen = set(flags_chosen.get("flag_ids", [])) | {"research_only"}
    chosen |= {f["id"] for f in tools.results["flags"] if f["severity"] == "error"}  # errors are never dropped
    facts = {"case": asdict(tools.case), **{k: v for k, v in tools.results.items() if k != "flags"},
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
