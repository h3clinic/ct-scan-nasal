"""Reliability check for the agent pipeline: a fixed set of requests, scored by code.

Each request has an expectation the code can check (the patient id or parameter
direction the Case agent should set, and whether a what-if must be run). The
report counts completed runs, correct case choices, invalid tool calls and how
many the model repaired after the error came back, what-ifs run when asked, and
whether the Explainer's numbers passed the check first time, after the retry, or
fell back to the code template.
"""

import json
import time
from dataclasses import dataclass
from typing import Callable

from . import agents
from .params import PARAM_SPECS


@dataclass(frozen=True)
class Probe:
    request: str
    case_ok: Callable[[dict], bool]  # gets facts["case"]
    wants_what_if: bool = False


def _patient(pid: str):
    return lambda case: case.get("patient_id") == pid


def _params(check):
    return lambda case: "valve_angle_deg" in case and check(case)


def _no_patient(case: dict) -> bool:
    """For an unknown id: any valid case is fine, as long as it isn't a made-up patient."""
    return "patient_id" not in case or case["patient_id"].startswith("P") and case["patient_id"] != "P999"


PROBES = [
    Probe("Patient P022: which side is harder to breathe through?", _patient("P022")),
    Probe("Look at patient p039. What if the left side's narrowest section were 30% wider?", _patient("P039"), True),
    Probe("Show me NasalSeg patient 96 and widen the right narrowest section by 20%.", _patient("P096"), True),
    Probe("How obstructed is patient P064?", _patient("P064")),
    Probe("Patient P123, and what if its narrower side were widened by 40%?", _patient("P123"), True),
    Probe("Explain the airflow of patient P001.", _patient("P001")),
    Probe("Patient P999: how does it look?", _no_patient),
    Probe("Patient P022 but make the valve narrower by 20% on the left.", _patient("P022"), True),
    Probe("A normal, healthy nose.", _params(lambda c: abs(c["septum_deviation"]) < 0.2 and 12 <= c["valve_angle_deg"] <= 18)),
    Probe("Narrow nasal valve, about 9 degrees, everything else normal.", _params(lambda c: c["valve_angle_deg"] <= 11)),
    Probe("Septum strongly deviated to the right.", _params(lambda c: c["septum_deviation"] < -0.3)),
    Probe("Septum bends to the left a little and the turbinates are enlarged.",
          _params(lambda c: c["septum_deviation"] > 0 and c["turbinate_scale"] > 1.0)),
    Probe("Valve angle of 40 degrees.", _params(lambda c: PARAM_SPECS["valve_angle_deg"].low <= c["valve_angle_deg"]
                                                 <= PARAM_SPECS["valve_angle_deg"].high)),
    Probe("Big turbinates, 1.5 times normal. What if they were reduced to 0.9?",
          _params(lambda c: c["turbinate_scale"] >= 1.3), True),
    Probe("Valve 10 degrees, septum to the right 0.5. What if the valve were widened to 15 degrees?",
          _params(lambda c: c["valve_angle_deg"] <= 12 and c["septum_deviation"] < 0), True),
    Probe("turbinate scale 0.7, valve 18, septum 0", _params(lambda c: abs(c["turbinate_scale"] - 0.7) < 0.05)),
]


def score(probe: Probe, out: dict) -> dict:
    tool_steps = [s for s in out["log"] if s["kind"] == "tool"]
    errors = [s for s in tool_steps if isinstance(s.get("result"), dict) and "error" in s["result"]]
    checks = [s for s in out["log"] if s["kind"] == "number_check"]
    return {
        "request": probe.request,
        "case": out["facts"]["case"],
        "case_ok": bool(probe.case_ok(out["facts"]["case"])),
        "tool_calls": len(tool_steps),
        "tool_errors": len(errors),
        "what_if_ok": (not probe.wants_what_if) or bool(out["facts"].get("what_if")),
        "explainer": ("first try" if checks and not checks[0]["unsupported"] and not checks[0].get("missing")
                      else "after retry" if out["summary_source"] == "explainer" else "template fallback"),
        "seconds": out["seconds"],
    }


def run(probes=PROBES, **pipeline_kwargs) -> dict:
    rows = []
    for probe in probes:
        start = time.time()
        try:
            out = agents.run_pipeline(probe.request, **pipeline_kwargs)
            rows.append({"completed": True, **score(probe, out)})
        except Exception as err:  # a crashed run is a result, not a reason to stop
            rows.append({"completed": False, "request": probe.request, "case_ok": False, "tool_calls": 0,
                         "tool_errors": 0, "what_if_ok": False, "explainer": "crashed",
                         "seconds": round(time.time() - start, 2), "error": f"{type(err).__name__}: {err}"})
    n = len(rows)
    summary = {
        "requests": n,
        "completed": sum(r["completed"] for r in rows),
        "case_correct": sum(r["case_ok"] for r in rows),
        "what_if_when_asked": f"{sum(r['what_if_ok'] for r in rows if any(p.request == r['request'] and p.wants_what_if for p in probes))}"
                              f"/{sum(p.wants_what_if for p in probes)}",
        "tool_calls": sum(r["tool_calls"] for r in rows),
        "tool_errors_sent_back": sum(r["tool_errors"] for r in rows),
        "explainer_first_try": sum(r["explainer"] == "first try" for r in rows),
        "explainer_after_retry": sum(r["explainer"] == "after retry" for r in rows),
        "explainer_template_fallback": sum(r["explainer"] == "template fallback" for r in rows),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
    }
    return {"model": agents.MODEL_INFO, "summary": summary, "runs": rows}


def main_report(result: dict) -> str:
    return json.dumps(result["summary"], indent=2)
