"""Surgeon agent: plans virtual surgery on a patient's CT airway, operates with the parametric techniques in
`nasal.decision.patient.operate`, checks each candidate airway in OpenFOAM, and finishes with the best result
it verified.

The language model only chooses tool calls. Code applies every operation to the voxel airway, OpenFOAM computes
every resistance, and the final summary must pass the same number check as the Explainer (one retry, then a
summary written by code). Each operation and each simulation is reported to a listener, so an app can redraw
the airway as the agent changes it. The technique notes are a general textbook summary that no ENT has reviewed,
and the operations' sizes (one to three voxel layers, about 0.6 mm each) are modeling choices, not surgical
prescriptions.
"""

import json
import time
from typing import Callable

import numpy as np

from . import agents, anatomy
from .agents import ToolError, _summary, _tool
from .decision import patient as ps
from .params import REGIONS, SIDES

MAX_TURNS = 24
MAX_SIMULATIONS = 6  # each OpenFOAM run takes minutes

TECHNIQUE_NOTES = {
    "turbinate_reduction": (
        "Shrinks the inferior turbinates to widen the main airflow channel (submucous resection, microdebrider, "
        "radiofrequency or coblation reduction, outfracture), through the nostrils with an endoscope. Risks: "
        "bleeding, crusting and dryness, adhesions, regrowth, and empty nose syndrome when too much tissue is "
        "removed. In the model: pushes the lateral wall of the turbinate region outward by `depth` layers."),
    "valve_repair": (
        "Widens or stiffens the narrowest front part of the nose (spreader grafts, alar batten grafts, flaring "
        "sutures, lateral crural strut grafts) through an open or closed rhinoplasty approach. Risks: change in "
        "appearance, graft shifting or showing, infection, persistent obstruction. In the model: widens the valve "
        "region sideways and vertically by `depth` layers."),
    "septoplasty": (
        "Straightens a deviated septum through an incision inside the nose, lifting the lining and removing or "
        "reshaping bent cartilage and bone while keeping an L-shaped support strut. Risks: bleeding, septal "
        "hematoma, perforation, saddle nose, numb upper teeth, rarely a cerebrospinal fluid leak. In the model: "
        "moves the septum toward the wider side by `depth` layers (the voxel version removes more airway than it "
        "adds, so it can raise resistance)."),
    "widen": (
        "A general cut that removes tissue around one region of one or both cavities (vestibule, valve, "
        "turbinate or choana) in a chosen direction. It stands for the part of a real procedure that opens that "
        "region; name the real technique when you explain it. In the model: grows that region by `depth` layers."),
}

SURGEON_PROMPT = (
    "You are a virtual nasal surgeon working on one patient's CT airway in a research model (rigid walls, one "
    "steady breath at 150 Pa; operations change the voxel airway by about 0.6 mm per layer). Goal: lower the "
    "nasal resistance with the least surgery, where it matters for this patient. Work like this: call "
    "airway_status to see where the resistance sits (by side and region); call technique_info before using a "
    "technique you have not used; operate with the smallest change that targets the main resistance; call "
    "simulate to check the result in OpenFOAM; undo operations that do not help. If the region that held the "
    "most resistance still does after a simulation, operate on it again or deeper (depth 2 or 3) before adding "
    "shallow cuts elsewhere. Which operation acts on which region: valve -> valve_repair; turbinate -> "
    "turbinate_reduction; choana or vestibule -> widen with that region; a septum pushed toward one side -> "
    "septoplasty (it can also raise resistance, so check it). Operate on the side where the resistance is. Try at most "
    "a few plans, since "
    "each simulation takes minutes. Finish with finish_surgery naming the state to keep. Never state numbers "
    "yourself: every number comes from the tools."
)
SUMMARY_PROMPT = (
    "Write a short summary (at most 150 words) for a clinician of the virtual surgery below: the operations kept, "
    "the OpenFOAM resistance before and after, what each operation changed, and the main risks. Use only numbers "
    "from the JSON, copied exactly; resistance is in Pa*s/cm^3."
)


class SurgerySession:
    """The working airway, its history of operations and every OpenFOAM result, as tools for the agent.

    `simulate_fn(tags) -> record` runs OpenFOAM on a tag volume; `baseline` is the untouched airway's OpenFOAM
    record if one is stored. `listener(event, data)` sees every operation and simulation as it happens."""

    def __init__(self, name: str, tags: np.ndarray, cuts: dict, spacing: np.ndarray, origin: np.ndarray,
                 simulate_fn: Callable[[np.ndarray], dict], baseline: dict | None = None,
                 listener: Callable[[str, dict], None] | None = None):
        self.name, self.cuts, self.spacing, self.origin = name, cuts, spacing, origin
        self.original, self.tags = tags, tags
        self.history: list[dict] = []
        self.simulate_fn, self.listener = simulate_fn, listener or (lambda event, data: None)
        self.simulated: dict[str, dict] = {}  # plan key -> OpenFOAM record
        if baseline is not None:
            self.simulated[self.key()] = baseline
        self.results: dict = {}
        self.runs = 0  # OpenFOAM runs made in this session
        self.finished: dict | None = None
        self.tried_before: dict[str, float] = {}  # plan key -> % drop, from earlier attempts on the same case

    def key(self, history: list[dict] | None = None) -> str:
        return json.dumps([h["operation"] for h in (self.history if history is None else history)], sort_keys=True)

    def _emit(self, event: str, **data) -> None:
        self.listener(event, {"session": self, **data})

    def _changes(self) -> dict:
        return {"voxels_added": int(((self.tags > 0) & (self.original == 0)).sum()),
                "voxels_removed": int(((self.tags == 0) & (self.original > 0)).sum())}

    # ---- tools

    def airway_status(self, args: dict) -> dict:
        geometry = anatomy.measure(self.tags, self.spacing)
        out = {"patient_id": self.name, "operations": [h["operation"] for h in self.history], **self._changes(),
               "geometry": {k: round(v, 2) for k, v in geometry.items() if k.endswith(("_mm2", "_cm3"))}}
        base = self.simulated.get(self.key([]))
        if base:
            out["before_surgery"] = _summary(base)
            ranked = sorted(((s, r) for s in SIDES for r in REGIONS), key=lambda sr: -base[f"{sr[0]}_{sr[1]}_resistance"])
            out["highest_resistance_regions_before"] = [
                {"side": s, "region": r, "resistance": round(base[f"{s}_{r}_resistance"], 4)} for s, r in ranked[:3]]
        current = self.simulated.get(self.key())
        out["current_openfoam"] = _summary(current) if current else "not simulated yet: call simulate"
        return out

    def technique_info(self, args: dict) -> dict:
        technique = args.get("technique")
        if technique not in TECHNIQUE_NOTES:
            raise ToolError(f"technique must be one of {list(TECHNIQUE_NOTES)}")
        return {"technique": technique, "notes": TECHNIQUE_NOTES[technique], "max_depth": ps.MAX_DEPTH,
                "regions": list(ps.REGION_NAMES), "directions": sorted(ps.DIRECTIONS)}

    def operate(self, args: dict) -> dict:
        op = {k: args[k] for k in ("technique", "side", "depth", "region", "direction") if args.get(k) is not None}
        op.setdefault("side", "both")
        op.setdefault("depth", 1)
        if op.get("technique") != "widen":  # region and direction only steer the general cut
            op.pop("region", None)
            op.pop("direction", None)
        try:
            new = ps.operate(self.tags, self.cuts, self.spacing, **op)
        except ValueError as err:
            raise ToolError(str(err)) from err
        added = int(((new > 0) & (self.tags == 0)).sum())
        removed = int(((new == 0) & (self.tags > 0)).sum())
        if not added and not removed:
            raise ToolError("that operation changed nothing here (the region may already be at its limit)")
        self.tags = new
        self.history.append({"operation": op, "voxels_added": added, "voxels_removed": removed})
        self._emit("operate", operation=op)
        return {"applied": op, "voxels_added": added, "voxels_removed": removed, **{f"total_{k}": v for k, v in self._changes().items()},
                "operations_so_far": len(self.history), "simulated": self.key() in self.simulated}

    def undo(self, args: dict) -> dict:
        if not self.history:
            raise ToolError("nothing to undo")
        undone = self.history.pop()
        self.tags = self.original
        for h in self.history:
            self.tags = ps.operate(self.tags, self.cuts, self.spacing, **h["operation"])
        self._emit("operate", operation={"undo": undone["operation"]})
        return {"undone": undone["operation"], "operations_left": [h["operation"] for h in self.history]}

    def simulate(self, args: dict) -> dict:
        key = self.key()
        if key not in self.simulated:
            if self.runs >= MAX_SIMULATIONS:
                raise ToolError(f"simulation limit reached ({MAX_SIMULATIONS}); finish with a state already simulated")
            self._emit("simulating")
            try:
                self.simulated[key] = self.simulate_fn(self.tags)
                self.runs += 1
            except Exception as err:  # Docker down, mesh failure: tell the model instead of crashing
                raise ToolError(f"OpenFOAM failed: {type(err).__name__}") from err
        record = self.simulated[key]
        out = {"operations": [h["operation"] for h in self.history], **_summary(record)}
        base = self.simulated.get(self.key([]))
        if base and key != self.key([]):
            out["total_change_pct_vs_before"] = round(100 * (record["total_resistance"] / base["total_resistance"] - 1), 1)
            out["region_change_pct_vs_before"] = {
                f"{s} {r}": round(100 * (record[f"{s}_{r}_resistance"] / base[f"{s}_{r}_resistance"] - 1), 1)
                for s in SIDES for r in REGIONS}
        self._emit("simulated", record=record)
        return out

    def finish_surgery(self, args: dict, force: bool = False) -> dict:
        if not force and self.key() in self.tried_before:
            raise ToolError(f"this exact plan was kept in an earlier attempt ({self.tried_before[self.key()]}% lower); "
                            "change it: another technique, side, region or depth")
        tried = bool(self.history) or any(k != self.key([]) for k in self.simulated)  # an operated plan gets simulated below
        if not tried and not force:
            raise ToolError("check at least one operation in OpenFOAM before finishing: operate, then simulate "
                            "(you can undo it afterwards if it does not help)")
        if self.key() not in self.simulated:
            self.simulate({})  # never finish on an unverified airway
        self.finished = {"operations": [h["operation"] for h in self.history], **self._changes(),
                         "after": _summary(self.simulated[self.key()])}
        base = self.simulated.get(self.key([]))
        if base:
            self.finished["before"] = _summary(base)
            self.finished["total_change_pct"] = round(
                100 * (self.finished["after"]["total_resistance"] / base["total_resistance"] - 1), 1)
        self._emit("finished", result=self.finished)
        return self.finished


SURGEON_TOOLS = ["airway_status", "technique_info", "operate", "undo", "simulate", "finish_surgery"]
agents.SCHEMAS.update({
    "airway_status": _tool("airway_status", "Current airway: operations so far, voxels changed, cross-sections, and "
                           "OpenFOAM resistance per side and region before surgery and now (if simulated).",
                           {"type": "object", "properties": {}}),
    "technique_info": _tool("technique_info", "How a technique is done, its risks, and what it changes in the model.",
                            {"type": "object", "properties": {"technique": {"type": "string", "enum": list(TECHNIQUE_NOTES)}},
                             "required": ["technique"]}),
    "operate": _tool("operate", "Apply one virtual operation to the current airway, on top of earlier ones.", {
        "type": "object", "properties": {
            "technique": {"type": "string", "enum": list(ps.TECHNIQUES)},
            "side": {"type": "string", "enum": ["left", "right", "both"]},
            "depth": {"type": "integer", "minimum": 1, "maximum": ps.MAX_DEPTH, "description": "voxel layers, about 0.6 mm each"},
            "region": {"type": "string", "enum": list(ps.REGION_NAMES), "description": "widen only"},
            "direction": {"type": "string", "enum": sorted(ps.DIRECTIONS), "description": "widen only"}},
        "required": ["technique"]}),
    "undo": _tool("undo", "Remove the last operation.", {"type": "object", "properties": {}}),
    "simulate": _tool("simulate", f"Run OpenFOAM on the current airway (several minutes; at most {MAX_SIMULATIONS} "
                      "per session) and return its resistance and the change from before surgery.",
                      {"type": "object", "properties": {}}),
    "finish_surgery": _tool("finish_surgery", "Finish and keep the current airway (it is simulated first if needed).",
                            {"type": "object", "properties": {"reason": {"type": "string"}}}),
})


def run_surgeon(session: SurgerySession, goal: str, client, model: str, provider: str = "OpenAI",
                listener: Callable[[dict], None] | None = None) -> dict:
    """Let the model operate until it finishes; returns the result, a checked summary and the agent log."""
    from .providers import request_options

    log = agents.AgentLog(listener=listener)
    start = time.time()
    opts = request_options(provider)
    agents.run_agent(client, model, "surgeon", SURGEON_PROMPT, f"Patient {session.name}. Goal: {goal}", session,
                     SURGEON_TOOLS, log, final_tool="finish_surgery", max_turns=MAX_TURNS, stop_at_limit=True,
                     options=opts)
    if session.finished is None:
        log.add("surgeon", "fallback", reason="the agent did not finish; keeping the current airway")
        session.finish_surgery({}, force=True)
    facts = {"surgery": session.finished, "history": session.history,
             "technique_notes": {h["operation"]["technique"]: TECHNIQUE_NOTES[h["operation"]["technique"]]
                                 for h in session.history}}
    summary, source = None, "surgeon"
    for attempt in range(2):
        text = agents.run_agent(client, model, "surgeon", SUMMARY_PROMPT, json.dumps(facts, indent=1), session, [],
                                log, options=opts)
        bad = agents.unsupported_numbers(text, facts, (0.6,))
        log.add("surgeon", "number_check", unsupported=bad, missing=[])
        if not bad:
            summary = text
            break
    if summary is None:
        summary, source = summary_by_code(session.finished), "template (the summary failed the number check)"
    return {"result": session.finished, "summary": summary, "summary_source": source, "log": log.steps,
            "seconds": round(time.time() - start, 1)}


def summary_by_code(finished: dict) -> str:
    ops = "; ".join(", ".join(f"{k} {v}" for k, v in op.items()) for op in finished["operations"]) or "no operation"
    text = f"Operations kept: {ops}. OpenFOAM total resistance after: {finished['after']['total_resistance']} Pa*s/cm^3"
    if "before" in finished:
        text += f", before: {finished['before']['total_resistance']} ({finished['total_change_pct']}%)"
    return text + f". Voxels added: {finished['voxels_added']}, removed: {finished['voxels_removed']}."
