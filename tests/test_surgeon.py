"""Surgeon agent tests: a scripted model and a stand-in for OpenFOAM on a synthetic airway."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_agents import ScriptedClient, call, reply  # noqa: E402
from test_anatomy import SPACING, fake_labels  # noqa: E402

from nasal import anatomy, surgeon  # noqa: E402
from nasal.params import REGIONS, SIDES  # noqa: E402


def fake_openfoam(tags):
    """Resistance falls as the airway grows: enough to tell plans apart, no solver needed."""
    side = round(1000.0 / (tags > 0).sum(), 4)
    rec = {"total_resistance": round(side / 2, 4), "total_flow_cm3_s": 1.0}
    for s in SIDES:
        rec[f"{s}_resistance"] = side
        for r, share in zip(REGIONS, (0.1, 0.2, 0.5, 0.2)):
            rec[f"{s}_{r}_resistance"] = round(side * share, 4)
    return rec


def session(events=None):
    tags, cuts = anatomy.airway(fake_labels())
    return surgeon.SurgerySession("P003", tags, cuts, SPACING, np.zeros(3), fake_openfoam,
                                  baseline=fake_openfoam(tags),
                                  listener=(lambda e, d: events.append(e)) if events is not None else None)


def test_session_operates_simulates_and_undoes():
    events = []
    s = session(events)
    before = s.airway_status({})
    assert before["operations"] == [] and before["current_openfoam"]["total_resistance"] == before["before_surgery"]["total_resistance"]
    out = s.operate({"technique": "turbinate_reduction", "side": "right", "depth": 2})
    assert out["voxels_added"] > 0 and out["simulated"] is False
    sim = s.simulate({})
    assert sim["total_change_pct_vs_before"] < 0 and s.runs == 1
    s.simulate({})
    assert s.runs == 1  # the same airway is not simulated twice
    s.undo({})
    assert np.array_equal(s.tags, s.original) and s.history == []
    assert events == ["operate", "simulating", "simulated", "simulated", "operate"]


def test_bad_operations_go_back_to_the_model():
    s = session()
    for args in ({"technique": "laser"}, {"technique": "widen", "region": "valve"}, {"technique": "valve_repair", "depth": 9}):
        try:
            s.operate(args)
        except surgeon.ToolError as err:
            assert str(err)
        else:
            raise AssertionError(f"accepted {args}")
    assert s.history == []


def test_surgeon_agent_runs_and_its_summary_is_number_checked():
    events = []
    s = session(events)
    after = fake_openfoam(  # what the plan below will produce
        __import__("nasal.decision.patient", fromlist=["x"]).operate(s.original, s.cuts, SPACING, "valve_repair", side="both", depth=1))
    client = ScriptedClient([
        reply(calls=[call("airway_status", {})]),
        reply(calls=[call("technique_info", {"technique": "valve_repair"}, 1)]),
        reply(calls=[call("operate", {"technique": "laser"}, 2)]),  # rejected, sent back
        reply(calls=[call("operate", {"technique": "valve_repair"}, 3)]),
        reply(calls=[call("finish_surgery", {"reason": "valve was the target"}, 4)]),  # simulated by code first
        reply(content="Made up 0.1234."),  # fails the number check
        reply(content=f"Valve repair lowered total resistance to {after['total_resistance']} Pa*s/cm^3."),
    ])
    out = surgeon.run_surgeon(s, "lower resistance", client, "test-model", provider="OpenRouter")
    assert out["result"]["operations"] == [{"technique": "valve_repair", "side": "both", "depth": 1}]
    assert out["result"]["after"]["total_resistance"] == after["total_resistance"] and out["result"]["total_change_pct"] < 0
    assert out["summary_source"] == "surgeon" and str(after["total_resistance"]) in out["summary"]
    assert "simulated" in events and events[-1] == "finished"
    assert all("store" not in r for r in client.requests)  # only OpenAI accepts store=False
    errors = [st for st in out["log"] if st["kind"] == "tool" and isinstance(st.get("result"), dict) and "error" in st["result"]]
    assert len(errors) == 1


def test_unfinished_agent_still_ends_on_a_simulated_airway():
    s = session()
    client = ScriptedClient([reply(calls=[call("operate", {"technique": "turbinate_reduction"}, i)]) for i in range(surgeon.MAX_TURNS)]
                            + [reply(content="x 9.99"), reply(content="y 8.88")])
    out = surgeon.run_surgeon(s, "go", client, "m")
    assert out["result"]["after"] and out["summary_source"].startswith("template")
    assert surgeon.summary_by_code(out["result"]).startswith("Operations kept: technique turbinate_reduction")


def test_finishing_without_trying_anything_is_refused():
    s = session()
    try:
        s.finish_surgery({})
    except surgeon.ToolError as err:
        assert "at least one operation" in str(err)
    else:
        raise AssertionError("finished without trying an operation")
    assert s.airway_status({})["highest_resistance_regions_before"][0]["region"] == "turbinate"
    s.operate({"technique": "turbinate_reduction"})
    s.simulate({})
    s.undo({})
    assert s.finish_surgery({})["operations"] == []  # tried, undone: keeping no surgery is allowed


def test_a_plan_kept_in_an_earlier_attempt_cannot_be_kept_again():
    s = session()
    s.operate({"technique": "valve_repair"})
    s.tried_before = {s.key(): 4.2}
    try:
        s.finish_surgery({})
    except surgeon.ToolError as err:
        assert "earlier attempt (4.2% lower)" in str(err)
    else:
        raise AssertionError("kept a repeated plan")
    s.operate({"technique": "turbinate_reduction", "side": "left"})
    assert s.finish_surgery({})["operations"][-1]["technique"] == "turbinate_reduction"
