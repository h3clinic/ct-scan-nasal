"""Agent pipeline tests with a scripted stand-in for the model (no server needed)."""
import json
from types import SimpleNamespace as NS

import pytest

from nasal import agents
from nasal.params import CaseParams
from nasal.resistance import evaluate


def call(name, args, i=0):
    raw = args if isinstance(args, str) else json.dumps(args)
    return NS(id=f"c{i}", function=NS(name=name, arguments=raw))


def reply(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class ScriptedClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


CASE = {"valve_angle_deg": 9.0, "turbinate_scale": 1.0, "septum_deviation": 0.5}


def happy_script(explainer_texts):
    return [
        reply(calls=[call("set_case", CASE)]),
        reply(calls=[call("predict_resistance", CASE, 1), call("formula_resistance", CASE, 2)]),
        reply(calls=[call("what_if", {"label": "wider valve", "valve_angle_deg": 15}, 3)]),
        reply(calls=[call("finish", {"reason": "done"}, 4)]),
        reply(calls=[call("review_results", {}, 5)]),
        reply(calls=[call("report_flags", {"flag_ids": ["no_surrogate"]}, 6)]),
        *[reply(content=t) for t in explainer_texts],
    ]


def test_pipeline_runs_and_numbers_come_from_code():
    expected = round(evaluate(CaseParams(**CASE))["total_resistance"], 4)
    after = round(evaluate(CaseParams(15.0, 1.0, 0.5))["total_resistance"], 4)
    client = ScriptedClient(happy_script([
        f"Total resistance is {expected} Pa*s/cm^3.",  # leaves out the what-if: must be retried
        f"Total resistance is {expected} Pa*s/cm^3, or {after} with a wider valve.",
    ]))
    out = agents.run_pipeline("narrow valve, septum to the left", client=client)
    assert out["facts"]["prediction"]["total_resistance"] == expected
    assert out["summary_source"] == "explainer"
    assert {f["id"] for f in out["facts"]["flags"]} >= {"model_scope"}
    assert out["facts"]["what_if"][0]["total_resistance_after"] < expected
    agents_seen = {s["agent"] for s in out["log"]}
    assert agents_seen == {"case", "planner", "critic", "explainer"}


def test_invented_numbers_fall_back_to_the_template():
    client = ScriptedClient(happy_script(["Resistance is 0.9999.", "Still 0.12345678 here."]))
    out = agents.run_pipeline("narrow valve", client=client)
    assert out["summary_source"].startswith("template")
    assert not agents.unsupported_numbers(out["summary"], out["facts"])


def test_bad_tool_arguments_are_sent_back_for_repair():
    tools, log = agents.Tools(), agents.AgentLog()
    client = ScriptedClient([
        reply(calls=[call("set_case", "{not json")]),
        reply(calls=[call("set_case", {**CASE, "valve_angle_deg": 45}, 1)]),
        reply(calls=[call("set_case", CASE, 2)]),
    ])
    args = agents.run_agent(client, "m", "case", "sys", "user", tools, ["set_case"], log, final_tool="set_case")
    assert args == CASE and tools.case == CaseParams(**CASE)
    errors = [s["result"]["error"] for s in log.steps if s["kind"] == "tool" and "error" in s["result"]]
    assert "not valid JSON" in errors[0] and "out of range" in errors[1]
    sent_back = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(sent_back) == 2


def test_critic_cannot_cite_unknown_flags():
    tools, log = agents.Tools(), agents.AgentLog()
    client = ScriptedClient([
        reply(calls=[call("review_results", {})]),
        reply(calls=[call("report_flags", {"flag_ids": ["made_up"]}, 1)]),
        reply(calls=[call("report_flags", {"flag_ids": ["no_case"]}, 2)]),
    ])
    args = agents.run_agent(client, "m", "critic", "sys", "user", tools,
                            ["review_results", "report_flags"], log, final_tool="report_flags")
    assert args["flag_ids"] == ["no_case"]


def test_unsupported_numbers():
    facts = {"prediction": {"total_resistance": 0.0579, "left_resistance": 0.1163}}
    assert agents.unsupported_numbers("about 0.058 and 0.1163, at 150 Pa", facts) == []
    assert agents.unsupported_numbers("about 0.07", facts) == ["0.07"]


def test_tool_schemas_follow_param_specs():
    props = agents.SCHEMAS["set_case"]["function"]["parameters"]["properties"]
    assert props["valve_angle_deg"]["minimum"] == 8.0 and props["septum_deviation"]["maximum"] == 1.0


def test_patient_mode_selects_patient_and_runs_patient_tools():
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_surrogate import _fake_patients

    from nasal.surrogate import PatientSurrogate

    table = _fake_patients(30)
    model = PatientSurrogate.fit(table)
    before = round(model.predict(table.iloc[3].to_dict())["total_resistance"], 4)
    client = ScriptedClient([
        reply(calls=[call("set_patient", {"patient_id": "p999"})]),  # unknown: error goes back to the model
        reply(calls=[call("set_patient", {"patient_id": "P003"}, 1)]),
        reply(calls=[call("patient_resistance", {"patient_id": "P003"}, 2),
                     call("patient_what_if", {"patient_id": "P003", "side": "left", "narrowest_area_change_pct": 30}, 3)]),
        reply(calls=[call("finish", {}, 4)]),
        reply(calls=[call("review_results", {}, 5)]),
        reply(calls=[call("report_flags", {"flag_ids": ["model_scope"]}, 6)]),
        reply(content="placeholder"),  # no numbers: fails the check, retried
        reply(content="placeholder"),  # fails again: code writes the template summary
    ])
    out = agents.run_pipeline("Patient P003: what if the left narrowest section were 30% wider?", client=client,
                              patients=table, patient_surrogate=model)
    assert out["facts"]["case"] == {"patient_id": "P003"}
    assert out["facts"]["prediction"]["total_resistance"] == before
    assert out["facts"]["what_if"][0]["total_resistance_after"] < before
    assert "Real CT airway" in out["facts"]["flags"][-1]["text"]
    assert out["summary_source"].startswith("template") and str(before) in out["summary"]


def test_agent_eval_scores_a_run():
    from nasal import agent_eval

    client = ScriptedClient(happy_script(["Total resistance is 0.0001."] * 2))
    probe = agent_eval.Probe("narrow valve", lambda case: case.get("valve_angle_deg") == 9.0, wants_what_if=True)
    result = agent_eval.run([probe], client=client)
    s = result["summary"]
    assert s["completed"] == 1 and s["case_correct"] == 1 and s["what_if_when_asked"] == "1/1"
    assert s["explainer_template_fallback"] == 1 and s["tool_errors_sent_back"] == 0


def test_planner_that_never_finishes_keeps_its_results():
    what_if = {"label": "wider valve", "valve_angle_deg": 15}
    client = ScriptedClient([
        reply(calls=[call("set_case", CASE)]),
        reply(calls=[call("predict_resistance", CASE, 1)]),
        *[reply(calls=[call("what_if", what_if, 2 + i)]) for i in range(agents.PLANNER_MAX_TURNS - 1)],
        reply(calls=[call("review_results", {}, 20)]),
        reply(calls=[call("report_flags", {"flag_ids": ["no_surrogate"]}, 21)]),
        reply(content="placeholder"),
        reply(content="placeholder"),
    ])
    out = agents.run_pipeline("valve 9 degrees, what if wider?", client=client)
    assert "prediction" in out["facts"]
    assert any(s["kind"] == "fallback" and s["agent"] == "planner" for s in out["log"])


def test_surgery_tool_needs_labels_and_template_names_the_plan():
    tools = agents.Tools(patients=__import__("pandas").DataFrame([{"patient": "P001"}]), patient_surrogate=object())
    with pytest.raises(agents.ToolError):
        tools.patient_surgery_options({"patient_id": "P001"})
    facts = {"prediction": {"total_resistance": 0.2, "left_resistance": 0.4, "right_resistance": 0.4, "source": "s"},
             "surgery": {"recommended_plan": "turbinate reduction + valve repair", "recommended_drop_pct": 6.0,
                         "look_at_scan_plan": "turbinate reduction", "look_at_scan_drop_pct": 3.8,
                         "openfoam_check": {"recommended_plan_drop_pct": 14.1, "best_plan": "valve repair",
                                            "best_drop_pct": 15.0}}}
    text = agents.template_summary(facts)
    assert "turbinate reduction + valve repair" in text and "14.1%" in text
    assert agents.missing_numbers(text, facts) == []
    assert agents.plan_name("turbinate_reduction+valve_repair") == "turbinate reduction + valve repair"
