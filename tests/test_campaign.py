"""Campaign tests: synthetic base airway, a stand-in for OpenFOAM and a scripted agent (no dataset, Docker or model)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_agents import ScriptedClient, call, reply  # noqa: E402
from test_anatomy import SPACING, fake_labels  # noqa: E402
from test_surgeon import fake_openfoam  # noqa: E402

from nasal import campaign, mutations  # noqa: E402


def base(pid):
    lab = fake_labels()
    lab[13:17, 2:22, 5:13] = 4
    lab[3:7, 2:22, 5:13] = 3
    return lab, SPACING, np.zeros(3)


def executor(tags, spacing, origin, cuts, case_dir):
    return fake_openfoam(tags)


def operate_then_finish(ops):
    return [*(reply(calls=[call("operate", op, i)]) for i, op in enumerate(ops)),
            reply(calls=[call("finish_surgery", {}, 99)]), reply(content="Done.")]


CASE = mutations.Case("C00-00000", "P001", 0, [mutations.Pathology("turbinate_hypertrophy", "left", 1)])


def test_oracle_undoes_the_pathologies_in_reverse_order():
    applied = [mutations.Pathology("valve_collapse", "right", 2), mutations.Pathology("septal_deviation", "left", 1),
               mutations.Pathology("choanal_narrowing", "left", 0)]
    assert campaign.oracle_plan(applied) == [{"technique": "septoplasty", "depth": 1},
                                             {"technique": "valve_repair", "side": "right", "depth": 2}]


def test_a_good_plan_succeeds_first_time_and_is_remembered(tmp_path):
    client = ScriptedClient(operate_then_finish([{"technique": "turbinate_reduction", "side": "left", "depth": 1}]))
    rows = campaign.run([CASE], tmp_path, executor, client, "m", base=base, log=lambda m: None)
    assert rows[0]["solved"] and len(rows[0]["attempts"]) == 1 and rows[0]["oracle_drop_pct"] > 0
    memory = [json.loads(line) for line in (tmp_path / "experience.jsonl").open()]
    assert memory[0]["success"] and memory[0]["operations"][0]["technique"] == "turbinate_reduction"
    assert campaign.report(tmp_path)["solved_first_try"] == 1
    assert campaign.run([CASE], tmp_path, executor, client, "m", base=base) == []  # resumes: nothing left


def test_a_weak_plan_is_retried_with_feedback_and_past_lessons(tmp_path):
    experience = campaign.Experience(tmp_path / "experience.jsonl")
    experience.add({"case_id": "old", "features": [0.0] * 9, "summary": "the left side was harder",
                    "operations": [{"technique": "turbinate_reduction", "side": "left"}], "drop_pct": 9.0, "success": True})
    weak = {"technique": "valve_repair", "side": "right", "depth": 1}
    good = {"technique": "turbinate_reduction", "side": "left", "depth": 1}
    client = ScriptedClient(operate_then_finish([weak]) + operate_then_finish([good]))
    row = campaign.run_case(CASE, executor, client, "m", tmp_path / "cases", experience, base=base, log=lambda m: None)
    assert [a["success"] for a in row["attempts"]] == [False, True]
    second_goal = next(r for r in client.requests if "Earlier attempts on this airway" in r["messages"][1]["content"])
    text = second_goal["messages"][1]["content"]
    assert "Past attempts on similar airways" in text and "worked" in text and "fell short" in text
    assert "highest regional resistances" in text


def test_unsolvable_case_stops_after_the_attempt_limit(tmp_path):
    weak = {"technique": "valve_repair", "side": "right", "depth": 1}
    client = ScriptedClient(operate_then_finish([weak]) * 2)
    row = campaign.run_case(CASE, executor, client, "m", tmp_path, campaign.Experience(tmp_path / "e.jsonl"),
                            max_attempts=2, base=base, log=lambda m: None)
    assert not row["solved"] and len(row["attempts"]) == 2
