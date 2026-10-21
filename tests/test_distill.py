"""Distillation tests: facts, checks, student data and the local answer path (no Ollama or GPU needed)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from test_agents import ScriptedClient, patient_tables, reply  # noqa: E402

from nasal import agents, distill  # noqa: E402
from nasal.patient_flow import PatientFlow  # noqa: E402


def _tools_for(pid="P003"):
    patients, surgery = patient_tables()
    patients.loc[patients.patient == pid, "left_turbinate_resistance"] = 0.5  # left turbinate highest
    patients.loc[patients.patient == pid, "left_resistance"] = 0.68
    tools = agents.Tools(patients=patients, flow=PatientFlow(patients, surgery))
    tools.set_patient({"patient_id": pid})
    tools.patient_resistance({"patient_id": pid})
    return tools, patients, surgery


def test_comparisons_are_computed_by_code():
    tools, patients, _ = _tools_for()
    facts = distill.facts_for(tools, "P003", patients.total_resistance)
    c = facts["comparison"]
    assert c["harder_side"] == "left" and c["highest_region"] == "left turbinate"
    assert c["harder_side_times_easier_side"] == round(0.68 / 0.6, 2)
    assert c["simulated_patients"] == 4 and c["patient_total_resistance_percentile"] == 75  # 3 of 4 lower


def test_number_check_allows_method_constants_only():
    facts = {"prediction": {"total_resistance": 0.2633}}
    assert distill.check_answer("Total 0.2633 Pa*s/cm^3; each layer is 0.6 mm; the 90% rule.", facts) == ""
    assert "0.31" in distill.check_answer("Total 0.2633, maybe 0.31 after.", facts)


def test_student_data_swaps_in_the_short_prompt(tmp_path):
    row = {"patient": "P001", "split": "train", "messages": [
        {"role": "system", "content": distill.SYSTEM}, {"role": "user", "content": "Results JSON:\n{}\n\nQuestion: q"},
        {"role": "assistant", "content": "a"}]}
    src = tmp_path / "examples.jsonl"
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert distill.student_examples(src, tmp_path / "student.jsonl") == 1
    out = json.loads((tmp_path / "student.jsonl").read_text(encoding="utf-8"))
    assert out["messages"][0]["content"] == distill.STUDENT_SYSTEM and out["messages"][1:] == row["messages"][1:]
    assert len(distill.STUDENT_SYSTEM) < len(distill.SYSTEM) / 5
    assert "empty nose syndrome" in distill.SYSTEM and not any(ch.isdigit() for ch in distill.PROCEDURE_NOTES)


def test_local_answer_picks_tools_from_the_question_and_falls_back_to_code():
    _, patients, surgery = _tools_for()
    flow = PatientFlow(patients, surgery)
    good = ScriptedClient([reply(content="Total resistance is 0.3 Pa*s/cm^3; with turbinate reduction it is 0.268, "
                                         "10.7% lower in OpenFOAM.")])
    out = distill.answer_locally("What would a turbinate reduction do?", "P003", patients, flow, client=good)
    assert out["steps"][:3] == ["patient_resistance", "patient_what_if", "patient_surgery_options"]
    assert out["summary_source"] == "distilled Qwen" and out["facts"]["what_if"][0]["total_resistance_after"] == 0.268
    assert good.requests[0]["messages"][0]["content"] == distill.STUDENT_SYSTEM
    bad = ScriptedClient([reply(content="It is 9.99."), reply(content="Still 8.88.")])
    out = distill.answer_locally("How is the airflow?", "P003", patients, flow, client=bad)
    assert out["steps"] == ["patient_resistance", "number check: failed", "number check: failed"]
    assert out["summary_source"].startswith("template") and "0.3" in out["summary"]


def test_local_answer_without_openfoam_says_so():
    patients = pd.DataFrame([{"patient": "P001", "total_resistance": float("nan")}])
    out = distill.answer_locally("q", "P001", patients, PatientFlow(patients, None), client=NS())
    assert out["summary_source"] == "code" and "No OpenFOAM result" in out["summary"]
