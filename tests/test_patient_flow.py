"""PatientFlow tests: stored OpenFOAM runs, the Workflow++ rule on simulated totals (no Docker needed)."""
import pandas as pd
import pytest

from nasal.patient_flow import PatientFlow, plus_plus_from_totals, run_settings


def test_run_settings_split_cores_between_cases():
    assert run_settings(cores=20) == (8, 2)
    assert run_settings(cores=20, procs=6) == (6, 3)
    assert run_settings(cores=4) == (2, 1)
    assert run_settings(cores=1) == (1, 1)


def test_plus_plus_picks_the_smallest_plan_near_the_best():
    tr, vr, sp = "turbinate_reduction", "valve_repair", "septoplasty"
    totals = {(): 1.0, (tr,): 0.89, (vr,): 0.97, (tr, vr): 0.88, (sp,): 1.1}
    assert plus_plus_from_totals(totals) == (tr,)  # 11% is within 90% of the best 12%
    assert plus_plus_from_totals({**totals, (tr,): 0.92}) == (tr, vr)
    assert plus_plus_from_totals({(): 1.0, (sp,): 1.1}) == ()  # nothing helps: no surgery


def test_stored_runs_win_and_missing_runs_are_not_guessed():
    patients = pd.DataFrame([{"patient": "P001", "total_resistance": 0.2}, {"patient": "P002", "total_resistance": float("nan")}])
    surgery = pd.DataFrame([{"patient": "P001", "plan": "none", "total_resistance": 0.2001},
                            {"patient": "P001", "plan": "valve_repair", "total_resistance": 0.19}])
    flow = PatientFlow(patients, surgery)
    recs = flow.records("P001", [(), ("valve_repair",)])
    assert recs[()]["total_resistance"] == 0.2 and recs[("valve_repair",)]["total_resistance"] == 0.19
    assert "stored" in recs[()]["source"]
    with pytest.raises(LookupError, match="no OpenFOAM run for none"):
        flow.records("P002", [()])
    with pytest.raises(LookupError, match="labels are not loaded"):
        PatientFlow(patients, surgery, live=True).records("P002", [()])
