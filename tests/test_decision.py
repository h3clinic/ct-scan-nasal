import pandas as pd

from nasal.cases import build_table, sample_cases
from nasal.decision import PROCEDURES, apply_plan, plus_plus_rule, reclassify, standard_rule, summarize
from nasal.params import CaseParams
from nasal.resistance import evaluate


def test_procedures_stay_in_range_and_open_their_region():
    p = CaseParams(valve_angle_deg=10.0, turbinate_scale=1.5, septum_deviation=0.8)
    before = evaluate(p)  # deviation narrows the left side
    for name in PROCEDURES:
        assert not apply_plan(p, (name,)).out_of_range()
    after = {name: evaluate(apply_plan(p, (name,))) for name in PROCEDURES}
    assert after["septoplasty"]["left_resistance"] < before["left_resistance"]
    assert after["valve_repair"]["left_valve_resistance"] < before["left_valve_resistance"]
    # The baseline books the larger valve-to-cavity expansion loss against the valve,
    # so turbinate reduction lowers the turbinate share but can raise the valve share.
    assert after["turbinate_reduction"]["left_turbinate_resistance"] < before["left_turbinate_resistance"]


def test_standard_rule_always_reduces_turbinates():
    assert standard_rule(CaseParams()) == ("turbinate_reduction",)
    assert standard_rule(CaseParams(septum_deviation=-0.9)) == ("turbinate_reduction", "septoplasty")


def test_plus_plus_targets_a_narrow_valve():
    p = CaseParams(valve_angle_deg=8.0, turbinate_scale=0.7, septum_deviation=0.0)
    assert "valve_repair" in plus_plus_rule(p)


def test_recommendation_is_labeled_and_ranked():
    from nasal.decision.recommend import RESEARCH_ONLY, recommend

    out = recommend(CaseParams(valve_angle_deg=9.0, turbinate_scale=1.3, septum_deviation=0.6), evaluate, "formula")
    assert out["label"] == RESEARCH_ONLY
    assert out["thresholds"]["status"].startswith("placeholder")
    drops = [o["drop_pct"] for o in out["options"]]
    assert drops == sorted(drops, reverse=True)
    assert out["suggested"]["drop_pct"] >= 0.9 * drops[0]
    assert len(out["why"]) == len(out["suggested"]["plan"].split("+"))


def test_reclassification_table():
    table = reclassify(build_table(sample_cases(20, seed=1)))
    assert len(table) == 20
    assert (table.plus_plus_after <= table.resistance_before).all()
    assert set(summarize(table).columns) >= {"changed_pct", "plus_plus_drop_pct"}
    assert isinstance(table, pd.DataFrame)
