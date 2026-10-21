"""Studio helper tests (no Gradio, server, Docker or dataset needed)."""
import numpy as np
import pytest

from nasal import studio
from nasal.decision import patient as ps
from nasal.params import REGIONS, SIDES


def test_plan_from_choices_uses_candidate_order():
    plan = studio.plan_from_choices(["Valve repair", "Turbinate reduction"])
    assert plan == ("turbinate_reduction", "valve_repair")
    assert plan in ps.candidate_plans()
    assert studio.plan_from_choices([]) == ()


def test_changed_voxels_finds_added_and_removed():
    before = np.zeros((3, 3, 3), np.int8)
    before[1, 1, 1] = before[0, 0, 0] = 3
    after = before.copy()
    after[0, 0, 0] = 0
    after[2, 2, 2] = 4
    out = studio.changed_voxels(before, after, np.array([0.5, 0.5, 1.0]), np.array([10.0, 0, 0]))
    assert out["added"].tolist() == [[11.0, 1.0, 2.0]]
    assert out["removed"].tolist() == [[10.0, 0.0, 0.0]]


def _record(scale=1.0):
    rec = {f"{s}_{r}_resistance": 0.1 * scale for s in SIDES for r in REGIONS}
    rec.update({f"{s}_resistance": 0.4 * scale for s in SIDES}, total_resistance=0.2 * scale)
    return rec


def test_region_table_reports_change():
    table = studio.region_table(_record(), _record(0.9))
    assert len(table) == 2 * (len(REGIONS) + 1) + 1
    assert np.allclose(table["change %"], -10.0)
    assert "after" not in studio.region_table(_record())


def test_plan_ranking_lists_every_plan_with_openfoam_drops():
    results = {"none": {"total_resistance": 2.0}, "turbinate_reduction": {"total_resistance": 1.8},
               "turbinate_reduction+valve_repair": {"total_resistance": 1.5}}
    table = studio.plan_ranking(results, ("turbinate_reduction",))
    assert len(table) == len(ps.candidate_plans())
    assert table["plan"].iloc[0] == "turbinate reduction + valve repair"
    assert table["OpenFOAM drop %"].tolist()[:2] == pytest.approx([25.0, 10.0])
    assert table["OpenFOAM drop %"].tolist()[2:] == ["not run"] * (len(table) - 2)
    assert table["standard rule"].tolist().count("picks") == 1 and table["standard rule"].iloc[1] == "picks"
    assert set(studio.plan_ranking({}, ())["OpenFOAM drop %"]) == {"not run"}

