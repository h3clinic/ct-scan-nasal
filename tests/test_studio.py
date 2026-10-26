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
    table = studio.plan_ranking(results)
    assert len(table) == len(ps.candidate_plans())
    assert table["plan"].iloc[0] == "turbinate reduction + valve repair"
    assert table["OpenFOAM drop %"].tolist()[:2] == pytest.approx([25.0, 10.0])
    assert table["OpenFOAM drop %"].tolist()[2:] == ["not run"] * (len(table) - 2)
    assert list(table.columns) == ["plan", "OpenFOAM drop %"]  # no placeholder standard-rule column
    assert set(studio.plan_ranking({})["OpenFOAM drop %"]) == {"not run"}



def test_surgeon_labels_show_depth_in_mm_or_layers():
    assert studio.surgeon_label({"technique": "valve_repair", "side": "both", "depth_mm": 1.2}) == "valve repair (both, 1.2 mm)"
    assert studio.surgeon_label({"technique": "turbinate_reduction", "side": "left", "depth": 2}) == "turbinate reduction (left, 2 layers)"
    assert studio.surgeon_label({"technique": "septoplasty", "side": "both", "depth_mm": 0.6}) == "septoplasty (0.6 mm)"
    assert studio.surgeon_label({"undo": {"technique": "widen", "side": "left", "depth": 1, "region": "choana",
                                          "direction": "vertical"}}) == "undo widen choana vertical (left, 1 layer)"


def test_actions_work_without_a_layout_or_nasalseg():
    actions = studio.make_actions()
    for name in ("reconstruct", "simulate_original", "plan_once", "plan_all", "show_original", "agent_operate", "ask",
                 "spectate", "tick", "toggle_play", "save_timeline", "load_timeline", "watch_current", "load_prefs",
                 "set_key", "run_setup_checks"):
        assert callable(actions[name]), name
    assert actions["ids"] == [] and actions["default"] is None
    st, *_, info, status = actions["reconstruct"](None, None)
    assert st == {} and "Upload a CT scan" in info
    app = studio.build_app()
    assert app.actions.keys() == actions.keys()
