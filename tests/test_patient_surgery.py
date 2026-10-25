"""Virtual surgery on voxel airways (synthetic label volume; no dataset or OpenFOAM needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import SPACING, fake_labels  # noqa: E402

from nasal import anatomy  # noqa: E402
from nasal.decision import patient as ps  # noqa: E402

L, R = anatomy.LABELS["left"], anatomy.LABELS["right"]


def airway(left_wider=False):
    lab = fake_labels()
    if left_wider:
        lab[16, 2:22, 6:12] = 4  # left cavity 4 voxels wide instead of 3
    return anatomy.airway(lab)


def test_valve_repair_grows_only_the_valve_regions():
    tags, cuts = airway()
    names = anatomy.region_names(tags)
    out = ps.valve_repair(tags, cuts, SPACING)
    added = (out > 0) & (tags == 0)
    assert added.any()
    ys = np.argwhere(added)[:, 1]
    valve_ys = np.argwhere(np.isin(names, ["left_valve", "right_valve"]))[:, 1]
    assert ys.min() >= valve_ys.min() - 0 and ys.max() <= valve_ys.max()


def test_turbinate_reduction_moves_the_lateral_walls_out():
    tags, cuts = airway()
    out = ps.turbinate_reduction(tags, cuts, SPACING)
    added = np.argwhere((out > 0) & (tags == 0))
    assert set(added[:, 0]) == {3, 16}  # right cavity grows to x=3, left to x=16; the septum is untouched
    assert (out[7:13] == tags[7:13]).all()


def bulging_septum():
    """The synthetic airway with its septum pushed 2 voxels toward the left cavity in a few turbinate rows."""
    tags, cuts = airway()
    rows = slice(10, 15)
    sub = tags[:, rows]
    sub[7:9][sub[7:9] == 0] = R  # the right cavity reaches into the old septum ...
    sub[13:15][sub[13:15] == L] = 0  # ... and the left cavity loses its medial voxels
    return tags, cuts, rows


def test_septoplasty_leaves_a_straight_septum_alone():
    tags, cuts = airway()
    assert np.allclose(ps.septum_rows(tags, SPACING)["deviation"], 0)
    assert np.array_equal(ps.septoplasty(tags, cuts, SPACING, depth=3), tags)


def test_septoplasty_moves_a_local_bulge_back_and_keeps_the_septum_thick():
    tags, cuts, rows = bulging_septum()
    dev = ps.septum_rows(tags, SPACING)
    assert dev["deviation"].max() >= 1.5  # the bulging rows sit about 2 voxels toward the left
    out = ps.septoplasty(tags, cuts, SPACING, depth=3)
    gained = (out == L) & (tags == 0)
    lost = (tags == R) & (out == 0)
    assert gained.sum() == lost.sum() > 0  # one cavity gains exactly what the other gives up
    assert set(np.argwhere(gained)[:, 1]) <= set(range(10, 15))  # only the deviated rows change
    assert set(np.argwhere(gained)[:, 0]) == {13, 14} and set(np.argwhere(lost)[:, 0]) == {7, 8}
    assert np.array_equal(out[:, :10], tags[:, :10]) and np.array_equal(out[:, 15:], tags[:, 15:])
    one = ps.septoplasty(tags, cuts, SPACING, depth=1)  # depth caps the move; it never overshoots the plane
    assert ((one == L) & (tags == 0)).sum() == gained.sum() // 2
    assert np.array_equal(ps.septoplasty(out, cuts, SPACING, depth=3), out)


def test_growth_never_breaches_a_one_voxel_septum():
    lab = fake_labels()
    lab[lab == 4] = 0
    lab[8:11, 2:22, 6:12] = 4  # left cavity right next to the right one: septum is x=7 only
    lab[4:16, 22:27, 6:12] = 5
    tags, cuts = anatomy.airway(lab)
    out = ps.apply_patient_plan(tags, cuts, SPACING, ps.PATIENT_PROCEDURES)
    touching = (out == R) & np.roll(out == L, -1, axis=0)  # a left voxel right next to a right voxel
    assert not touching.any()
    assert (out[7:9, 2:22, 6:12] == 0).any(axis=0).all()  # a septum voxel remains at every (y, z)


def test_rules_on_patients():
    tags, cuts = airway(left_wider=True)
    geometry = anatomy.measure(tags, SPACING)
    assert ps.standard_rule_patient(geometry) == ("turbinate_reduction", "septoplasty")
    # toy evaluator: resistance falls as the airway grows
    plan, predicted = ps.plus_plus_rule_patient(tags, cuts, SPACING, lambda t: {"total_resistance": 1 / (t > 0).sum()})
    assert plan and predicted[()] > min(predicted.values())
    assert set(predicted) == {(), *ps.candidate_plans()}


def test_compare_surgery_scores_rules_against_openfoam():
    rows = []
    for plan, cfd, sur in [("none", 1.0, 1.0), ("turbinate_reduction", 0.8, 0.85), ("valve_repair", 0.6, 0.7),
                           ("turbinate_reduction+valve_repair", 0.58, 0.6)]:
        rows.append({"patient": "P001", "plan": plan, "total_resistance": cfd, "surrogate_total": sur,
                     "standard_plan": "turbinate_reduction", "plus_plus_plan": "valve_repair"})
    out = ps.compare_surgery(pd.DataFrame(rows)).iloc[0]
    assert out.openfoam_best_plan == "turbinate_reduction+valve_repair"
    assert out.plus_plus_near_best and not out.standard_near_best
    assert out.standard_drop_pct == pytest.approx(20) and out.plus_plus_drop_pct == pytest.approx(40)
    # the fixed plan is scored as a baseline: here it is the best plan, and Workflow++ did not pick it
    assert out.fixed_drop_pct == pytest.approx(42) and out.fixed_near_best and not out.plus_plus_is_fixed


def test_every_plan_leaves_one_connected_airway():
    from scipy import ndimage

    lab = fake_labels()
    lab[16, 2:22, 6:12] = 4
    lab[14, 10:12, 6:7] = 0  # a notch, so eroding the medial layer would cut off a pocket
    tags, cuts = anatomy.airway(lab)
    for plan in ps.candidate_plans():
        out = ps.apply_patient_plan(tags, cuts, SPACING, plan)
        assert ndimage.label(out > 0)[1] == 1, plan


def test_parametric_operation_matches_the_fixed_procedure_at_one_layer():
    tags, cuts = airway()
    fixed = ps.turbinate_reduction(tags, cuts, SPACING)
    fixed[~anatomy._largest_component(fixed > 0)] = 0
    assert np.array_equal(ps.operate(tags, cuts, SPACING, "turbinate_reduction"), fixed)


def test_operation_side_and_depth():
    tags, cuts = airway()
    right = ps.operate(tags, cuts, SPACING, "turbinate_reduction", side="right")
    added = (right > 0) & (tags == 0)
    assert added.any() and (right[added] == R).all()
    one = ps.operate(tags, cuts, SPACING, "valve_repair", depth=1)
    two = ps.operate(tags, cuts, SPACING, "valve_repair", depth=2)
    assert (two > 0).sum() > (one > 0).sum() > (tags > 0).sum()


def test_widen_cuts_one_region_in_a_chosen_direction():
    tags, cuts = airway()
    names = anatomy.region_names(tags)
    out = ps.operate(tags, cuts, SPACING, "widen", side="left", region="choana", direction="vertical")
    added = np.argwhere((out > 0) & (tags == 0))
    assert len(added) and (out[tuple(added.T)] == L).all()
    choana_ys = np.argwhere(names == "left_choana")[:, 1]
    assert added[:, 1].min() >= choana_ys.min() and added[:, 0].min() >= np.argwhere(names == "left_choana")[:, 0].min()


@pytest.mark.parametrize("kwargs", [{"technique": "laser"}, {"technique": "valve_repair", "depth": 4},
                                    {"technique": "valve_repair", "side": "middle"}, {"technique": "widen"},
                                    {"technique": "widen", "region": "valve", "direction": "diagonal"}])
def test_operation_rejects_choices_outside_the_tools(kwargs):
    tags, cuts = airway()
    with pytest.raises(ValueError):
        ps.operate(tags, cuts, SPACING, **kwargs)


def test_depth_in_mm_follows_the_unequal_voxel_sizes():
    tags, cuts = airway()
    spacing = np.array([0.6, 0.6, 1.5])  # NasalSeg-like: thin in-plane, thick between slices
    assert ps.layers_for(1.2, spacing, 0) == 2 and ps.layers_for(1.2, spacing, 2) == 1
    assert ps.layers_for(0.6, spacing, 2) == 0  # under half a slice: no vertical change
    shallow = ps.operate(tags, cuts, spacing, "valve_repair", depth_mm=0.6)
    added = np.argwhere((shallow > 0) & (tags == 0))
    assert len(added) and set(added[:, 2]) <= set(np.nonzero(tags.any(axis=(0, 1)))[0])  # sideways only
    deeper = ps.operate(tags, cuts, spacing, "valve_repair", depth_mm=1.2)
    assert ((deeper > 0) & (tags == 0)).sum() > len(added)
    with pytest.raises(ValueError, match="under half a voxel"):
        ps.operate(tags, cuts, spacing, "widen", region="valve", direction="vertical", depth_mm=0.6)
    with pytest.raises(ValueError, match="depth_mm"):
        ps.operate(tags, cuts, spacing, "valve_repair", depth_mm=5.0)
    assert np.array_equal(ps.operate(tags, cuts, spacing, "valve_repair"),
                          ps.operate(tags, cuts, spacing, "valve_repair", depth=1))  # layers still work


def test_edited_volume_reports_added_and_removed_separately():
    tags, cuts, _ = bulging_septum()
    out = ps.septoplasty(tags, cuts, SPACING, depth=3)
    vol = ps.edited_volume(tags, out, SPACING)
    voxel = float(np.prod(SPACING))
    assert vol["added_mm3"] == vol["removed_mm3"] == pytest.approx(((out == L) & (tags == 0)).sum() * voxel, abs=0.1)


def test_no_surgery_is_right_when_no_plan_helps_enough():
    rows = [{"patient": "P002", "plan": plan, "total_resistance": cfd, "surrogate_total": cfd,
             "standard_plan": "turbinate_reduction", "plus_plus_plan": "none"}
            for plan, cfd in [("none", 1.0), ("turbinate_reduction", 0.98), ("turbinate_reduction+valve_repair", 0.97)]]
    out = ps.compare_surgery(pd.DataFrame(rows)).iloc[0]
    assert out.plus_plus_drop_pct == 0 and out.plus_plus_near_best  # best is 3%, under the 5% minimum benefit
    assert ps.choose_plan({(): 1.0, ("valve_repair",): 0.97}) == ()
    assert ps.choose_plan({(): 1.0, ("valve_repair",): 0.9}) == ("valve_repair",)


def test_rows_whose_cavity_edge_jumps_are_not_taken_for_the_septum():
    tags, cuts = airway()
    rows_before = ps.septum_rows(tags, SPACING)
    j = int(np.median(rows_before["j"]))
    # in one coronal row the left cavity's medial air is gone and only a pocket far out is left
    pocket = tags[13:16, j] == L
    tags[13:16, j][pocket] = 0
    tags[18, j][pocket.any(axis=0)] = L  # 5 voxels further out than the medial edge was
    after = ps.septum_rows(tags, SPACING)
    assert j not in set(after["j"])  # that row's "septum" would be far thicker than elsewhere
    assert len(after["j"]) == len(rows_before["j"]) - (rows_before["j"] == j).sum()
