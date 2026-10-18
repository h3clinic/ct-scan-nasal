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


def test_septoplasty_shifts_the_septum_toward_the_wider_side():
    tags, cuts = airway(left_wider=True)
    out = ps.septoplasty(tags, cuts, SPACING)
    assert ((tags == L) & (out == 0))[13].any()  # wide (left) cavity loses its medial layer
    assert ((tags == 0) & (out == R))[7].any()   # narrow (right) cavity gains one medially
    m0, m1 = anatomy.measure(tags, SPACING), anatomy.measure(out, SPACING)
    assert m1["right_mean_area_mm2"] > m0["right_mean_area_mm2"]
    assert m1["left_mean_area_mm2"] < m0["left_mean_area_mm2"]


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


def test_every_plan_leaves_one_connected_airway():
    from scipy import ndimage

    lab = fake_labels()
    lab[16, 2:22, 6:12] = 4
    lab[14, 10:12, 6:7] = 0  # a notch, so eroding the medial layer would cut off a pocket
    tags, cuts = anatomy.airway(lab)
    for plan in ps.candidate_plans():
        out = ps.apply_patient_plan(tags, cuts, SPACING, plan)
        assert ndimage.label(out > 0)[1] == 1, plan
