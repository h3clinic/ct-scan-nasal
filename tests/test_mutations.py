"""Pathology operators and case generation on a synthetic airway (no dataset or OpenFOAM needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import SPACING, fake_labels  # noqa: E402

from nasal import anatomy, mutations  # noqa: E402
from nasal.decision import patient as ps  # noqa: E402


def airway():
    lab = fake_labels()
    lab[13:17, 2:22, 5:13] = 4  # a roomier left cavity, so it can take a few layers
    lab[3:7, 2:22, 5:13] = 3
    return anatomy.airway(lab)


def test_each_pathology_narrows_its_own_region_and_side():
    tags, cuts = airway()
    names = anatomy.region_names(tags)
    for kind, region in (("turbinate_hypertrophy", "turbinate"), ("valve_collapse", "valve"), ("choanal_narrowing", "choana")):
        out = mutations.OPERATORS[kind](tags, cuts, SPACING, "left")
        removed = (tags > 0) & (out == 0)
        assert removed.any(), kind
        assert set(np.unique(names[removed])) <= {f"left_{region}"}, kind


def test_reduction_roughly_undoes_hypertrophy():
    tags, cuts = airway()
    narrowed = mutations.turbinate_hypertrophy(tags, cuts, SPACING, "left")
    restored = ps.operate(narrowed, cuts, SPACING, "turbinate_reduction", side="left")
    # region bounds move a little once the wall moves, so the inverse is approximate (here 20 of 80 voxels differ)
    assert ((restored > 0) != (tags > 0)).sum() <= 0.35 * ((tags > 0) & (narrowed == 0)).sum()


def test_septal_bulge_trades_air_one_for_one_in_a_central_patch():
    tags, cuts = airway()
    out = mutations.septal_deviation(tags, cuts, SPACING, "left")
    left, right = anatomy.LABELS["left"], anatomy.LABELS["right"]
    lost, gained = (tags == left).sum() - (out == left).sum(), (out == right).sum() - (tags == right).sum()
    assert lost > 0 and lost == gained  # the septum moves, keeping its thickness
    changed = np.argwhere(out != tags)
    rows = ps.septum_rows(tags, SPACING)
    assert changed[:, 1].min() > rows["j"].min() and changed[:, 1].max() < rows["j"].max()  # local, not end to end


def test_severity_is_lowered_rather_than_closing_a_side():
    tags, cuts = airway()
    out, applied = mutations.apply(tags, cuts, SPACING, [mutations.Pathology("valve_collapse", "left", 3)])
    assert mutations.still_open(out, cuts) and 0 <= applied[0].severity <= 3
    many = [mutations.Pathology("choanal_narrowing", "right", 3)] * 6
    out, applied = mutations.apply(tags, cuts, SPACING, many)
    assert mutations.still_open(out, cuts)
    assert sum(p.severity for p in applied) < 18  # the airway could not take all of it


def test_cases_are_reproducible_and_round_trip():
    a = mutations.generate(50, ["P001", "P002", "P003"], seed=7)
    b = mutations.generate(50, ["P003", "P002", "P001"], seed=7)
    assert [c.to_json() for c in a] == [c.to_json() for c in b]
    assert len({c.to_json() for c in a}) > 40
    assert all(1 <= len(c.pathologies) <= 3 for c in a)
    assert mutations.Case.from_json(a[0].to_json()) == a[0]
