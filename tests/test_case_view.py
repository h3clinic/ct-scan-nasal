"""Case viewer: per-pathology removed voxels, slice choice and the figure (synthetic airway, no dataset)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import SPACING, fake_labels  # noqa: E402

from nasal import anatomy, case_view, mutations  # noqa: E402


def labels():
    lab = fake_labels()
    lab[13:17, 2:22, 5:13] = 4
    lab[3:7, 2:22, 5:13] = 3
    return lab


def test_each_pathology_gets_the_voxels_it_removed():
    tags, cuts = anatomy.airway(labels())
    pathologies = [mutations.Pathology("turbinate_hypertrophy", "left", 1),
                   mutations.Pathology("valve_collapse", "right", 1)]
    deformed, parts = case_view.removed_by_pathology(tags, cuts, SPACING, pathologies)
    expected, _ = mutations.apply(tags, cuts, SPACING, pathologies)
    assert np.array_equal(deformed, expected)
    assert all(m.any() for _, m in parts) and not (parts[0][1] & parts[1][1]).any()
    removed = parts[0][1] | parts[1][1]
    assert np.array_equal(removed, (tags > 0) & (deformed == 0))


def test_slices_cover_every_region_and_the_most_changed_slice():
    tags, cuts = anatomy.airway(labels())
    _, parts = case_view.removed_by_pathology(tags, cuts, SPACING, [mutations.Pathology("choanal_narrowing", "left", 1)])
    names = [n for n, _ in case_view.slice_positions(tags, parts)]
    assert names[:2] == ["vestibule", "valve"] and "choana" in names
    js = [j for _, j in case_view.slice_positions(tags, parts)]
    assert js == sorted(js)


def test_figure_has_a_panel_per_slice():
    lab = labels()
    ct = np.where(lab > 0, -900.0, 40.0)
    case = mutations.Case("C99-00000", "P000", 0, [mutations.Pathology("valve_collapse", "left", 1)])
    fig = case_view.case_figure(case, ct, lab, SPACING, {"oracle_drop_pct": 10.0, "attempts": [{"drop_pct": 4.0}], "solved": False})
    tags, cuts = anatomy.airway(lab)
    _, parts = case_view.removed_by_pathology(tags, cuts, SPACING, case.pathologies)
    assert len(fig.axes) == len(case_view.slice_positions(tags, parts))
    assert "agent's best 4.0% lower" in fig._suptitle.get_text()
