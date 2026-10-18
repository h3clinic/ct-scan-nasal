"""CT-to-airway tests on a small synthetic CT phantom (no dataset needed)."""
import numpy as np
import pytest

from nasal import anatomy
from nasal import segment as sg

pytest.importorskip("skimage")
SPACING = np.array([0.5, 0.5, 1.0])
TISSUE, AIR = 40, -1000


def phantom(shift: int = 0):
    """A head of tissue with outside air around it, two cavities along +y separated by a
    septum, joining a wide nasopharynx behind them, a lateral sinus joined to the right
    cavity by a one-voxel neck, and a thin air bridge over the septum.
    Axis 0 is x (+ is the patient's left), 1 is y (+ posterior), 2 is z (+ superior).
    `shift` moves the whole airway sideways (an off-centre head or deviated septum)."""
    ct = np.full((40, 50, 20), AIR, dtype=np.int16)
    ct[3:37, 2:48, 2:18] = TISSUE                    # the head
    a = 10 + shift
    expert = np.zeros(ct.shape, dtype=np.int8)
    expert[a + 4:a + 8, 5:30, 6:14] = 3                 # right cavity
    expert[a + 12:a + 16, 5:30, 6:14] = 4               # left cavity
    expert[a + 4:a + 16, 30:40, 4:14] = 5               # nasopharynx
    ct[expert > 0] = AIR
    ct[a - 6:a + 1, 10:20, 6:14] = AIR                  # sinus, lateral to the right cavity
    ct[a + 1:a + 4, 15, 10] = AIR                       # one-voxel neck joining it to the cavity
    ct[a + 8:a + 12, 12, 13] = AIR                      # thin bridge over the septum
    expert[a + 8:a + 12, 12, 13] = 0
    return ct, expert


def test_enclosed_air_drops_air_outside_the_head():
    ct, expert = phantom()
    inside = sg.enclosed_air(ct < sg.AIR_THRESHOLD_HU)
    assert not inside[:, :, 0].any() and not inside[0].any()
    assert inside[expert > 0].all()


def test_labels_follow_the_cavities_not_the_midline():
    for shift in (0, 6):  # with the airway off-centre the sides must not swap
        _, expert = phantom(shift)
        labels = sg.label_airway(expert > 0, mid=20.0)
        assert sg.dice(labels == 3, expert == 3) > 0.95
        assert sg.dice(labels == 4, expert == 4) > 0.95
        assert sg.dice(labels == 5, expert == 5) > 0.9


def test_a_thin_bridge_over_the_septum_does_not_start_the_nasopharynx():
    ct, expert = phantom()
    air = (expert > 0) | (sg.enclosed_air(ct < sg.AIR_THRESHOLD_HU) & (np.arange(ct.shape[1]) == 12)[None, :, None])
    labels = sg.label_airway(air, mid=20.0)
    assert not (labels[:, :30] == 5).any()


def test_watershed_separates_the_sinus_at_its_neck():
    ct, expert = phantom()
    reg = sg.air_regions(ct, SPACING)
    a = 10
    sinus_ids = set(np.unique(reg["regions"][a - 6:a + 1, 10:20, 6:14])) - {0}
    cavity_ids = set(np.unique(reg["regions"][a + 4:a + 8, 5:30, 6:14])) - {0}
    assert sinus_ids and not (sinus_ids & cavity_ids)


def test_features_targets_and_training_find_the_airway():
    pytest.importorskip("sklearn")
    cases = [phantom(s) for s in (0, 2, 4, 6)]
    regs = [sg.air_regions(ct, SPACING) for ct, _ in cases]
    feats = [sg.region_features(ct, SPACING, r) for (ct, _), r in zip(cases, regs)]
    assert all(f.shape[1] == len(sg.FEATURES) for f in feats)
    targets = [sg.region_targets(r, ex) for (_, ex), r in zip(cases, regs)]
    model = sg.train(feats, targets)
    ct, expert = phantom(3)
    labels = sg.segment(ct, SPACING, model)
    assert sg.compare(labels, expert)["dice_airway"] > 0.85
    tags, _ = anatomy.airway(labels)  # the output feeds the existing pipeline unchanged
    assert set(np.unique(tags)) == {0, 3, 4, 5}


def test_folds_keep_duplicate_scans_together():
    pids = [f"P{i:03d}" for i in range(1, 21)]
    fold_of = sg.folds(pids, k=5)
    assert fold_of["P003"] == fold_of["P002"] and fold_of["P010"] == fold_of["P009"]
    assert set(fold_of.values()) == set(range(5))


def test_written_labels_load_back(tmp_path):
    pytest.importorskip("nrrd")
    _, expert = phantom()
    origin = np.array([-10.0, -20.0, -5.0])
    sg.write_labels(tmp_path / "x_seg.nrrd", expert, SPACING, origin)
    back, spacing, org = anatomy.load_labels(tmp_path / "x_seg.nrrd")
    assert (back == expert).all() and np.allclose(spacing, SPACING) and np.allclose(org, origin)
