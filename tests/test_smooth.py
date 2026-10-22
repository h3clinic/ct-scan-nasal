"""Smooth-wall airway tests on a synthetic label volume (no dataset or OpenFOAM needed)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import SPACING, fake_labels  # noqa: E402

from nasal import anatomy, smooth  # noqa: E402


def _volume(verts, faces):
    return abs(np.einsum("ij,ij->i", verts[faces[:, 0]], np.cross(verts[faces[:, 1]], verts[faces[:, 2]])).sum() / 6)


def test_smooth_surface_is_closed_with_flat_openings_and_keeps_the_volume():
    tags, cuts = anatomy.airway(fake_labels())
    s = smooth.surface(tags, SPACING, np.zeros(3), cuts)
    patches = Counter(s["patch"])
    assert {"wall", "inlet_left", "inlet_right", "outlet"} <= set(patches)
    edges = np.sort(np.concatenate([s["faces"][:, [0, 1]], s["faces"][:, [1, 2]], s["faces"][:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()  # every edge shared by exactly two faces: closed, so snappyHexMesh can mesh it
    voxel = (tags > 0).sum() * np.prod(SPACING)
    assert abs(_volume(s["verts"], s["faces"]) / voxel - 1) < 0.25
    inlet = s["verts"][s["faces"][s["patch"] == "inlet_left"].ravel()]
    assert np.ptp(inlet[:, 1]) < 1e-6  # the nostril opening is a flat plane


def test_case_has_snappy_mesh_and_region_planes(tmp_path):
    tags, cuts = anatomy.airway(fake_labels())
    smooth.write_smooth_case(tmp_path / "case", tags, SPACING, np.zeros(3), cuts, processes=2, max_iterations=90)
    allrun = (tmp_path / "case" / "Allrun").read_text()
    assert "snappyHexMesh -overwrite" in allrun and "blockMesh" in allrun
    control = (tmp_path / "case" / "system" / "controlDict").read_text()
    assert all(f"{s}_b{i}" in control for s in ("left", "right") for i in (1, 2, 3))
    stl = (tmp_path / "case" / "constant" / "triSurface" / "airway.stl").read_text()
    assert all(f"solid {p}" in stl for p in ("wall", "inlet_left", "inlet_right", "outlet"))


def test_repair_fills_ct_air_gaps_but_not_tissue_or_sinuses():
    lab = fake_labels()
    lab[13:16, 21, 6:12] = 0  # an unlabeled slice between the left cavity and the pharynx...
    ct = np.full(lab.shape, 40.0)  # soft tissue everywhere...
    ct[lab > 0] = -800.0
    ct[13:16, 21, 6:12] = -850.0  # ...but the gap is air on the CT
    ct[0:3, 15:18, 6:12] = -900.0  # air next to the sinus, not next to the airway: must stay out
    fixed = smooth.repair_labels(lab, ct)
    assert (fixed[13:16, 21, 6:12] > 0).all() and set(np.unique(fixed[13:16, 21, 6:12])) <= {4, 5}
    assert (fixed[lab == 1] == 1).all() and (fixed[lab == 2] == 2).all()  # sinuses untouched
    assert (fixed[0:3, 15:18, 6:12] == 0).all()
    assert (fixed[(lab == 0) & (ct > -300)] == 0).all()  # tissue never becomes airway


def test_ct_wall_follows_the_ct_and_leaves_out_the_sinuses():
    lab = fake_labels()
    tags, cuts = anatomy.airway(lab)
    ct = np.where(lab > 0, -950.0, 40.0)  # sinuses are air on the CT too
    s = smooth.surface(tags, SPACING, np.zeros(3), cuts, ct=ct, sinus=np.isin(lab, [1, 2]))
    assert {"wall", "inlet_left", "inlet_right", "outlet"} <= set(s["patch"])
    edges = np.sort(np.concatenate([s["faces"][:, [0, 1]], s["faces"][:, [1, 2]], s["faces"][:, [2, 0]]]), axis=1)
    assert (np.unique(edges, axis=0, return_counts=True)[1] == 2).all()
    voxel = (tags > 0).sum() * np.prod(SPACING)
    assert 0.6 < _volume(s["verts"], s["faces"]) / voxel < 1.25  # a sharp CT puts the wall near the label's
    x_mm = s["verts"][:, 0]
    assert x_mm.min() > 2.5 * SPACING[0] and x_mm.max() < 16.5 * SPACING[0]  # never out into the sinuses (x 0-3, 17-20)
