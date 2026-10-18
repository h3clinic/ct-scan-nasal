"""Patient-airway tests on a small synthetic label volume (no dataset or OpenFOAM needed)."""
import numpy as np
import pytest

from nasal import anatomy

SPACING = np.array([0.5, 0.5, 1.0])
ORIGIN = np.array([-10.0, -20.0, -5.0])


def fake_labels():
    """Two cavities along +y (posterior) joined to a pharynx that drops in -z, plus lateral sinuses.
    Axis 0 is x (+ is the patient's left), 1 is y (+ posterior), 2 is z (+ superior)."""
    lab = np.zeros((20, 30, 14), dtype=np.int16)
    lab[4:7, 2:22, 6:12] = 3    # right cavity (low x)
    lab[13:16, 2:22, 6:12] = 4  # left cavity (high x)
    lab[4:16, 22:27, 6:12] = 5  # pharynx behind both cavities...
    lab[8:12, 22:27, 1:6] = 5   # ...dropping down to the outlet
    lab[0:3, 5:15, 6:12] = 1    # right sinus
    lab[17:20, 5:15, 6:12] = 2  # left sinus
    lab[4:7, 0:2, 9:10] = 3     # a thin tip that the inlet cut should remove
    return lab


def test_labels_are_checked_by_position():
    lab = fake_labels()
    assert anatomy.check_labels(lab)
    swapped = lab.copy()
    swapped[lab == 3], swapped[lab == 4] = 4, 3
    assert not anatomy.check_labels(swapped)


def test_airway_cuts_remove_thin_ends_and_sinuses():
    tags, cuts = anatomy.airway(fake_labels())
    assert set(np.unique(tags)) == {0, 3, 4, 5}
    assert cuts["inlet_right"] == 2 and cuts["inlet_left"] == 2
    assert not (tags[:, :2] > 0).any()
    assert cuts["outlet"] == 6  # the narrow lower tail (k < 6) is cut off


def test_voxel_mesh_is_closed_and_ordered():
    tags, cuts = anatomy.airway(fake_labels())
    mesh = anatomy.voxel_mesh(tags, SPACING, ORIGIN, cuts)
    n_int, owner, nb = mesh["n_internal"], mesh["owner"], mesh["neighbour"]
    assert mesh["n_cells"] == int((tags > 0).sum())
    assert (owner[:n_int] < nb).all()
    assert (np.diff(owner[:n_int]) >= 0).all()  # upper-triangular order
    pts = mesh["points"][mesh["faces"]]
    area = 0.5 * np.cross(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1])
    # every cell is closed: outward area vectors of its faces sum to zero
    total = np.zeros((mesh["n_cells"], 3))
    np.add.at(total, owner, area)
    np.add.at(total, nb, -area[:n_int])
    assert np.allclose(total, 0, atol=1e-12)
    patches = {p["name"]: p for p in mesh["patches"]}
    inlet_area = np.abs(area[patches["inlet_left"]["start"]:][:patches["inlet_left"]["n"]]).sum(axis=0)
    assert patches["inlet_left"]["n"] == 3 * 6  # x by z cross-section of the left cavity
    assert inlet_area[1] == pytest.approx(18 * 0.5 * 1.0 * 1e-6)  # faces point along -y
    assert patches["outlet"]["n"] == 12 * 5 and patches["wall"]["type"] == "wall"


def test_measure_reports_both_sides():
    tags, _ = anatomy.airway(fake_labels())
    m = anatomy.measure(tags, SPACING)
    assert m["left_min_area_mm2"] == pytest.approx(3 * 6 * 0.5 * 1.0)
    assert m["right_length_mm"] == pytest.approx(20 * 0.5)
    assert m["pharynx_volume_cm3"] > 0


def test_read_internal_field_parses_scalars_and_vectors(tmp_path):
    (tmp_path / "p").write_text("FoamFile{}\ninternalField   nonuniform List<scalar> \n3\n(\n1\n2.5\n-3e-2\n)\n;\n")
    (tmp_path / "U").write_text("internalField nonuniform List<vector> 2((1 2 3) (4 5 6));\n")
    assert anatomy.read_internal_field(tmp_path / "p").tolist() == [1, 2.5, -0.03]
    assert anatomy.read_internal_field(tmp_path / "U").shape == (2, 3)


def test_reversed_axis_is_flipped_on_load(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    lab = fake_labels()
    header = {"space": "left-posterior-superior", "space directions": np.diag([0.5, 0.5, -1.0]),
              "space origin": np.array([0.0, 0.0, 13.0])}
    nrrd.write(str(tmp_path / "P999_seg.nrrd"), lab[:, :, ::-1].copy(), header)
    data, spacing, origin = anatomy.load_labels(tmp_path / "P999_seg.nrrd")
    assert spacing.tolist() == [0.5, 0.5, 1.0] and origin[2] == pytest.approx(0.0)
    assert (data == lab).all()


def test_missing_cavity_is_a_clear_error():
    lab = fake_labels()
    lab[lab == 4] = 0
    with pytest.raises(ValueError, match="left"):
        anatomy.airway(lab)
