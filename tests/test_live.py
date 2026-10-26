"""Live OpenFOAM view: sampling setup, reading a running case and the figures (no Docker needed)."""
import numpy as np
import pytest

from nasal import anatomy, live
from nasal.resistance import AIR_DENSITY
from nasal.simulate import case_files


def small_airway():
    """Two cavities (tags 3 and 4) joined by a nasopharynx (5), as `anatomy.airway` would cut it."""
    tags = np.zeros((8, 12, 6), dtype=np.int8)
    tags[1:3, 1:9, 2:4] = anatomy.LABELS["right"]
    tags[5:7, 1:9, 2:4] = anatomy.LABELS["left"]
    tags[1:7, 9:11, 1:4] = anatomy.LABELS["pharynx"]
    cuts = {"inlet_right": 1, "inlet_left": 1, "outlet": 1}
    return tags, cuts, np.array([0.5, 0.5, 1.5]), np.array([-10.0, 20.0, 5.0])


def test_planes_go_through_each_cavity_and_off_the_cell_centres():
    tags, _, spacing, origin = small_airway()
    planes = {name: (p, n) for name, p, n in live.planes(tags, spacing, origin)}
    assert set(planes) == {"sagittal_left", "sagittal_right", "axial"}
    x_right = planes["sagittal_right"][0][0] * 1e3
    assert x_right == pytest.approx(origin[0] + 1.75 * spacing[0])  # median voxel 1.5 of 1..2, a quarter voxel on
    assert planes["axial"][1] == (0, 0, 1)


def test_live_sampling_is_added_only_when_asked(tmp_path):
    tags, cuts, spacing, origin = small_airway()
    assert "live" not in case_files(1, inlets=("inlet_left", "inlet_right"), sample_planes=False, mesh_commands=[])["system/controlDict"]
    anatomy.write_airway_case(tmp_path / "plain", tags, spacing, origin, cuts, 1, max_iterations=90)
    assert "writeInterval 25;" not in (tmp_path / "plain" / "system" / "controlDict").read_text()
    anatomy.write_airway_case(tmp_path / "live", tags, spacing, origin, cuts, 1, max_iterations=90, live_every=25)
    control = (tmp_path / "live" / "system" / "controlDict").read_text()
    assert "writeInterval 25;" in control and "sagittal_left" in control and "patches (wall)" in control


LOG = """Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.01, No Iterations 2
GAMG:  Solving for p, Initial residual = 0.5, Final residual = 0.01, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.05, Final residual = 0.001, No Iterations 3
Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.2, Final residual = 0.01, No Iterations 2
GAMG:  Solving for p, Initial residual = 0.1, Final residual = 0.01, No Iterations 3
"""


def test_log_gives_the_iteration_and_the_first_residual_of_each_field():
    iteration, residuals = live.parse_log(LOG)
    assert iteration == 2
    assert residuals["p"] == [(1, 0.5), (2, 0.1)] and residuals["Ux"] == [(1, 1.0), (2, 0.2)]
    assert live.parse_log("") == (0, {})


def write_raw(path, field, points_m, values):
    cols = "x y z  " + (field if values.shape[1] == 1 else " ".join(f"{field}_{c}" for c in "xyz"))
    rows = "\n".join(" ".join(f"{v:g}" for v in (*p, *val)) for p, val in zip(points_m, values))
    path.write_text(f"# {field}  FACE_DATA {len(points_m)}\n# {cols}\n{rows}\n")


def fake_running_case(case, iterations=(25, 50), incomplete_last=False):
    (case / "log.simpleFoam").write_text(LOG.replace("Time = 2", "Time = 60"))
    for inlet, phi in (("inlet_left", -4e-4), ("inlet_right", -6e-4)):
        d = case / "postProcessing" / f"{inlet}Flow" / "0"
        d.mkdir(parents=True)
        d.joinpath("surfaceFieldValue.dat").write_text(
            "# Time sum(phi)\n" + "".join(f"{i}\t{phi * min(1, i / 30):.6e}\n" for i in range(1, 61)))
    pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    for it in iterations:
        d = case / "postProcessing" / "live" / str(it)
        d.mkdir(parents=True)
        write_raw(d / "p_wall.raw", "p", pts, np.array([[100.0], [10.0]]))
        write_raw(d / "p_axial.raw", "p", pts, np.array([[50.0], [5.0]]))
        if not (incomplete_last and it == iterations[-1]):  # OpenFOAM writes every p before any U
            write_raw(d / "U_axial.raw", "U", pts, np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]))


def test_poll_reads_flows_resistance_and_the_newest_complete_sample(tmp_path):
    fake_running_case(tmp_path, incomplete_last=True)
    snap = live.LiveCase(tmp_path, max_iterations=90).poll()
    assert snap["stage"] == "solving" and snap["iteration"] == 60
    h = snap["history"]
    assert h["left_flow"][-1] == pytest.approx(400.0) and h["right_flow"][-1] == pytest.approx(600.0)
    # the inlet pressure has ramped to 150 Pa by iteration 30 (90 // 3): 150 / 1000 cm3/s
    assert h["total_resistance"][-1] == pytest.approx(0.15)
    fields = snap["fields"]
    assert fields["iteration"] == 25  # 50 has no U yet
    assert fields["wall_pressure"] == pytest.approx([100 * AIR_DENSITY, 10 * AIR_DENSITY])
    assert fields["wall_points"][1] == pytest.approx([1.0, 0.0, 0.0])  # mm
    assert fields["planes"]["axial"][1] == pytest.approx([5.0, 1.0])
    assert (tmp_path / "postProcessing" / "live" / "50").exists()  # newer and incomplete: kept


def test_older_samples_are_deleted_once_a_newer_one_is_read(tmp_path):
    fake_running_case(tmp_path)
    case = live.LiveCase(tmp_path, max_iterations=90)
    assert case.poll()["fields"]["iteration"] == 50
    assert not (tmp_path / "postProcessing" / "live" / "25").exists()


def test_meshing_stage_and_status_text(tmp_path):
    snap = live.LiveCase(tmp_path, 900).poll()
    assert snap["stage"] == "meshing" and snap["history"] is None and snap["fields"] is None
    assert "building the mesh" in live.status_text(snap)
    fake_running_case(tmp_path)
    text = live.status_text(live.LiveCase(tmp_path, 90).poll())
    assert "iteration 60/90" in text and "flow 1000 cm³/s" in text and "0.1500" in text
    early = live.status_text(live.LiveCase(tmp_path, 900).poll())  # ramp ends at 300
    assert "still ramping up" in early


def test_figures_build_with_and_without_a_sample(tmp_path):
    fake_running_case(tmp_path)
    snap = live.LiveCase(tmp_path, 90).poll()
    surface = {"verts": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), "faces": np.array([[0, 1, 2]])}
    fig = live.wall_figure(surface, snap, "live")
    assert fig.data[0].intensity is not None and len(fig.data) == 2  # wall plus the axial plane
    assert len(live.wall_figure(surface, {"fields": None}, "x").data) == 1
    chart = live.progress_figure(snap)
    assert {t.name for t in chart.data} >= {"resistance", "p", "Ux"}
