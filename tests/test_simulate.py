import numpy as np
import pandas as pd
import pytest

from nasal import simulate as sim
from nasal.cases import build_table, sample_cases
from nasal.cli import select_cases
from nasal.geometry import REGION_LENGTH, airway
from nasal.params import PARAM_SPECS, REGIONS, CaseParams
from nasal.resistance import AIR_DENSITY, CM3, REFERENCE_PRESSURE, evaluate


def test_region_boundaries_follow_region_lengths():
    lengths = [REGION_LENGTH[r] for r in REGIONS]
    assert sim.region_boundaries() == pytest.approx(list(np.cumsum(lengths)[:-1]))


def test_case_files_set_reference_pressure_and_planes():
    files = sim.case_files(processes=4)
    ramp = f"({sim.RAMP_ITERATIONS} {REFERENCE_PRESSURE / AIR_DENSITY:.6g})"
    assert f"p0 table ((0 0) {ramp})" in files["0/p"]
    for i, x in enumerate(sim.region_boundaries(), start=1):
        assert f"x{i} {{ type cuttingPlane; point ({x:.6g} 0 0)" in files["system/controlDict"]
    assert "numberOfSubdomains 4" in files["system/decomposeParDict"]
    assert "mpirun --allow-run-as-root --bind-to none -np 4 simpleFoam -parallel" in files["Allrun"]
    assert f"nu {sim.NU:.6g}" in files["constant/transportProperties"]


def test_case_files_turbulence_options():
    rans = sim.case_files(turbulence="kOmegaSST")
    assert {"0/k", "0/omega", "0/nut"} <= set(rans)
    assert "RASModel kOmegaSST" in rans["constant/turbulenceProperties"]
    laminar = sim.case_files(processes=1, turbulence="laminar")
    assert "0/k" not in laminar and "simulationType laminar" in laminar["constant/turbulenceProperties"]
    assert "mpirun" not in laminar["Allrun"]
    with pytest.raises(ValueError):
        sim.case_files(turbulence="kEpsilonMagic")


def _write_plane(path, values, area):
    cols = values if values.ndim == 2 else values[:, None]
    rows = np.hstack([np.zeros((len(area), 3)), cols, area])
    np.savetxt(path, rows, header="test plane")


def _fake_case(root, flow, plane_pt, iterations, max_iterations):
    """postProcessing output for a uniform axial flow through 4-face planes."""
    for patch, sign in (("inletFlow", -1), ("outletFlow", 1)):
        d = root / "postProcessing" / patch / "0"
        d.mkdir(parents=True)
        lines = [f"{i} {sign * flow:.10e}" for i in range(1, iterations + 1)]
        (d / "surfaceFieldValue.dat").write_text("# Time sum(phi)\n" + "\n".join(lines))
    planes = root / "postProcessing" / "planes" / str(iterations)
    planes.mkdir(parents=True)
    area = np.tile([[0.25e-4, 0.0, 0.0]], (4, 1))  # 1 cm^2 in total
    u = flow / 1e-4
    suffix = "" if iterations < max_iterations else "Mean"
    for i, pt in enumerate(plane_pt, start=1):
        _write_plane(planes / f"p{suffix}_x{i}.raw", np.full(4, pt - 0.5 * u**2), area)
        _write_plane(planes / f"U{suffix}_x{i}.raw", np.tile([[u, 0.0, 0.0]], (4, 1)), area)


@pytest.mark.parametrize("iterations", [800, 3000])
def test_read_side_recovers_region_drops(tmp_path, iterations):
    flow = 400 * CM3
    pt = [100.0, 60.0, 50.0]  # kinematic total pressure at x1, x2, x3
    _fake_case(tmp_path, flow, pt, iterations, sim.MAX_ITERATIONS)
    side = sim.read_side(tmp_path)
    assert side["flow_cm3_s"] == pytest.approx(400)
    assert side["converged"] == (iterations < sim.MAX_ITERATIONS)
    assert side["mass_imbalance"] == pytest.approx(0)
    drops = np.diff([REFERENCE_PRESSURE] + [AIR_DENSITY * p for p in pt] + [0.0]) * -1
    expected = dict(zip(REGIONS, drops / 400))
    assert side["region_resistance"] == pytest.approx(expected)
    assert sum(side["region_resistance"].values()) == pytest.approx(side["resistance"])


def test_record_uses_the_baseline_columns():
    region = dict.fromkeys(REGIONS, 0.05)
    side = {"flow_cm3_s": 750.0, "resistance": 0.2, "region_resistance": region,
            "iterations": 3000, "converged": False, "unsteadiness": 0.01, "cells": 1000}
    record = sim.to_record({"left": side, "right": side})
    assert set(evaluate(CaseParams())) <= set(record)
    assert record["total_flow_cm3_s"] == pytest.approx(1500)
    assert record["total_resistance"] == pytest.approx(REFERENCE_PRESSURE / 1500)


def test_select_cases_picks_extremes_and_nominal():
    table = build_table(sample_cases(20))
    picked = select_cases(table, ids=["case_0003"], extremes=True, nominal=True)
    assert picked.case_id.iloc[0] == "nominal"
    assert "case_0003" in set(picked.case_id)
    for name in PARAM_SPECS:
        assert table[name].min() in set(picked[name]) and table[name].max() in set(picked[name])
    with pytest.raises(ValueError):
        select_cases(table, ids=["case_9999"])


def test_gmsh_mesh_has_named_patches(tmp_path):
    pytest.importorskip("gmsh")
    path = tmp_path / "side.msh"
    cells = sim.mesh_side(airway(CaseParams())["left"], path, cells_across=3)
    assert cells > 1000
    text = path.read_text()
    for name in ("inlet", "outlet", "wall", "fluid"):
        assert f'"{name}"' in text


@pytest.mark.skipif(not sim.docker_available(), reason="Docker is not available")
def test_openfoam_run_end_to_end(tmp_path):
    """A short, coarse run: checks the pipeline, not the numbers."""
    pytest.importorskip("gmsh")
    iterations = 300
    sim.mesh_side(airway(CaseParams())["left"], tmp_path / "side.msh", cells_across=3)
    for rel, text in sim.case_files(processes=2, max_iterations=iterations).items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, newline="\n")
    sim.run_case(tmp_path)
    side = sim.read_side(tmp_path, max_iterations=iterations)
    assert side["flow_cm3_s"] > 0
    assert sum(side["region_resistance"].values()) == pytest.approx(side["resistance"])
    assert side["mass_imbalance"] < 0.01


def test_compare_to_baseline_is_one_for_the_baseline_itself():
    table = build_table(sample_cases(3))
    ratio = sim.compare_to_baseline(table)
    assert list(ratio.case_id) == list(table.case_id)
    assert np.allclose(ratio.drop(columns="case_id").to_numpy(), 1.0)
