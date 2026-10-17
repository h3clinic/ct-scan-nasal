import numpy as np
import pytest

from nasal import docs
from nasal.cases import build_table, sample_cases
from nasal.geometry import airway, area_profile, side_mesh
from nasal.params import PARAM_SPECS, CaseParams
from nasal.resistance import REFERENCE_PRESSURE, evaluate


def test_out_of_range_is_rejected():
    bad = CaseParams(valve_angle_deg=40.0)
    assert bad.out_of_range() == ["valve_angle_deg"]
    with pytest.raises(ValueError):
        evaluate(bad)


def test_samples_stay_in_range_and_are_reproducible():
    a = sample_cases(50, seed=3)
    assert a == sample_cases(50, seed=3)
    assert all(not c.out_of_range() for c in a)


def test_mesh_is_watertight_and_matches_area_profile():
    geom = airway(CaseParams())["left"]
    mesh = side_mesh(geom)
    x, area, _ = area_profile(geom)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.trapezoid(area, x), rel=0.02)


def test_region_resistances_sum_to_side_total():
    r = evaluate(CaseParams())
    for side in ("left", "right"):
        parts = sum(v for k, v in r.items() if k.startswith(f"{side}_") and k.endswith("_resistance") and k != f"{side}_resistance")
        assert parts == pytest.approx(r[f"{side}_resistance"])
    assert r["total_flow_cm3_s"] == pytest.approx(REFERENCE_PRESSURE / r["total_resistance"])


def test_symmetric_case_has_equal_sides():
    r = evaluate(CaseParams(septum_deviation=0.0))
    assert r["left_resistance"] == pytest.approx(r["right_resistance"])


def test_narrower_valve_raises_valve_resistance():
    wide = evaluate(CaseParams(valve_angle_deg=18))
    narrow = evaluate(CaseParams(valve_angle_deg=10))
    assert narrow["left_valve_resistance"] > wide["left_valve_resistance"]


def test_larger_turbinate_raises_turbinate_resistance():
    small = evaluate(CaseParams(turbinate_scale=0.7))
    large = evaluate(CaseParams(turbinate_scale=1.5))
    assert large["left_turbinate_resistance"] > small["left_turbinate_resistance"]


def test_deviation_narrows_the_side_it_bends_into():
    r = evaluate(CaseParams(septum_deviation=0.8))
    assert r["left_resistance"] > r["right_resistance"]
    r = evaluate(CaseParams(septum_deviation=-0.8))
    assert r["right_resistance"] > r["left_resistance"]


def test_table_has_one_row_per_case_and_every_parameter():
    table = build_table(sample_cases(5))
    assert len(table) == 5
    assert set(PARAM_SPECS) <= set(table.columns)


def test_readme_generated_sections_are_current():
    text = docs.README.read_text(encoding="utf-8")
    assert docs.render(text) == text, "README is stale: run `python -m nasal.docs`"
