"""Demo helper tests (no Gradio, server or dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import fake_labels  # noqa: E402

from nasal import anatomy, demo  # noqa: E402
from nasal.params import REGIONS, SIDES  # noqa: E402


def test_every_airway_voxel_gets_a_region():
    tags, _ = anatomy.airway(fake_labels())
    names = anatomy.region_names(tags)
    assert (names[tags > 0] != "").all() and (names[tags == 0] == "").all()
    expected = {f"{s}_{r}" for s in SIDES for r in REGIONS}
    assert set(names[tags > 0]) == expected
    left = tags == anatomy.LABELS["left"]
    ys = np.argwhere(left & (names == "left_vestibule"))[:, 1]
    assert ys.max() < np.argwhere(left & (names == "left_turbinate"))[:, 1].min()


def test_with_widening_changes_only_the_named_side():
    row = {"left_min_area_mm2": 100.0, "left_mean_area_mm2": 150.0, "right_min_area_mm2": 80.0, "right_mean_area_mm2": 90.0}
    out = demo.with_widening(row, {"left": 30, "right": 50})
    assert out["left_min_area_mm2"] == pytest.approx(130) and out["left_mean_area_mm2"] == 150
    assert out["right_min_area_mm2"] == pytest.approx(120) and out["right_mean_area_mm2"] == pytest.approx(120)


def test_comparison_table_reports_errors():
    rec = {f"{s}_{r}_resistance": 0.1 for s in SIDES for r in REGIONS}
    rec.update({f"{s}_resistance": 0.4 for s in SIDES}, total_resistance=0.2)
    pred = {k: v * 1.1 for k, v in rec.items()}
    table = demo.comparison_table(rec, pred)
    assert len(table) == 2 * (len(REGIONS) + 1) + 1
    assert np.allclose(table["error %"], 10.0)


def test_describe_step_formats_agent_events():
    assert "set_patient" in demo.describe_step({"agent": "case", "kind": "model", "seconds": 0.5,
                                                "tool_calls": [{"name": "set_patient", "arguments": "{}"}]})
    assert "passed" in demo.describe_step({"agent": "explainer", "kind": "number_check", "unsupported": [], "missing": []})
    assert demo.describe_step({"agent": "case", "kind": "tool", "result": {"ok": True}}) is None


def test_resistance_figure_colors_vertices_by_region():
    pytest.importorskip("plotly")
    surface = {"verts": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), "faces": np.array([[0, 1, 2]]),
               "region": np.array(["left_valve", "left_valve", "right_choana"], dtype=object)}
    fig = demo.resistance_figure(surface, {"left_valve": 0.2, "right_choana": 0.05}, "test")
    assert list(fig.data[0].intensity) == [0.2, 0.2, 0.05]
