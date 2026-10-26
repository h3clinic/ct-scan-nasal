"""Studio view helper tests (no Gradio, server or dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_anatomy import fake_labels  # noqa: E402

from nasal import anatomy, view  # noqa: E402
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


def test_describe_step_formats_agent_events():
    assert "set_patient" in view.describe_step({"agent": "case", "kind": "model", "seconds": 0.5,
                                                "tool_calls": [{"name": "set_patient", "arguments": "{}"}]})
    assert "passed" in view.describe_step({"agent": "explainer", "kind": "number_check", "unsupported": [], "missing": []})
    assert view.describe_step({"agent": "case", "kind": "tool", "result": {"ok": True}}) is None


def test_resistance_figure_colors_vertices_by_region():
    pytest.importorskip("plotly")
    surface = {"verts": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), "faces": np.array([[0, 1, 2]]),
               "region": np.array(["left_valve", "left_valve", "right_choana"], dtype=object)}
    fig = view.resistance_figure(surface, {"left_valve": 0.2, "right_choana": 0.05}, "test")
    assert list(fig.data[0].intensity) == [0.2, 0.2, 0.05]


def test_as_markdown_fixes_units_and_asterisks():
    assert view.as_markdown("drops from 0.144 Pa*s/cm^3 to 0.13 Pa*s/cm^3") == "drops from 0.144 Pa·s/cm³ to 0.13 Pa·s/cm³"
    assert view.as_markdown("a*b model_scope") == r"a\*b model\_scope"
