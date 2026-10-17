import numpy as np
import pytest

from nasal.cases import build_table, sample_cases
from nasal.params import REGIONS, CaseParams
from nasal.resistance import evaluate

torch = pytest.importorskip("torch")
from nasal import surrogate as sg  # noqa: E402


@pytest.fixture(scope="module")
def table():
    # The formula baseline stands in for simulation results: same columns.
    return build_table(sample_cases(60, seed=1))


@pytest.fixture(scope="module")
def model(table):
    return sg.Surrogate.fit(table, epochs=1500)


def test_side_rows_mirror_the_septum(table):
    x, y, rows = sg.side_rows(table.head(1))
    assert x.shape == (2, 3) and y.shape == (2, len(REGIONS))
    assert x[0, 2] == -x[1, 2]
    assert list(rows) == [0, 0]


def test_prediction_has_baseline_columns_and_sums(model):
    record = model.predict(CaseParams())
    assert set(record) == set(evaluate(CaseParams()))
    for side in ("left", "right"):
        parts = sum(record[f"{side}_{r}_resistance"] for r in REGIONS)
        assert parts == pytest.approx(record[f"{side}_resistance"])


def test_symmetric_case_predicts_equal_sides(model):
    record = model.predict(CaseParams(septum_deviation=0.0))
    assert record["left_resistance"] == pytest.approx(record["right_resistance"])


def test_fits_a_smooth_target_closely(table, model):
    predicted = [model.predict(sg._params(r)) for r in table.itertuples(index=False)]
    import pandas as pd

    err = sg.errors(table, pd.DataFrame(predicted))
    assert err["side_mean_pct"] < 5


def test_save_and_load_round_trip(tmp_path, model):
    model.save(tmp_path / "s.pt")
    again = sg.Surrogate.load(tmp_path / "s.pt")
    a, b = model.predict(CaseParams(10, 1.2, 0.3)), again.predict(CaseParams(10, 1.2, 0.3))
    assert a == pytest.approx(b)


def test_report_holds_out_every_case(table):
    result = sg.report(table.head(20), folds=4, epochs=300)
    assert result["cases"] == 20
    assert {"surrogate", "formula_baseline", "predict_seconds_per_case"} <= set(result)
    assert result["formula_baseline"]["side_mean_pct"] == pytest.approx(0, abs=1e-9)
    assert np.isfinite(result["surrogate"]["total_mean_pct"])
