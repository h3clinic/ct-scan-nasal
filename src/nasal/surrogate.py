"""Fast stand-in for the OpenFOAM runs: a small PyTorch network per nasal side.

The two sides are mirror images, so one network serves both. A side is
described by the valve angle, the turbinate scale and how far the septum bends
toward it (`septum_deviation` for the left side, its negative for the right),
which doubles the training data and makes a symmetric case predict equal
sides. The network predicts log region resistances, so every prediction is
positive and the regions sum to the side resistance exactly as in
`nasal.resistance.evaluate`, whose columns the predictions use.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .params import PARAM_SPECS, REGIONS, SIDES, CaseParams
from .resistance import REFERENCE_PRESSURE, evaluate

FEATURES = ("valve_angle_deg", "turbinate_scale", "deviation_toward")
HIDDEN = 64
EPOCHS = 4000
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5


def side_rows(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (n_sides, 3), log region resistances (n_sides, 4) and the
    row index each side came from, for every side of every case."""
    feats, targets, rows = [], [], []
    for i, row in enumerate(table.itertuples(index=False)):
        for side in SIDES:
            toward = row.septum_deviation if side == "left" else -row.septum_deviation
            feats.append([row.valve_angle_deg, row.turbinate_scale, toward])
            targets.append([np.log(getattr(row, f"{side}_{r}_resistance")) for r in REGIONS])
            rows.append(i)
    return np.array(feats), np.array(targets), np.array(rows)


def _scale(x: np.ndarray) -> np.ndarray:
    """Map features onto [-1, 1] using the PARAM_SPECS ranges."""
    specs = [PARAM_SPECS["valve_angle_deg"], PARAM_SPECS["turbinate_scale"], PARAM_SPECS["septum_deviation"]]
    lo = np.array([s.low for s in specs])
    hi = np.array([s.high for s in specs])
    return 2 * (x - lo) / (hi - lo) - 1


@dataclass
class Surrogate:
    state: dict
    target_mean: list[float]
    target_std: list[float]

    @staticmethod
    def _network():
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, len(REGIONS)),
        )

    @classmethod
    def fit(cls, table: pd.DataFrame, epochs: int = EPOCHS, seed: int = 0) -> "Surrogate":
        import torch

        x, y, _ = side_rows(table)
        mean, std = y.mean(axis=0), y.std(axis=0) + 1e-9
        torch.manual_seed(seed)
        net = cls._network()
        xt = torch.tensor(_scale(x), dtype=torch.float32)
        yt = torch.tensor((y - mean) / std, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.mean((net(xt) - yt) ** 2)
            loss.backward()
            opt.step()
        return cls({k: v.detach().clone() for k, v in net.state_dict().items()}, mean.tolist(), std.tolist())

    def _net(self):
        if not hasattr(self, "_cached"):
            net = self._network()
            net.load_state_dict(self.state)
            net.eval()
            self._cached = net
        return self._cached

    def region_resistance(self, features: np.ndarray) -> np.ndarray:
        """Region resistances (Pa*s/cm^3), shape (n, 4), for side features (n, 3)."""
        import torch

        with torch.no_grad():
            z = self._net()(torch.tensor(_scale(np.atleast_2d(features)), dtype=torch.float32)).numpy()
        return np.exp(z * np.array(self.target_std) + np.array(self.target_mean))

    def predict(self, params: CaseParams, pressure: float = REFERENCE_PRESSURE) -> dict:
        """Flat record with the same columns as `nasal.resistance.evaluate`."""
        params.validate()
        d = params.septum_deviation
        regions = self.region_resistance(np.array([
            [params.valve_angle_deg, params.turbinate_scale, d],
            [params.valve_angle_deg, params.turbinate_scale, -d],
        ]))
        record, total_flow = {}, 0.0
        for side, reg in zip(SIDES, regions):
            r = float(reg.sum())
            total_flow += pressure / r
            record[f"{side}_flow_cm3_s"] = pressure / r
            record[f"{side}_resistance"] = r
            for name, value in zip(REGIONS, reg):
                record[f"{side}_{name}_resistance"] = float(value)
        return {"total_resistance": pressure / total_flow, "total_flow_cm3_s": total_flow, **record}

    def save(self, path: Path) -> None:
        import torch

        torch.save({"state": self.state, "target_mean": self.target_mean, "target_std": self.target_std}, path)

    @classmethod
    def load(cls, path: Path) -> "Surrogate":
        import torch

        data = torch.load(path, weights_only=True)
        return cls(data["state"], data["target_mean"], data["target_std"])


def _params(row) -> CaseParams:
    return CaseParams(**{k: getattr(row, k) for k in PARAM_SPECS})


def cross_validate(table: pd.DataFrame, folds: int = 5, epochs: int = EPOCHS, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold predictions for every case: each case is predicted by a
    network that never saw it. Folds split by case, so both sides of a case
    are held out together."""
    order = np.random.default_rng(seed).permutation(len(table))
    rows = []
    for k in range(folds):
        test = order[k::folds]
        model = Surrogate.fit(table.iloc[np.setdiff1d(order, test)], epochs=epochs, seed=seed)
        for i in test:
            row = table.iloc[i]
            rows.append({"case_id": row.case_id, **model.predict(_params(row))})
    return pd.DataFrame(rows).set_index("case_id").loc[table.case_id].reset_index()


def errors(truth: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Mean and max absolute percentage error for total, side and region resistance."""
    out = {}
    groups = {
        "total": ["total_resistance"],
        "side": [f"{s}_resistance" for s in SIDES],
        "region": [f"{s}_{r}_resistance" for s in SIDES for r in REGIONS],
    }
    for name, cols in groups.items():
        e = np.abs(predicted[cols].to_numpy() / truth[cols].to_numpy() - 1) * 100
        out[f"{name}_mean_pct"] = float(e.mean())
        out[f"{name}_max_pct"] = float(e.max())
    return out


def report(table: pd.DataFrame, folds: int = 5, epochs: int = EPOCHS) -> dict:
    """Cross-validated surrogate error, the formula baseline's error on the
    same cases, and the speedup over the simulation."""
    table = table.reset_index(drop=True)
    cv = cross_validate(table, folds, epochs)
    baseline = pd.DataFrame([evaluate(_params(r)) for r in table.itertuples(index=False)])
    model = Surrogate.fit(table, epochs=epochs)
    start = time.perf_counter()
    n = 200
    for r in table.sample(n, replace=True, random_state=0).itertuples(index=False):
        model.predict(_params(r))
    predict_seconds = (time.perf_counter() - start) / n
    result = {
        "cases": len(table),
        "folds": folds,
        "surrogate": errors(table, cv),
        "formula_baseline": errors(table, baseline),
        "predict_seconds_per_case": predict_seconds,
    }
    seconds = [c for c in ("left_seconds", "right_seconds") if c in table]
    if seconds and table[seconds].notna().all(axis=1).any():
        timed = table[seconds].dropna().sum(axis=1)
        result["simulation_seconds_per_case_median"] = float(timed.median())
        result["speedup"] = float(timed.median() / predict_seconds)
    return result


def load_simulated(paths: list[Path]) -> pd.DataFrame:
    """Concatenate simulated.csv files, keeping one row per case_id."""
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates("case_id", keep="last").reset_index(drop=True)


def write_report(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
