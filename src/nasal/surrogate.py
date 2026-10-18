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
from dataclasses import asdict, dataclass
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


# ---------------------------------------------------------------- real patients

PATIENT_FEATURES = ("volume_cm3", "length_mm", "inlet_area_mm2", "min_area_mm2", "mean_area_mm2", "min_area_pos")
PATIENT_RIDGE = 1.0


def _patient_design(geometry_rows: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(geometry_rows, dtype=float), 1e-3, None))


def patient_side_rows(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Measured geometry (n_sides, 6) and log region resistances (n_sides, 4) for every side."""
    feats, targets = [], []
    for _, row in table.iterrows():
        for side in SIDES:
            feats.append([row[f"{side}_{f}"] for f in PATIENT_FEATURES])
            targets.append([np.log(row[f"{side}_{r}_resistance"]) for r in REGIONS])
    return np.array(feats, dtype=float), np.array(targets, dtype=float)


@dataclass
class PatientSurrogate:
    """Power law per region: log(region resistance) is linear in the log of one
    side's measured geometry (PATIENT_FEATURES), fitted by ridge regression on
    standardized inputs. With about two hundred sides this generalized better
    than the MLP used for the idealized cases (README, Real patients)."""

    weights: list  # (1 + n_features) x n_regions, on standardized log features
    x_mean: list[float]  # mean of the log features
    x_std: list[float]

    @classmethod
    def fit(cls, table: pd.DataFrame, ridge: float = PATIENT_RIDGE) -> "PatientSurrogate":
        x, y = patient_side_rows(table)
        z = _patient_design(x)
        mean, std = z.mean(axis=0), z.std(axis=0) + 1e-9
        a = np.hstack([np.ones((len(z), 1)), (z - mean) / std])
        penalty = ridge * np.eye(a.shape[1])
        penalty[0, 0] = 0.0  # the intercept is not shrunk
        w = np.linalg.solve(a.T @ a + penalty, a.T @ y)
        return cls(w.tolist(), mean.tolist(), std.tolist())

    def zscores(self, geometry_rows: np.ndarray) -> np.ndarray:
        """How far each side's log geometry is from the training mean, in std devs."""
        return (_patient_design(np.atleast_2d(geometry_rows)) - np.array(self.x_mean)) / np.array(self.x_std)

    def region_resistance(self, geometry_rows: np.ndarray) -> np.ndarray:
        z = self.zscores(geometry_rows)
        a = np.hstack([np.ones((len(z), 1)), z])
        return np.exp(a @ np.array(self.weights))

    def side_features(self, geometry: dict) -> np.ndarray:
        return np.array([[geometry[f"{s}_{f}"] for f in PATIENT_FEATURES] for s in SIDES], dtype=float)

    def predict(self, geometry: dict, pressure: float = REFERENCE_PRESSURE) -> dict:
        """Record with `evaluate`'s columns from one patient's measured geometry
        (keys like left_min_area_mm2, as written by `nasal.anatomy.measure`)."""
        regions = self.region_resistance(self.side_features(geometry))
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
        Path(path).write_text(json.dumps({"features": list(PATIENT_FEATURES), **asdict(self)}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "PatientSurrogate":
        data = json.loads(Path(path).read_text())
        if data.pop("features") != list(PATIENT_FEATURES):
            raise ValueError(f"{path} was trained on different features")
        return cls(**data)


def patient_report(table: pd.DataFrame, folds: int = 5, seed: int = 0) -> dict:
    """Cross-validated error of the patient surrogate, next to predicting the
    training-fold average (geometric mean) for every patient."""
    table = table.reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(table))
    preds, means = {}, {}
    for k in range(folds):
        test = order[k::folds]
        train = table.iloc[np.setdiff1d(order, test)]
        model = PatientSurrogate.fit(train)
        avg = {c: float(np.exp(np.log(train[c]).mean())) for c in train.columns if c.endswith("_resistance")}
        for i in test:
            preds[i] = model.predict(table.iloc[i].to_dict())
            means[i] = avg
    cv = pd.DataFrame([preds[i] for i in range(len(table))])
    avg = pd.DataFrame([means[i] for i in range(len(table))])
    model = PatientSurrogate.fit(table)
    start = time.perf_counter()
    for _ in range(200):
        model.predict(table.iloc[0].to_dict())
    return {"patients": len(table), "folds": folds, "features": list(PATIENT_FEATURES),
            "model": "power law per region (ridge on log geometry)",
            "surrogate": errors(table, cv), "average_patient": errors(table, avg),
            "predict_seconds_per_case": (time.perf_counter() - start) / 200}
