from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import airway, side_mesh
from .params import PARAM_SPECS, CaseParams
from .resistance import evaluate


def sample_cases(n: int, seed: int = 0) -> list[CaseParams]:
    rng = np.random.default_rng(seed)
    columns = {
        name: rng.uniform(spec.low, spec.high, n) for name, spec in PARAM_SPECS.items()
    }
    return [CaseParams(**{k: float(v[i]) for k, v in columns.items()}) for i in range(n)]


def build_table(cases: list[CaseParams]) -> pd.DataFrame:
    rows = []
    for i, params in enumerate(cases):
        rows.append({"case_id": f"case_{i:04d}", **vars(params), **evaluate(params)})
    return pd.DataFrame(rows)


def write_meshes(table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in table.itertuples(index=False):
        params = CaseParams(**{k: getattr(row, k) for k in PARAM_SPECS})
        for side, geom in airway(params).items():
            side_mesh(geom).export(out_dir / f"{row.case_id}_{side}.stl")
