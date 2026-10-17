"""Reclassification table: standard vs Workflow++ plan per case, scored by virtual surgery."""

import pandas as pd

from ..params import PARAM_SPECS, CaseParams
from ..resistance import evaluate
from .rules import Evaluator, plus_plus_rule, region_fractions, standard_rule
from .surgery import apply_plan


def _score(p: CaseParams, plan: tuple[str, ...], evaluator: Evaluator) -> float:
    return evaluator(apply_plan(p, plan))["total_resistance"]


def reclassify(cases: pd.DataFrame, evaluator: Evaluator = evaluate) -> pd.DataFrame:
    rows = []
    for row in cases.to_dict("records"):
        p = CaseParams(**{k: row[k] for k in PARAM_SPECS})
        record = evaluator(p)
        fractions = region_fractions(record)
        standard, plus = standard_rule(p), plus_plus_rule(p, evaluator)
        before = record["total_resistance"]
        rows.append({
            "case_id": row.get("case_id"),
            **{k: row[k] for k in PARAM_SPECS},
            "dominant_region": max(fractions, key=fractions.get),
            "resistance_before": before,
            "standard_plan": "+".join(standard) or "none",
            "plus_plus_plan": "+".join(plus) or "none",
            "changed": standard != plus,
            "standard_after": _score(p, standard, evaluator),
            "plus_plus_after": _score(p, plus, evaluator),
        })
    table = pd.DataFrame(rows)
    table["standard_drop_pct"] = 100 * (1 - table.standard_after / table.resistance_before)
    table["plus_plus_drop_pct"] = 100 * (1 - table.plus_plus_after / table.resistance_before)
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per dominant region: how often the plan changed and mean resistance drop."""
    return table.groupby("dominant_region").agg(
        cases=("changed", "size"),
        changed_pct=("changed", lambda c: 100 * c.mean()),
        standard_drop_pct=("standard_drop_pct", "mean"),
        plus_plus_drop_pct=("plus_plus_drop_pct", "mean"),
    )
