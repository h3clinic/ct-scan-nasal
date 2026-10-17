"""Decision rules. Both return a plan: a tuple of procedure names in PROCEDURES order.

standard_rule mimics the visual default: look at what is prominent, then
turbinate reduction with or without septoplasty. plus_plus_rule reads the
regional resistance map and picks the smallest plan that captures most of the
achievable gain. All thresholds are PLACEHOLDERS until an ENT reviews them.
"""

from itertools import combinations
from typing import Callable

from ..params import REGIONS, SIDES, CaseParams
from ..resistance import evaluate
from .surgery import PROCEDURES, apply_plan

Evaluator = Callable[[CaseParams], dict]

# PLACEHOLDER values, not clinical thresholds. Pending ENT review.
PLACEHOLDER_THRESHOLDS = {
    "visible_deviation": 0.3,  # |septum_deviation| above this looks deviated
    "near_best_gain": 0.9,  # ++ accepts a plan reaching this share of the best reduction
}


def standard_rule(p: CaseParams) -> tuple[str, ...]:
    """Visual default: turbinate reduction, plus septoplasty if the septum looks bent."""
    plan = ["turbinate_reduction"]
    if abs(p.septum_deviation) > PLACEHOLDER_THRESHOLDS["visible_deviation"]:
        plan.append("septoplasty")
    return tuple(plan)


def region_fractions(record: dict) -> dict[str, float]:
    """Share of each side's resistance per region, for the more obstructed side."""
    side = max(SIDES, key=lambda s: record[f"{s}_resistance"])
    total = record[f"{side}_resistance"]
    return {r: record[f"{side}_{r}_resistance"] / total for r in REGIONS}


def candidate_plans() -> list[tuple[str, ...]]:
    names = tuple(PROCEDURES)
    return [c for k in range(1, len(names) + 1) for c in combinations(names, k)]


def plus_plus_rule(p: CaseParams, evaluator: Evaluator = evaluate) -> tuple[str, ...]:
    """Smallest plan whose virtual-surgery resistance drop is near the best plan's."""
    before = evaluator(p)["total_resistance"]
    gains = {plan: before - evaluator(apply_plan(p, plan))["total_resistance"] for plan in candidate_plans()}
    best = max(gains.values())
    if best <= 0:
        return ()
    good = [plan for plan, g in gains.items() if g >= PLACEHOLDER_THRESHOLDS["near_best_gain"] * best]
    return min(good, key=lambda plan: (len(plan), -gains[plan]))
