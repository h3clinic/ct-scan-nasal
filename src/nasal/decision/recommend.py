"""Final stage: suggest a course of surgery for one case.

Every option is scored by virtual surgery through the evaluator: the trained
surrogate when data/surrogate.pt exists, else the uncalibrated formula. The
suggestion comes from plus_plus_rule, whose thresholds are placeholders pending
ENT review, so the output is a ranked what-if, not a clinical recommendation.

python -m nasal.decision.recommend --valve-angle-deg 9 --turbinate-scale 1.3 --septum-deviation 0.6
"""

import argparse
import json
from pathlib import Path

from ..params import PARAM_SPECS, REGIONS, SIDES, CaseParams
from ..resistance import evaluate
from .rules import PLACEHOLDER_THRESHOLDS, Evaluator, candidate_plans, plus_plus_rule, region_fractions, standard_rule
from .surgery import apply_plan

MODEL_NOTE = "Rules are unreviewed placeholders; idealized synthetic geometry."
REASONS = {
    "turbinate_reduction": "the turbinate region",
    "septoplasty": "the side narrowed by septal deviation",
    "valve_repair": "the nasal valve",
}


def default_evaluator(surrogate_path: Path = Path("data/surrogate.pt")) -> tuple[Evaluator, str]:
    from ..agents import load_surrogate

    model = load_surrogate(surrogate_path)
    if model is not None:
        return model.predict, "surrogate"
    return evaluate, "formula baseline (uncalibrated)"


def _label(plan: tuple[str, ...]) -> str:
    return "+".join(plan) or "none"


def _option(p: CaseParams, plan: tuple[str, ...], before: float, evaluator: Evaluator) -> dict:
    after = evaluator(apply_plan(p, plan))
    return {
        "plan": _label(plan),
        "total_resistance_after": round(after["total_resistance"], 4),
        "drop_pct": round(100 * (1 - after["total_resistance"] / before), 1),
        **{f"{s}_resistance_after": round(after[f"{s}_resistance"], 4) for s in SIDES},
    }


def recommend(p: CaseParams, evaluator: Evaluator | None = None, source: str | None = None) -> dict:
    if evaluator is None:
        evaluator, source = default_evaluator()
    p = p.validate()
    record = evaluator(p)
    before = record["total_resistance"]
    fractions = region_fractions(record)
    suggested, standard = plus_plus_rule(p, evaluator), standard_rule(p)
    options = sorted((_option(p, plan, before, evaluator) for plan in candidate_plans()),
                     key=lambda o: -o["drop_pct"])
    by_plan = {o["plan"]: o for o in options}
    worse = max(SIDES, key=lambda s: record[f"{s}_resistance"])
    return {
        "label": MODEL_NOTE,
        "evaluator": source or "custom",
        "case": {k: getattr(p, k) for k in PARAM_SPECS},
        "before": {
            "total_resistance": round(before, 4),
            "worse_side": worse,
            **{f"{s}_resistance": round(record[f"{s}_resistance"], 4) for s in SIDES},
            "region_share_pct_worse_side": {r: round(100 * fractions[r], 1) for r in REGIONS},
        },
        "dominant_region": max(fractions, key=fractions.get),
        "suggested": by_plan.get(_label(suggested), {"plan": "none", "drop_pct": 0.0}),
        "standard": by_plan[_label(standard)],
        "changed_from_standard": suggested != standard,
        "why": [f"{name}: targets {REASONS[name]}" for name in suggested]
               or ["No procedure lowered modeled resistance."],
        "options": options,
        "thresholds": {"status": "placeholder, pending ENT review", **PLACEHOLDER_THRESHOLDS},
        "not_modeled": ["inferior/middle/superior flow split", "wall shear and ENS-risk patterns",
                        "mucosal state", "patient symptoms"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Suggest a course of surgery for one case. " + MODEL_NOTE)
    for name, spec in PARAM_SPECS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(CaseParams(), name),
                        help=f"{spec.description} ({spec.low} to {spec.high} {spec.unit})")
    ap.add_argument("--surrogate", type=Path, default=Path("data/surrogate.pt"))
    args = ap.parse_args(argv)
    evaluator, source = default_evaluator(args.surrogate)
    p = CaseParams(**{k: getattr(args, k) for k in PARAM_SPECS})
    print(json.dumps(recommend(p, evaluator, source), indent=2))


if __name__ == "__main__":
    main()
