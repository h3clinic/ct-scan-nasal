"""Virtual surgery: each procedure is a change to the case parameters.

Procedure sizes are modeling choices, not clinical recommendations. Results are
clamped to PARAM_SPECS so a virtual patient never leaves the generated range.
"""

from dataclasses import replace

from ..params import PARAM_SPECS, CaseParams


def _clamp(name: str, value: float) -> float:
    spec = PARAM_SPECS[name]
    return min(max(value, spec.low), spec.high)


def turbinate_reduction(p: CaseParams, fraction: float = 0.3) -> CaseParams:
    """Inferior turbinate reduction: shrink turbinate size by `fraction`."""
    return replace(p, turbinate_scale=_clamp("turbinate_scale", p.turbinate_scale * (1 - fraction)))


def septoplasty(p: CaseParams) -> CaseParams:
    """Straighten the septum."""
    return replace(p, septum_deviation=0.0)


def valve_repair(p: CaseParams, widen_deg: float = 4.0) -> CaseParams:
    """Nasal valve procedure (e.g. spreader grafts): widen the valve angle."""
    return replace(p, valve_angle_deg=_clamp("valve_angle_deg", p.valve_angle_deg + widen_deg))


PROCEDURES = {
    "turbinate_reduction": turbinate_reduction,
    "septoplasty": septoplasty,
    "valve_repair": valve_repair,
}


def apply_plan(p: CaseParams, plan: tuple[str, ...]) -> CaseParams:
    for name in plan:
        p = PROCEDURES[name](p)
    return p
