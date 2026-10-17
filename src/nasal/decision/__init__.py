"""Surgical decision layer: virtual surgery, standard vs Workflow++ rules, reclassification.

Every number here comes from a resistance evaluator (formula baseline today, the
surrogate or OpenFOAM later). Thresholds are placeholders pending ENT review.
"""

from .compare import reclassify, summarize
from .rules import PLACEHOLDER_THRESHOLDS, plus_plus_rule, standard_rule
from .surgery import PROCEDURES, apply_plan

__all__ = [
    "PLACEHOLDER_THRESHOLDS",
    "PROCEDURES",
    "apply_plan",
    "plus_plus_rule",
    "reclassify",
    "standard_rule",
    "summarize",
]
