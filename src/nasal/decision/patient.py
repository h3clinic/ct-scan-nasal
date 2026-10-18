"""Virtual surgery and decision rules on real patient airways (NasalSeg voxels).

Each procedure adds or moves one voxel layer
(about 0.6 mm) of airway where it acts; the sizes are modeling choices, not
surgical prescriptions, and the rule thresholds are placeholders pending ENT review.

- valve_repair: widen both valve regions (20-35% of each cavity's length) sideways and vertically.
- turbinate_reduction: push both cavities' lateral wall outward in the turbinate region (35-85%).
- septoplasty: shift the septum one voxel toward the wider side (by mean cross-section)
  in the valve and turbinate regions: the wider cavity loses its medial layer, the narrower gains one.

Growth never touches the other cavity (the septum keeps at least one voxel) and never
crosses the nostril or outlet cuts, so the operated airway meshes like the original.
"""

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import anatomy
from ..params import SIDES
from .rules import PLACEHOLDER_THRESHOLDS

PATIENT_PROCEDURES = ("turbinate_reduction", "septoplasty", "valve_repair")
# PLACEHOLDER value, not a clinical threshold. Pending ENT review.
PATIENT_THRESHOLDS = {"visible_asymmetry": 1.3}  # mean cross-section ratio above which the septum "looks" deviated
LATERAL = {"left": +1, "right": -1}  # LPS: +x is the patient's left
SIX = ndimage.generate_binary_structure(3, 1)

TagEvaluator = Callable[[np.ndarray], dict]


def _shift(mask: np.ndarray, axis: int, step: int) -> np.ndarray:
    out = np.zeros_like(mask)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if step > 0:
        src[axis], dst[axis] = slice(None, -step), slice(step, None)
    else:
        src[axis], dst[axis] = slice(-step, None), slice(None, step)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def _grow(tags: np.ndarray, cuts: dict, side: str, regions: tuple[str, ...],
          directions: list[tuple[int, int]]) -> np.ndarray:
    lab, other = anatomy.LABELS[side], anatomy.LABELS["right" if side == "left" else "left"]
    names = anatomy.region_names(tags)
    source = (tags == lab) & np.isin(names, [f"{side}_{r}" for r in regions])
    new = np.zeros_like(source)
    for axis, step in directions:
        new |= _shift(source, axis, step)
    new &= tags == 0
    new &= ~ndimage.binary_dilation(tags == other, SIX)  # the septum keeps at least one voxel
    new[:, : cuts[anatomy.INLETS[side]] + 1] = False  # stay behind the nostril cut
    new[:, :, : cuts["outlet"] + 1] = False
    out = tags.copy()
    out[new] = lab
    return out


def _wider_side(tags: np.ndarray, spacing: np.ndarray) -> str:
    m = anatomy.measure(tags, spacing)
    return max(SIDES, key=lambda s: m[f"{s}_mean_area_mm2"])


def valve_repair(tags, cuts, spacing):
    for side in SIDES:
        tags = _grow(tags, cuts, side, ("valve",), [(0, 1), (0, -1), (2, 1), (2, -1)])
    return tags


def turbinate_reduction(tags, cuts, spacing):
    for side in SIDES:
        tags = _grow(tags, cuts, side, ("turbinate",), [(0, LATERAL[side])])
    return tags


def septoplasty(tags, cuts, spacing):
    wide = _wider_side(tags, spacing)
    narrow = "right" if wide == "left" else "left"
    lab = anatomy.LABELS[wide]
    names = anatomy.region_names(tags)
    w = tags == lab
    in_regions = np.isin(names, [f"{wide}_valve", f"{wide}_turbinate"])
    medial = -LATERAL[wide]
    # remove the wide side's medial layer where the cavity is at least two voxels wide
    boundary = w & ~_shift(w, 0, -medial) & _shift(w, 0, medial) & in_regions
    out = tags.copy()
    out[boundary] = 0
    return _grow(out, cuts, narrow, ("valve", "turbinate"), [(0, -LATERAL[narrow])])


PROCEDURE_FUNCS = {"turbinate_reduction": turbinate_reduction, "septoplasty": septoplasty, "valve_repair": valve_repair}


def apply_patient_plan(tags: np.ndarray, cuts: dict, spacing: np.ndarray, plan: tuple[str, ...]) -> np.ndarray:
    for name in PATIENT_PROCEDURES:  # fixed order, so a plan's result doesn't depend on how it was listed
        if name in plan:
            tags = PROCEDURE_FUNCS[name](tags, cuts, spacing)
            # removing the septum-side layer can cut off thin pockets of airway; a mesh with an
            # unconnected region cannot be solved, so keep the part connected to the rest
            tags[~anatomy._largest_component(tags > 0)] = 0
    return tags


def candidate_plans() -> list[tuple[str, ...]]:
    return [c for k in range(1, len(PATIENT_PROCEDURES) + 1) for c in combinations(PATIENT_PROCEDURES, k)]


def standard_rule_patient(geometry: dict) -> tuple[str, ...]:
    """Visual default: turbinate reduction, plus septoplasty if the sides look unequal."""
    areas = [geometry[f"{s}_mean_area_mm2"] for s in SIDES]
    plan = ["turbinate_reduction"]
    if max(areas) / min(areas) > PATIENT_THRESHOLDS["visible_asymmetry"]:
        plan.append("septoplasty")
    return tuple(plan)


def plus_plus_rule_patient(tags, cuts, spacing, evaluator: TagEvaluator) -> tuple[tuple[str, ...], dict]:
    """Smallest plan whose predicted drop is near the best plan's; also returns every plan's prediction."""
    before = evaluator(tags)["total_resistance"]
    predicted = {plan: evaluator(apply_patient_plan(tags, cuts, spacing, plan))["total_resistance"]
                 for plan in candidate_plans()}
    gains = {plan: before - r for plan, r in predicted.items()}
    best = max(gains.values())
    if best <= 0:
        return (), {(): before, **predicted}
    good = [p for p, g in gains.items() if g >= PLACEHOLDER_THRESHOLDS["near_best_gain"] * best]
    return min(good, key=lambda p: (len(p), -gains[p])), {(): before, **predicted}


def surrogate_evaluator(surrogate, spacing: np.ndarray) -> TagEvaluator:
    """Resistance of an airway tag volume from its measured geometry, via the patient surrogate."""
    return lambda tags: surrogate.predict(anatomy.measure(tags, spacing))


def label(plan: tuple[str, ...]) -> str:
    return "+".join(plan) or "none"


def compare_surgery(sim: pd.DataFrame) -> pd.DataFrame:
    """Per patient: OpenFOAM drop of each rule's plan, the plan OpenFOAM finds best, and whether
    Workflow++ (chosen with the surrogate) lands in OpenFOAM's near-best set.

    `sim` has one row per patient and plan with columns patient, plan, total_resistance
    (OpenFOAM), surrogate_total, standard_plan and plus_plus_plan."""
    rows = []
    for pid, g in sim.groupby("patient"):
        cfd = dict(zip(g.plan, g.total_resistance))
        sur = dict(zip(g.plan, g.surrogate_total))
        before = cfd.get("none")
        if before is None:
            continue
        drops = {p: 100 * (1 - r / before) for p, r in cfd.items() if p != "none"}
        best_plan = max(drops, key=drops.get)
        near = {p for p, d in drops.items() if d >= PLACEHOLDER_THRESHOLDS["near_best_gain"] * drops[best_plan]}
        std, pp = g.standard_plan.iloc[0], g.plus_plus_plan.iloc[0]
        rows.append({
            "patient": pid, "resistance_before": before,
            "openfoam_best_plan": best_plan, "openfoam_best_drop_pct": drops[best_plan],
            "standard_plan": std, "standard_drop_pct": drops.get(std, np.nan),
            "plus_plus_plan": pp, "plus_plus_drop_pct": drops.get(pp, np.nan),
            "plus_plus_near_best": pp in near, "standard_near_best": std in near,
            "surrogate_drop_error_pct": float(np.mean([abs(100 * (1 - sur[p] / sur["none"]) - drops[p])
                                                       for p in drops if p in sur])),
        })
    return pd.DataFrame(rows)
