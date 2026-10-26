"""Virtual surgery and decision rules on real patient airways (NasalSeg voxels).

Each procedure adds or moves one voxel layer
(about 0.6 mm) of airway where it acts; the sizes are modeling choices, not
surgical prescriptions, and the rule thresholds are placeholders pending ENT review.

- valve_repair: widen both valve regions (20-35% of each cavity's length) sideways and vertically.
- turbinate_reduction: push both cavities' lateral wall outward in the turbinate region (35-85%).
- septoplasty: straighten local deviations of the septum in the valve and turbinate regions, back toward a
  plane fitted to the septal midline; the septum keeps its thickness, so one cavity gains what the other
  gives up, and a straight septum is left alone.

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


def valve_repair(tags, cuts, spacing):
    for side in SIDES:
        tags = _grow(tags, cuts, side, ("valve",), [(0, 1), (0, -1), (2, 1), (2, -1)])
    return tags


def turbinate_reduction(tags, cuts, spacing):
    for side in SIDES:
        tags = _grow(tags, cuts, side, ("turbinate",), [(0, LATERAL[side])])
    return tags


MAX_SEPTUM_MM = 8.0  # wider gaps between the cavities are not the septum (a modeling choice, not anatomy data)
SEPTUM_THICKER_VOX = 3  # ... nor a septum more than this many voxels thicker than its median
BIWEIGHT_MIN_VOX = 2.0  # the septal plane ignores rows further off than this (or 4.685 robust spreads)


def septum_rows(tags: np.ndarray, spacing: np.ndarray) -> dict:
    """The septum in every coronal row (fixed y, z) of the valve and turbinate regions where both cavities are
    present and at most MAX_SEPTUM_MM apart: the right cavity's medial edge (largest x), the left cavity's
    medial edge (smallest x), the septum's midline between them and its deviation from a plane fitted to all
    midlines (in voxels, + toward the patient's left). Arrays are aligned; `j`, `k` index the rows."""
    names = anatomy.region_names(tags)
    right = (tags == anatomy.LABELS["right"]) & np.isin(names, ["right_valve", "right_turbinate"])
    left = (tags == anatomy.LABELS["left"]) & np.isin(names, ["left_valve", "left_turbinate"])
    both = right.any(axis=0) & left.any(axis=0)
    xs = np.arange(tags.shape[0])[:, None, None]
    r_edge = np.where(right, xs, -1).max(axis=0)
    l_edge = np.where(left, xs, tags.shape[0]).min(axis=0)
    gap = l_edge - r_edge - 1  # septum thickness, voxels
    rows = both & (l_edge > r_edge) & (gap * spacing[0] <= MAX_SEPTUM_MM)  # a septum lies between the medial edges
    if rows.any():
        # where a cavity's medial air is gone, its "edge" jumps to an air pocket further out and the septum looks
        # much thicker than elsewhere: such rows do not measure the septum, so they are left out
        rows &= gap <= np.median(gap[rows]) + SEPTUM_THICKER_VOX
    j, k = np.nonzero(rows)
    out = {"j": j, "k": k, "right_edge": r_edge[rows], "left_edge": l_edge[rows]}
    mid = (out["right_edge"] + out["left_edge"]) / 2.0
    out["midline"], out["deviation"] = mid, np.zeros_like(mid)
    if len(mid) >= 3:
        out["deviation"] = mid - _septal_plane(mid, j, k)
    return out


def _septal_plane(mid: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The plane the straight part of the septum follows, at each row. Least absolute deviations (iteratively
    reweighted) first, then Tukey's biweight, which gives rows far off the plane no weight at all: a local bulge
    then shows up as deviation instead of pulling the plane toward it. The biweight cutoff is set from the
    septum's own spread (median absolute deviation), but never under BIWEIGHT_MIN_VOX, so that the natural
    curvature of a septum is not taken for a bulge."""
    a = np.column_stack([np.ones_like(mid), j, k])
    w = np.ones_like(mid)
    for _ in range(30):
        coef = np.linalg.lstsq(a * w[:, None], mid * w, rcond=None)[0]
        w = 1.0 / np.sqrt(np.maximum(np.abs(mid - a @ coef), 1e-3))
    for _ in range(30):
        resid = mid - a @ coef
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        cutoff = max(4.685 * scale, BIWEIGHT_MIN_VOX)
        u = resid / cutoff
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
        coef = np.linalg.lstsq(a * np.sqrt(w)[:, None], mid * np.sqrt(w), rcond=None)[0]
    return a @ coef


def septoplasty(tags, cuts, spacing, depth: int = 1):
    """Straighten the septum where it deviates locally: in each coronal row of the valve and turbinate regions
    whose septal midline lies at least one voxel from the fitted septal plane, move the septum back toward the
    plane by up to `depth` voxels (never more than a quarter voxel past it). The septum keeps its thickness:
    the cavity it bulged into gains the air the other cavity gives up, and that cavity always keeps at least
    one voxel in the row. A straight septum is left unchanged."""
    rows = septum_rows(tags, spacing)
    out = tags.copy()
    right_lab, left_lab = anatomy.LABELS["right"], anatomy.LABELS["left"]
    for j, k, r_edge, l_edge, dev in zip(rows["j"], rows["k"], rows["right_edge"], rows["left_edge"], rows["deviation"]):
        s = int(min(depth, np.floor(abs(dev) + 0.25)))  # a quarter voxel of tolerance for the fitted plane
        if s < 1:
            continue
        row = out[:, j, k]
        if dev > 0:  # bulges toward the left cavity: move the septum toward the patient's right (-x)
            s = min(s, int((row == right_lab).sum()) - 1)
            if s < 1:
                continue
            lost = slice(r_edge - s + 1, r_edge + 1)
            row[lost][row[lost] == right_lab] = 0
            row[l_edge - s:l_edge] = left_lab
        else:  # bulges toward the right cavity: move it toward the patient's left (+x)
            s = min(s, int((row == left_lab).sum()) - 1)
            if s < 1:
                continue
            lost = slice(l_edge, l_edge + s)
            row[lost][row[lost] == left_lab] = 0
            row[r_edge + 1:r_edge + 1 + s] = right_lab
    return out


PROCEDURE_FUNCS = {"turbinate_reduction": turbinate_reduction, "septoplasty": septoplasty, "valve_repair": valve_repair}


def apply_patient_plan(tags: np.ndarray, cuts: dict, spacing: np.ndarray, plan: tuple[str, ...]) -> np.ndarray:
    for name in PATIENT_PROCEDURES:  # fixed order, so a plan's result doesn't depend on how it was listed
        if name in plan:
            tags = PROCEDURE_FUNCS[name](tags, cuts, spacing)
            # removing the septum-side layer can cut off thin pockets of airway; a mesh with an
            # unconnected region cannot be solved, so keep the part connected to the rest
            tags[~anatomy._largest_component(tags > 0)] = 0
    return tags


# ---- parametric operations, for the surgeon agent (nasal.surgeon)

TECHNIQUES = ("turbinate_reduction", "valve_repair", "septoplasty", "widen")
REGION_NAMES = ("vestibule", "valve", "turbinate", "choana")
DIRECTIONS = {"lateral", "vertical", "all"}
MAX_DEPTH = 3  # voxel layers per operation (about 0.6 mm each); a modeling limit, not a surgical one
MAX_DEPTH_MM = 2.0  # the same limit in millimetres, for depths given in mm


def _directions(side: str, direction: str) -> list[tuple[int, int]]:
    lateral = [(0, LATERAL[side])]
    vertical = [(2, 1), (2, -1)]
    return {"lateral": lateral, "vertical": vertical, "all": [(0, 1), (0, -1), *vertical]}[direction]


def layers_for(depth_mm: float, spacing: np.ndarray, axis: int) -> int:
    """Voxel layers along `axis` closest to `depth_mm` (0 when the depth is under half a voxel there)."""
    return int(round(depth_mm / float(spacing[axis])))


def operate(tags: np.ndarray, cuts: dict, spacing: np.ndarray, technique: str, side: str = "both",
            depth: int | None = None, region: str | None = None, direction: str | None = None,
            depth_mm: float | None = None) -> np.ndarray:
    """One virtual operation with chosen side and depth, on top of whatever was done before.

    The depth is `depth_mm` (millimetres, up to MAX_DEPTH_MM), turned into voxel layers along each axis it acts
    on: the voxels are about 0.6 mm wide but 1.5 mm between slices, so 1.2 mm is 2 layers sideways and 1
    vertically. `depth` gives whole layers instead (1 to MAX_DEPTH, on every axis; the default is 1 layer).
    turbinate_reduction and valve_repair act like the fixed procedures above, on one side or both; septoplasty
    straightens local septal deviations by up to that depth (side does not apply); widen opens one region of
    one or both cavities in a chosen direction (a general cut). Raises ValueError for arguments outside these
    choices."""
    if technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {TECHNIQUES}")
    if side not in (*SIDES, "both"):
        raise ValueError("side must be left, right or both")
    if depth_mm is not None:
        if isinstance(depth_mm, bool) or not isinstance(depth_mm, (int, float)) or not 0 < depth_mm <= MAX_DEPTH_MM:
            raise ValueError(f"depth_mm must be a number of millimetres above 0 and up to {MAX_DEPTH_MM}")
        per_axis = {axis: layers_for(depth_mm, spacing, axis) for axis in range(3)}
    else:
        depth = 1 if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f"depth must be a whole number of layers from 1 to {MAX_DEPTH}")
        per_axis = {axis: depth for axis in range(3)}
    sides = SIDES if side == "both" else (side,)
    if technique == "septoplasty":
        if per_axis[0] < 1:
            raise ValueError(f"depth_mm is under half a voxel sideways ({float(spacing[0]):.2f} mm)")
        tags = septoplasty(tags, cuts, spacing, min(per_axis[0], MAX_DEPTH))
    else:
        if technique == "widen":
            if region not in REGION_NAMES:
                raise ValueError(f"widen needs region, one of {REGION_NAMES}")
            if direction not in DIRECTIONS:
                raise ValueError(f"widen needs direction, one of {sorted(DIRECTIONS)}")
            regions, how = (region,), direction
        else:
            regions, how = (("turbinate",), "lateral") if technique == "turbinate_reduction" else (("valve",), "all")
        dirs = {s: _directions(s, how) for s in sides}
        rounds = max(per_axis[axis] for d in dirs.values() for axis, _ in d)
        if rounds < 1:
            raise ValueError(f"depth_mm is under half a voxel in every direction of this operation "
                             f"(voxels {float(spacing[0]):.2f} mm wide, {float(spacing[2]):.2f} mm high)")
        for i in range(rounds):
            for s in sides:
                active = [(axis, step) for axis, step in dirs[s] if per_axis[axis] > i]
                if active:
                    tags = _grow(tags, cuts, s, regions, active)
    tags = tags.copy()
    tags[~anatomy._largest_component(tags > 0)] = 0
    return tags


def edited_volume(before: np.ndarray, after: np.ndarray, spacing: np.ndarray) -> dict:
    """Airway added and removed by an edit, in mm3, reported separately (a net volume can hide large opposite
    changes). This measures the modeled airway, not tissue removed in surgery."""
    voxel = float(np.prod(spacing))
    return {"added_mm3": round(float(((after > 0) & (before == 0)).sum()) * voxel, 1),
            "removed_mm3": round(float(((after == 0) & (before > 0)).sum()) * voxel, 1)}


def candidate_plans() -> list[tuple[str, ...]]:
    return [c for k in range(1, len(PATIENT_PROCEDURES) + 1) for c in combinations(PATIENT_PROCEDURES, k)]


def standard_rule_patient(geometry: dict) -> tuple[str, ...]:
    """Visual default: turbinate reduction, plus septoplasty if the sides look unequal."""
    areas = [geometry[f"{s}_mean_area_mm2"] for s in SIDES]
    plan = ["turbinate_reduction"]
    if max(areas) / min(areas) > PATIENT_THRESHOLDS["visible_asymmetry"]:
        plan.append("septoplasty")
    return tuple(plan)


def choose_plan(totals: dict[tuple, float]) -> tuple[str, ...]:
    """Workflow++ on resistances per plan (`totals`, () = no surgery): the smallest plan whose drop reaches the
    near-best share of the best plan's, or no surgery when even the best plan lowers resistance by less than
    the minimum benefit (both placeholder thresholds)."""
    before = totals[()]
    gains = {plan: before - r for plan, r in totals.items() if plan}
    best = max(gains.values())
    if 100 * best / before < PLACEHOLDER_THRESHOLDS["min_benefit_pct"]:
        return ()
    good = [p for p, g in gains.items() if g >= PLACEHOLDER_THRESHOLDS["near_best_gain"] * best]
    return min(good, key=lambda p: (len(p), -gains[p]))


def plus_plus_rule_patient(tags, cuts, spacing, evaluator: TagEvaluator) -> tuple[tuple[str, ...], dict]:
    """Workflow++ (`choose_plan`) on the evaluator's predictions; also returns every plan's prediction."""
    before = evaluator(tags)["total_resistance"]
    predicted = {plan: evaluator(apply_patient_plan(tags, cuts, spacing, plan))["total_resistance"]
                 for plan in candidate_plans()}
    totals = {(): before, **predicted}
    return choose_plan(totals), totals


def surrogate_evaluator(surrogate, spacing: np.ndarray) -> TagEvaluator:
    """Resistance of an airway tag volume from its measured geometry, via the patient surrogate."""
    return lambda tags: surrogate.predict(anatomy.measure(tags, spacing))


def label(plan: tuple[str, ...]) -> str:
    return "+".join(plan) or "none"


FIXED_PLAN = "turbinate_reduction+valve_repair"  # baseline: the same plan for every patient, no personalization


def compare_surgery(sim: pd.DataFrame) -> pd.DataFrame:
    """Per patient: OpenFOAM drop of each rule's plan, the plan OpenFOAM finds best, and whether
    Workflow++ (chosen with the surrogate) lands in OpenFOAM's near-best set. The fixed plan (FIXED_PLAN for
    everyone) is scored the same way, so a rule only shows personalization if it beats that baseline. No
    surgery ('none', a 0% drop) counts as near-best when OpenFOAM's best plan stays under the minimum benefit.

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
        drops["none"] = 0.0
        if drops[best_plan] < PLACEHOLDER_THRESHOLDS["min_benefit_pct"]:
            near.add("none")  # no plan helps enough: no surgery is a right answer too
        std, pp = g.standard_plan.iloc[0], g.plus_plus_plan.iloc[0]
        rows.append({
            "patient": pid, "resistance_before": before,
            "openfoam_best_plan": best_plan, "openfoam_best_drop_pct": drops[best_plan],
            "standard_plan": std, "standard_drop_pct": drops.get(std, np.nan),
            "plus_plus_plan": pp, "plus_plus_drop_pct": drops.get(pp, np.nan),
            "plus_plus_near_best": pp in near, "standard_near_best": std in near,
            "fixed_drop_pct": drops.get(FIXED_PLAN, np.nan), "fixed_near_best": FIXED_PLAN in near,
            "plus_plus_is_fixed": pp == FIXED_PLAN,
            "surrogate_drop_error_pct": float(np.mean([abs(100 * (1 - sur[p] / sur["none"]) - drops[p])
                                                       for p in drops if p in sur])),
        })
    return pd.DataFrame(rows)
