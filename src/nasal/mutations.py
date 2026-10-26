"""Synthetic problem airways for the surgeon-agent campaign.

Each case starts from a real NasalSeg airway (expert labels, cut as in `anatomy.airway`) and adds one to three
pathologies with a random side and severity, drawn from a fixed seed so every case can be rebuilt exactly:

- turbinate_hypertrophy: the lateral wall of the turbinate region grows inward (the opposite of turbinate
  reduction);
- valve_collapse: the valve region narrows sideways and vertically;
- septal_deviation: a local bulge of the septum toward one side in the middle of the valve and turbinate
  regions, trading air one for one between the cavities (the inverse of the local septoplasty);
- choanal_narrowing: the choana region narrows on one side.

Each layer is one voxel (about 0.6 mm in-plane, 1.5 mm between slices). Severities and combinations are
modeling choices, not a clinical catalogue. A mutation never closes a side: both nostrils must still reach the
outlet through the airway, otherwise the severity is lowered. The case keeps what was done, so a campaign can
grade the agent; the agent itself only sees the airway and its OpenFOAM results.
"""

import json
import random
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from . import anatomy
from .decision import patient as ps
from .params import SIDES

PATHOLOGIES = ("turbinate_hypertrophy", "valve_collapse", "septal_deviation", "choanal_narrowing")
MAX_SEVERITY = 3
SIX = ndimage.generate_binary_structure(3, 1)


def _shrink(tags: np.ndarray, cuts: dict, side: str, regions: tuple[str, ...], directions: list[tuple[int, int]]) -> np.ndarray:
    """Remove the outer layer of a side's regions on the given sides of the boundary (inverse of `ps._grow`)."""
    lab = anatomy.LABELS[side]
    names = anatomy.region_names(tags)
    own = (tags == lab) & np.isin(names, [f"{side}_{r}" for r in regions])
    edge = np.zeros_like(own)
    for axis, step in directions:  # voxels whose neighbour in that direction is not airway
        edge |= own & ~ps._shift(tags > 0, axis, -step)
    out = tags.copy()
    out[edge] = 0
    return out


def turbinate_hypertrophy(tags, cuts, spacing, side):
    return _shrink(tags, cuts, side, ("turbinate",), [(0, ps.LATERAL[side])])


def valve_collapse(tags, cuts, spacing, side):
    return _shrink(tags, cuts, side, ("valve",), [(0, 1), (0, -1), (2, 1), (2, -1)])


def choanal_narrowing(tags, cuts, spacing, side):
    return _shrink(tags, cuts, side, ("choana",), [(0, 1), (0, -1), (2, 1), (2, -1)])


BULGE_SHARE = 0.5  # the bulge covers the middle half of the septum front to back and top to bottom


def septal_deviation(tags, cuts, spacing, side):
    """A local septal bulge toward `side`, one voxel per call: in the central patch of the valve and turbinate
    septum, the septum moves into the `side` cavity, which gives up its medial voxel while the other cavity gains
    one, so the septum keeps its thickness. It is the inverse of `decision.patient.septoplasty`, which moves rows
    that stray from a plane fitted to the (mostly straight) rest of the septum back toward it."""
    rows = ps.septum_rows(tags, spacing)
    if not len(rows["j"]):
        return tags
    out = tags.copy()
    right_lab, left_lab = anatomy.LABELS["right"], anatomy.LABELS["left"]
    patch = np.ones(len(rows["j"]), dtype=bool)
    for idx in (rows["j"], rows["k"]):
        lo, hi = np.quantile(idx, [(1 - BULGE_SHARE) / 2, (1 + BULGE_SHARE) / 2])
        patch &= (idx >= lo) & (idx <= hi)
    for j, k, r_edge, l_edge in zip(rows["j"][patch], rows["k"][patch], rows["right_edge"][patch], rows["left_edge"][patch]):
        row = out[:, j, k]
        if side == "left":  # the septum moves toward the patient's left (+x): left loses, right gains
            if (row == left_lab).sum() <= 1:
                continue
            row[l_edge] = 0
            row[r_edge + 1] = right_lab
        else:
            if (row == right_lab).sum() <= 1:
                continue
            row[r_edge] = 0
            row[l_edge - 1] = left_lab
    return out


OPERATORS = {"turbinate_hypertrophy": turbinate_hypertrophy, "valve_collapse": valve_collapse,
             "septal_deviation": septal_deviation, "choanal_narrowing": choanal_narrowing}


def still_open(tags: np.ndarray, cuts: dict) -> bool:
    """Both nostrils reach the outlet through the airway (face-connected)."""
    comp, _ = ndimage.label(tags > 0, SIX)
    out_ids = set(np.unique(comp[:, :, cuts["outlet"] + 1][tags[:, :, cuts["outlet"] + 1] > 0])) - {0}
    for side, inlet in anatomy.INLETS.items():
        j = cuts[inlet] + 1
        face = comp[:, j, :][tags[:, j, :] == anatomy.LABELS[side]]
        if not (set(np.unique(face)) - {0}) & out_ids:
            return False
    return all((tags == anatomy.LABELS[s]).any() for s in SIDES)


@dataclass
class Pathology:
    kind: str
    side: str
    severity: int


@dataclass
class Case:
    case_id: str
    base_patient: str
    seed: int
    pathologies: list[Pathology] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> "Case":
        d = json.loads(text)
        return Case(d["case_id"], d["base_patient"], d["seed"], [Pathology(**p) for p in d["pathologies"]])


def apply(tags: np.ndarray, cuts: dict, spacing: np.ndarray, pathologies: list[Pathology]) -> tuple[np.ndarray, list[Pathology]]:
    """Apply the pathologies in order, lowering a severity when it would close a side; returns the airway and
    the pathologies as actually applied (a severity can end at 0, which means the airway could not take it)."""
    applied = []
    for p in pathologies:
        done = 0
        for _ in range(p.severity):
            trial = OPERATORS[p.kind](tags, cuts, spacing, p.side)
            trial[~anatomy._largest_component(trial > 0)] = 0
            if not still_open(trial, cuts):
                break
            tags, done = trial, done + 1
        applied.append(Pathology(p.kind, p.side, done))
    return tags, applied


def random_case(case_id: str, base_patients: list[str], seed: int) -> Case:
    """One to three pathologies with random kinds, sides and severities, from `seed`."""
    rng = random.Random(seed)
    base = rng.choice(base_patients)
    kinds = rng.sample(PATHOLOGIES, rng.choice((1, 1, 2, 2, 3)))
    return Case(case_id, base, seed, [Pathology(k, rng.choice(SIDES), rng.randint(1, MAX_SEVERITY)) for k in kinds])


def generate(n: int, base_patients: list[str], seed: int = 0) -> list[Case]:
    """`n` reproducible cases; case i uses seed `seed * 1_000_003 + i`."""
    return [random_case(f"C{seed:02d}-{i:05d}", sorted(base_patients), seed * 1_000_003 + i) for i in range(n)]
