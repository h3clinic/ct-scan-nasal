"""Formula baseline for nasal airflow resistance: friction plus narrowing loss.

Per side, the pressure drop is a*Q + b*Q**2 (Rohrer form):
  a: laminar friction in each region (elliptical duct, Poiseuille)
  b: minor losses at the nostril entry, each area change, and the choanal exit
Absolute values are uncalibrated until compared against OpenFOAM.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import REGION_ASPECT, REGION_LENGTH, SideGeometry, airway
from .params import REGIONS, SIDES, CaseParams

AIR_DENSITY = 1.2  # kg/m^3
AIR_VISCOSITY = 1.8e-5  # Pa*s
REFERENCE_PRESSURE = 150.0  # Pa, the standard rhinomanometry reference
ENTRY_LOSS = 0.5
EXIT_LOSS = 1.0
CM3 = 1e-6


@dataclass(frozen=True)
class SideResult:
    side: str
    flow_cm3_s: float
    resistance: float  # Pa*s/cm^3 at REFERENCE_PRESSURE
    region_resistance: dict[str, float]  # Pa*s/cm^3, sums to resistance


def _friction(region: str, area: float) -> float:
    k = REGION_ASPECT[region]
    shape = (1 + k**2) / (2 * k)
    return 8 * np.pi * AIR_VISCOSITY * REGION_LENGTH[region] * shape / area**2


def _transition_loss(up: float, down: float) -> float:
    """Quadratic coefficient for a sudden contraction or expansion."""
    if down < up:
        return 0.5 * (1 - down / up) * AIR_DENSITY / (2 * down**2)
    return AIR_DENSITY / 2 * (1 / up - 1 / down) ** 2


def _region_coefficients(geom: SideGeometry) -> dict[str, tuple[float, float]]:
    """(a, b) per region. Each area-change loss is charged to the narrower of
    the two regions, since that constriction is what causes it."""
    areas = geom.areas
    a = {r: _friction(r, areas[r]) for r in REGIONS}
    b = dict.fromkeys(REGIONS, 0.0)
    first, last = REGIONS[0], REGIONS[-1]
    b[first] += ENTRY_LOSS * AIR_DENSITY / (2 * areas[first] ** 2)
    b[last] += EXIT_LOSS * AIR_DENSITY / (2 * areas[last] ** 2)
    for up, down in zip(REGIONS, REGIONS[1:]):
        narrower = up if areas[up] <= areas[down] else down
        b[narrower] += _transition_loss(areas[up], areas[down])
    return {r: (a[r], b[r]) for r in REGIONS}


def side_result(geom: SideGeometry, pressure: float = REFERENCE_PRESSURE) -> SideResult:
    coeffs = _region_coefficients(geom)
    a = sum(c[0] for c in coeffs.values())
    b = sum(c[1] for c in coeffs.values())
    q = (-a + np.sqrt(a**2 + 4 * b * pressure)) / (2 * b)
    q_cm3 = q / CM3
    region = {r: (ca * q + cb * q**2) / q_cm3 for r, (ca, cb) in coeffs.items()}
    return SideResult(geom.side, q_cm3, pressure / q_cm3, region)


def evaluate(params: CaseParams, pressure: float = REFERENCE_PRESSURE) -> dict:
    """Flat record of per-side and per-region resistance for one case."""
    sides = {s: side_result(g, pressure) for s, g in airway(params).items()}
    total_flow = sum(r.flow_cm3_s for r in sides.values())
    record = {"total_resistance": pressure / total_flow, "total_flow_cm3_s": total_flow}
    for side in SIDES:
        r = sides[side]
        record[f"{side}_flow_cm3_s"] = r.flow_cm3_s
        record[f"{side}_resistance"] = r.resistance
        for region in REGIONS:
            record[f"{side}_{region}_resistance"] = r.region_resistance[region]
    return record
