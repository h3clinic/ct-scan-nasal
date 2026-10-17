"""Idealized two-sided nasal airway built from case parameters.

Each side is a tube along x with one cross-sectional area per region. The
numbers are illustrative anatomy-scale values, not patient data.
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .params import REGIONS, SIDES, CaseParams

CM = 1e-2
CM2 = 1e-4

# Region lengths along the airway (m) and cross-section aspect ratio
# (height / width). The turbinate region is slit-like, hence the high ratio.
REGION_LENGTH = {"vestibule": 1.5 * CM, "valve": 1.0 * CM, "turbinate": 5.0 * CM, "choana": 2.0 * CM}
REGION_ASPECT = {"vestibule": 1.5, "valve": 3.0, "turbinate": 8.0, "choana": 1.5}

VESTIBULE_AREA = 1.2 * CM2
CHOANA_AREA = 1.5 * CM2
VALVE_HEIGHT = 2.1 * CM
TURBINATE_OPEN_AREA = 2.2 * CM2
TURBINATE_NOMINAL_AREA = 1.0 * CM2
MIN_AREA = 0.05 * CM2

# Septum deviation narrows the side it bends into and widens the other.
DEVIATION_NARROWING = 0.5
DEVIATION_WIDENING = 0.2
SIDE_OFFSET = 1.0 * CM
BLEND = 0.3 * CM


@dataclass(frozen=True)
class SideGeometry:
    side: str
    areas: dict[str, float]


def _deviation_factor(side: str, deviation: float) -> float:
    toward = deviation if side == "left" else -deviation
    if toward >= 0:
        return 1.0 - DEVIATION_NARROWING * toward
    return 1.0 + DEVIATION_WIDENING * (-toward)


def side_geometry(params: CaseParams, side: str) -> SideGeometry:
    params.validate()
    dev = _deviation_factor(side, params.septum_deviation)
    valve = VALVE_HEIGHT**2 * np.tan(np.radians(params.valve_angle_deg) / 2)
    turbinate = TURBINATE_OPEN_AREA - TURBINATE_NOMINAL_AREA * params.turbinate_scale
    areas = {
        "vestibule": VESTIBULE_AREA,
        "valve": valve * dev,
        "turbinate": turbinate * dev,
        "choana": CHOANA_AREA,
    }
    return SideGeometry(side, {k: max(v, MIN_AREA) for k, v in areas.items()})


def airway(params: CaseParams) -> dict[str, SideGeometry]:
    return {side: side_geometry(params, side) for side in SIDES}


def area_profile(geom: SideGeometry, stations: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed area and aspect ratio along x, blending across region boundaries."""
    bounds = np.cumsum([0.0] + [REGION_LENGTH[r] for r in REGIONS])
    x = np.linspace(0.0, bounds[-1], stations)
    area = np.full_like(x, geom.areas[REGIONS[0]])
    aspect = np.full_like(x, REGION_ASPECT[REGIONS[0]])
    for i, region in enumerate(REGIONS[1:], start=1):
        w = 0.5 * (1 + np.tanh((x - bounds[i]) / (BLEND / 2)))
        area = area * (1 - w) + geom.areas[region] * w
        aspect = aspect * (1 - w) + REGION_ASPECT[region] * w
    return x, area, aspect


def side_mesh(geom: SideGeometry, stations: int = 200, segments: int = 32) -> trimesh.Trimesh:
    """Watertight elliptical tube whose cross-section follows the area profile."""
    x, area, aspect = area_profile(geom, stations)
    width = np.sqrt(area / (np.pi * aspect))
    height = aspect * width
    y0 = SIDE_OFFSET if geom.side == "left" else -SIDE_OFFSET
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)

    rings = np.stack(
        [
            np.column_stack([np.full(segments, xi), y0 + wi * np.cos(t), hi * np.sin(t)])
            for xi, wi, hi in zip(x, width, height)
        ]
    )
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(stations - 1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = a + segments
            d = b + segments
            faces += [[a, b, c], [b, d, c]]

    start = len(verts)
    end = start + 1
    verts = np.vstack([verts, [x[0], y0, 0.0], [x[-1], y0, 0.0]])
    last = (stations - 1) * segments
    for j in range(segments):
        jn = (j + 1) % segments
        faces.append([start, jn, j])
        faces.append([end, last + j, last + jn])

    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
