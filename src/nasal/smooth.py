"""Smooth-walled airways: from the voxel airway to a smooth surface and an OpenFOAM snappyHexMesh case.

The CT voxels are 0.59 x 0.59 mm in-plane but 1.5 mm between slices, so the voxel airway has stair-stepped
walls. Here the airway becomes a signed distance field (mm, negative inside), which is smoothed with a
Gaussian; the nostril and outlet cuts are then imposed as exact planes, so the openings stay flat and in the
same place as in the voxel case. The zero level set is extracted with marching cubes and lightly Taubin-smoothed
(which smooths without shrinking), with opening vertices kept on their planes. The surface is written as one
STL with four regions (wall, inlet_left, inlet_right, outlet) and meshed with snappyHexMesh.

Region resistances use the same definition as the voxel case (`anatomy.read_patient`): flux-weighted total
pressure on cross-sections at the region boundaries, per side. The cross-sections are OpenFOAM cutting planes,
and each face is assigned to the left or right cavity by the nearest voxel of the cut airway.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from . import anatomy
from .params import REGIONS, SIDES
from .simulate import AVERAGE_LAST, MAX_ITERATIONS, case_files, read_flux_history, read_raw_surface, plane_total_pressure, side_from_planes

SIGMA_MM = 0.75  # Gaussian smoothing of the distance field; about one in-plane voxel and half a slice
EXTEND = 4  # voxels the airway is extruded past each cut before smoothing, so the openings keep their size
TAUBIN = (10, 0.5, -0.53)  # iterations, lambda, mu
PLANE_STEEPNESS = 50.0  # the cut planes dominate the field, so the openings land on one grid layer
BASE_CELL_MM = 0.8
SURFACE_LEVEL = 1  # snappyHexMesh refinement at the wall: 0.4 mm cells there
OPENING_LEVEL = 2  # 0.2 mm at the nostrils and the outlet, so a small opening is not lost


REPAIR_HU = -300.0  # the air threshold the CT segmentation uses (nasal.segment.AIR_THRESHOLD_HU)
REPAIR_STEPS = 2  # voxels the airway may grow into unlabeled CT air


def repair_labels(labels: np.ndarray, ct: np.ndarray, threshold: float = REPAIR_HU, steps: int = REPAIR_STEPS) -> np.ndarray:
    """Add unlabeled voxels that are air on the CT and within `steps` voxels of the airway (face connected),
    each with the label of the nearest airway voxel. The expert labels are conservative where the cavities
    join the nasopharynx: P022's left cavity meets it over only 21 voxel faces although 95% of the unlabeled
    voxels between them are air on the CT. Sinus labels (1, 2) are never changed, and growth starts only from
    the airway, so it stays local."""
    airway = np.isin(labels, list(anatomy.LABELS.values()))
    free_air = (labels == 0) & (ct < threshold)
    grown = airway.copy()
    six = ndimage.generate_binary_structure(3, 1)
    for _ in range(steps):
        grown |= ndimage.binary_dilation(grown, six) & free_air
    added = grown & ~airway
    out = labels.copy()
    if added.any():
        _, idx = ndimage.distance_transform_edt(~airway, return_indices=True)
        out[added] = labels[tuple(i[added] for i in idx)]
    return out


def _extruded_mask(tags: np.ndarray, cuts: dict, pad: int) -> tuple[np.ndarray, dict]:
    """Airway mask padded by `pad` voxels, with each opening extruded outward so smoothing does not close it.
    Returns the mask and, per opening, the padded index of its cut plane."""
    mask = np.pad(tags > 0, pad)
    t = np.pad(tags, pad)
    planes = {}
    for side, inlet in anatomy.INLETS.items():
        lab = anatomy.LABELS[side]
        j = cuts[inlet] + 1 + pad  # first kept slice (axis 1 grows backward)
        first = t[:, j, :] == lab
        for e in range(1, EXTEND + 1):
            mask[:, j - e, :] |= first
        planes[inlet] = j - 0.5
    k = cuts["outlet"] + 1 + pad  # first kept axial slice (axis 2 grows upward)
    first = t[:, :, k] == anatomy.LABELS["pharynx"]
    for e in range(1, EXTEND + 1):
        mask[:, :, k - e] |= first
    planes["outlet"] = k - 0.5
    return mask, planes


CT_LEVEL_HU = -500.0  # the wall: halfway between air (about -1000 HU) and soft tissue (about 0 HU)
CT_SIGMA_MM = 0.4  # light smoothing of the CT against noise
CT_MARGIN = 2  # voxels beyond the labeled airway where CT air still counts
HU_PER_MM = 1000.0  # scale that turns the HU field into a distance-like one, so the cut planes stay crisp


def _ct_field(mask: np.ndarray, ct: np.ndarray, spacing: np.ndarray, pad: int, sinus: np.ndarray | None) -> np.ndarray:
    """Distance-like field (mm, negative inside) whose zero level is where the smoothed CT crosses CT_LEVEL_HU,
    near the labeled airway only, without the sinuses, keeping only air connected to the labeled airway."""
    ctp = ndimage.gaussian_filter(np.pad(ct.astype(np.float32), pad, mode="edge"), sigma=CT_SIGMA_MM / np.asarray(spacing))
    six = ndimage.generate_binary_structure(3, 1)
    allowed = ndimage.binary_dilation(mask, six, iterations=CT_MARGIN)
    if sinus is not None:
        allowed &= ~np.pad(sinus, pad)
    field = np.where(allowed, (ctp - CT_LEVEL_HU) / HU_PER_MM, 1.0)
    air = field < 0
    comp, _ = ndimage.label(air, six)
    keep = np.isin(comp, np.unique(comp[mask & air]))
    return np.where(air & ~keep, np.abs(field), field)


def smooth_sdf(tags: np.ndarray, spacing: np.ndarray, cuts: dict, sigma_mm: float = SIGMA_MM,
               ct: np.ndarray | None = None, sinus: np.ndarray | None = None) -> tuple[np.ndarray, dict, np.ndarray, int]:
    """Smoothed field (mm, negative inside) with flat openings; also the opening planes (padded voxel index),
    the nearest-label map and the padding. Without `ct`, the field is the label's smoothed signed distance,
    shifted to keep the voxel airway's volume; with `ct` (on the same grid), the wall is where the CT crosses
    CT_LEVEL_HU near the labeled airway (`sinus` voxels excluded)."""
    pad = EXTEND + 4
    mask, planes = _extruded_mask(tags, cuts, pad)
    if ct is not None:
        sdf = _ct_field(mask, ct, spacing, pad, sinus)
    else:
        inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        sdf = ndimage.gaussian_filter(outside - inside, sigma=sigma_mm / np.asarray(spacing))
        # Smoothing thins narrow passages; shift the level so the smoothed airway keeps the voxel airway's volume.
        target = int(mask.sum())
        lo, hi = -2.0, 2.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (sdf < mid).sum() < target else (lo, mid)
        sdf = sdf - (lo + hi) / 2
    # nearest airway label for every voxel, to cut each opening only where its own cavity is
    t = np.pad(tags, pad)
    _, idx = ndimage.distance_transform_edt(t == 0, sampling=spacing, return_indices=True)
    nearest = t[tuple(idx)]
    grid = np.indices(sdf.shape)
    for side, inlet in anatomy.INLETS.items():
        own = nearest == anatomy.LABELS[side]
        cut = (planes[inlet] - grid[1]) * spacing[1] * PLANE_STEEPNESS  # positive in front of the nostril plane
        sdf = np.where(own, np.maximum(sdf, cut), sdf)
    own = nearest == anatomy.LABELS["pharynx"]
    cut = (planes["outlet"] - grid[2]) * spacing[2] * PLANE_STEEPNESS  # positive below the outlet plane
    sdf = np.where(own, np.maximum(sdf, cut), sdf)
    return sdf, planes, nearest, pad


def _taubin(verts: np.ndarray, faces: np.ndarray, fixed: np.ndarray, iterations: int, lam: float, mu: float) -> np.ndarray:
    """Taubin smoothing (alternating lambda/mu Laplacian steps, which does not shrink); `fixed` rows stay put."""
    n = len(verts)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    deg = np.bincount(edges[:, 0], minlength=n).astype(float)
    v = verts.copy()
    for _ in range(iterations):
        for f in (lam, mu):
            mean = np.zeros_like(v)
            np.add.at(mean, edges[:, 0], v[edges[:, 1]])
            step = mean / np.maximum(deg, 1)[:, None] - v
            step[fixed] = 0
            v = v + f * step
    return v


def surface(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict, sigma_mm: float = SIGMA_MM,
            ct: np.ndarray | None = None, sinus: np.ndarray | None = None) -> dict:
    """Smooth closed airway surface: vertices (mm, patient coordinates), faces, and each face's patch name."""
    from skimage import measure

    sdf, planes, nearest, pad = smooth_sdf(tags, spacing, cuts, sigma_mm, ct, sinus)
    verts, faces, _, _ = measure.marching_cubes(sdf, 0.0)  # in padded voxel index units
    # Opening faces lie within a voxel of their plane and face along its axis; snap their vertices onto it.
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    on_plane = {}
    for name, axis in [*((inlet, 1) for inlet in anatomy.INLETS.values()), ("outlet", 2)]:
        near = np.abs(tri[:, :, axis] - planes[name]).max(axis=1) < 0.75
        on_plane[name] = near & (np.abs(normal[:, axis]) > 0.9)
    fixed = np.zeros(len(verts), dtype=bool)
    for name, axis in [*((inlet, 1) for inlet in anatomy.INLETS.values()), ("outlet", 2)]:
        vs = np.unique(faces[on_plane[name]])
        verts[vs, axis] = planes[name]
        fixed[vs] = True
    idx = verts * spacing  # to mm for smoothing in true proportions
    idx = _taubin(idx, faces, fixed, *TAUBIN)
    patch = np.full(len(faces), "wall", dtype=object)
    centre = np.clip(np.rint(verts[faces].mean(axis=1)).astype(int), 0, np.array(nearest.shape) - 1)
    owner = nearest[tuple(centre.T)]  # which cavity each face belongs to (both nostrils can share one plane)
    for side, inlet in anatomy.INLETS.items():
        patch[on_plane[inlet] & (owner == anatomy.LABELS[side])] = inlet
    patch[on_plane["outlet"] & (owner == anatomy.LABELS["pharynx"])] = "outlet"
    verts_mm = origin + idx - pad * np.asarray(spacing)
    return {"verts": verts_mm, "faces": faces, "patch": patch, "sdf_shape": sdf.shape}


def write_stl(path: Path, verts_m: np.ndarray, faces: np.ndarray, patch: np.ndarray) -> None:
    """ASCII STL with one solid per patch (OpenFOAM reads the solid names as regions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for name in ("wall", "inlet_left", "inlet_right", "outlet"):
            f.write(f"solid {name}\n")
            for a, b, c in faces[patch == name]:
                p, q, r = verts_m[a], verts_m[b], verts_m[c]
                n = np.cross(q - p, r - p)
                n = n / (np.linalg.norm(n) or 1)
                f.write(f" facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n  outer loop\n")
                for v in (p, q, r):
                    f.write(f"   vertex {v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


def region_planes(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> list[dict]:
    """Cross-sections at each side's region boundaries, at the same slices `anatomy.read_patient` uses (m)."""
    out = []
    for side in SIDES:
        js = np.nonzero((tags == anatomy.LABELS[side]).any(axis=(0, 2)))[0]
        j0, j1 = js.min(), js.max()
        for i, f in enumerate(anatomy.REGION_FRACTIONS, start=1):
            j = int(round(j0 + f * (j1 - j0)))
            y = (origin[1] + j * spacing[1]) * 1e-3
            out.append({"name": f"{side}_b{i}", "side": side, "point": (0.0, y, 0.0), "normal": (0, 1, 0)})
    return out


def _foam(cls: str, obj: str, body: str) -> str:
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj)) + body


def write_smooth_case(case_dir: Path, tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict,
                      processes: int = 8, max_iterations: int = MAX_ITERATIONS, sigma_mm: float = SIGMA_MM,
                      base_cell_mm: float = BASE_CELL_MM, turbulence: str = "kOmegaSST",
                      ct: np.ndarray | None = None, sinus: np.ndarray | None = None) -> dict:
    """A complete OpenFOAM case for the smoothed airway (snappyHexMesh); returns the geometry record."""
    import shutil

    if case_dir.exists():
        shutil.rmtree(case_dir)
    surf = surface(tags, spacing, origin, cuts, sigma_mm, ct, sinus)
    verts_m = surf["verts"] * 1e-3
    write_stl(case_dir / "constant" / "triSurface" / "airway.stl", verts_m, surf["faces"], surf["patch"])
    planes = region_planes(tags, spacing, origin)
    lo, hi = verts_m.min(axis=0), verts_m.max(axis=0)
    cell = base_cell_mm * 1e-3
    lo, hi = lo - 2 * cell, hi + 2 * cell
    n = np.ceil((hi - lo) / cell).astype(int)
    # a point well inside the airway: the deepest airway voxel
    depth = ndimage.distance_transform_edt(tags > 0, sampling=spacing)
    deep = np.unravel_index(np.argmax(depth), depth.shape)
    inside = (origin + np.array(deep) * spacing) * 1e-3
    mesh = [
        "blockMesh > log.blockMesh",
        "snappyHexMesh -overwrite > log.snappyHexMesh",
    ]
    files = case_files(processes, turbulence, max_iterations, inlets=tuple(anatomy.INLETS.values()),
                       sample_planes=True, mesh_commands=mesh, write_format="ascii", reconstruct=False,
                       planes=[(p["name"], p["point"], p["normal"]) for p in planes])
    files["system/blockMeshDict"] = _foam("dictionary", "blockMeshDict", f"""
convertToMeters 1;
vertices
(
    ({lo[0]:.8g} {lo[1]:.8g} {lo[2]:.8g}) ({hi[0]:.8g} {lo[1]:.8g} {lo[2]:.8g}) ({hi[0]:.8g} {hi[1]:.8g} {lo[2]:.8g}) ({lo[0]:.8g} {hi[1]:.8g} {lo[2]:.8g})
    ({lo[0]:.8g} {lo[1]:.8g} {hi[2]:.8g}) ({hi[0]:.8g} {lo[1]:.8g} {hi[2]:.8g}) ({hi[0]:.8g} {hi[1]:.8g} {hi[2]:.8g}) ({lo[0]:.8g} {hi[1]:.8g} {hi[2]:.8g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary ( background {{ type patch; faces ( (0 3 2 1) (4 5 6 7) (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) ); }} );
""")
    regions = "\n".join(f"                {r} {{ name {r}; }}" for r in ("wall", "inlet_left", "inlet_right", "outlet"))
    patch_levels = "\n".join(
        f"                    {r} {{ level ({lv} {lv}); patchInfo {{ type {'wall' if r == 'wall' else 'patch'}; }} }}"
        for r, lv in (("wall", SURFACE_LEVEL), *((o, OPENING_LEVEL) for o in ("inlet_left", "inlet_right", "outlet"))))
    files["system/snappyHexMeshDict"] = _foam("dictionary", "snappyHexMeshDict", f"""
castellatedMesh true;
snap true;
addLayers false;
geometry
{{
    airway.stl
    {{
        type triSurfaceMesh;
        name airway;
        regions
        {{
{regions}
        }}
    }}
}}
castellatedMeshControls
{{
    maxLocalCells 4000000;
    maxGlobalCells 12000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
        airway
        {{
            level ({SURFACE_LEVEL} {SURFACE_LEVEL});
            regions
            {{
{patch_levels}
            }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}};
    locationInMesh ({inside[0]:.8g} {inside[1]:.8g} {inside[2]:.8g});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 50;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true;
    layers {{}};
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-16;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}
mergeTolerance 1e-6;
""")
    for rel, text in files.items():
        path = case_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    np.savez_compressed(case_dir / "airway.npz", tags=tags, spacing=spacing, origin=origin,
                        cuts=np.array([cuts["inlet_left"], cuts["inlet_right"], cuts["outlet"]]))
    (case_dir / "planes.json").write_text(__import__("json").dumps(planes), encoding="utf-8")
    return {**anatomy.measure(tags, spacing), "surface_faces": int(len(surf["faces"])),
            "base_cell_mm": base_cell_mm, "wall": "CT at %g HU" % CT_LEVEL_HU if ct is not None else "smoothed label",
            "smoothing_sigma_mm": sigma_mm}


def read_smooth(case_dir: Path, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Per-side flow and region resistances (Pa*s/cm^3) from a finished smooth case, in the same columns as
    `anatomy.read_patient`."""
    import json

    from scipy.spatial import cKDTree

    saved = np.load(case_dir / "airway.npz")
    tags, spacing, origin = saved["tags"], saved["spacing"], saved["origin"]
    planes = json.loads((case_dir / "planes.json").read_text(encoding="utf-8"))
    post = case_dir / "postProcessing"
    flows, checks = {}, {}
    for side, inlet in anatomy.INLETS.items():
        hist = read_flux_history(next((post / f"{inlet}Flow").glob("*/surfaceFieldValue.dat")))
        iterations = int(hist[-1, 0])
        window = hist[hist[:, 0] > iterations - AVERAGE_LAST, 1]
        flows[side] = -float(np.mean(window)) if iterations >= max_iterations else -float(hist[-1, 1])
        checks[f"{side}_unsteadiness"] = float(np.std(window) / abs(np.mean(window)))
        checks["iterations"] = iterations
    converged = checks["iterations"] < max_iterations
    p_name, u_name = ("p", "U") if converged else ("pMean", "UMean")
    plane_dir = max((d for d in (post / "planes").iterdir() if d.is_dir()), key=lambda d: float(d.name))
    vox = np.argwhere(tags > 0)
    tree = cKDTree((origin + vox * spacing) * 1e-3)
    vox_label = tags[tuple(vox.T)]
    record, total_flow = {}, 0.0
    for side in SIDES:
        pts = []
        for p in (p for p in planes if p["side"] == side):
            data = np.loadtxt(plane_dir / f"{p_name}_{p['name']}.raw", comments="#", ndmin=2)
            area, pr = read_raw_surface(plane_dir / f"{p_name}_{p['name']}.raw")
            _, u = read_raw_surface(plane_dir / f"{u_name}_{p['name']}.raw")
            _, near = tree.query(data[:, :3])
            mine = vox_label[near] == anatomy.LABELS[side]
            pts.append(plane_total_pressure(area[mine], pr[mine], u[mine]))
        result = side_from_planes(flows[side], pts)
        total_flow += result["flow_cm3_s"]
        record[f"{side}_flow_cm3_s"] = result["flow_cm3_s"]
        record[f"{side}_resistance"] = result["resistance"]
        for r in REGIONS:
            record[f"{side}_{r}_resistance"] = result["region_resistance"][r]
    from .resistance import REFERENCE_PRESSURE

    return {"total_resistance": REFERENCE_PRESSURE / total_flow, "total_flow_cm3_s": total_flow, **record,
            "converged": converged, **checks}
