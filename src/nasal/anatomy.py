"""Real patient nasal airways from the NasalSeg CT dataset. Research only, not medical advice.

NasalSeg (Zenodo record 13893419, DOI 10.1038/s41597-024-04176-1, CC BY 4.0)
has 130 CT scans with voxel labels. By their positions in LPS coordinates the
labels are: 1 right and 2 left maxillary sinus (lateral), 3 right and 4 left
nasal cavity (next to the midline), 5 nasopharynx (posterior). `check_labels`
verifies this per patient.

The flow domain is both nasal cavities plus the nasopharynx; the sinuses are
left out. Every airway voxel becomes one hexahedral OpenFOAM cell, written
directly as a polyMesh, so there is no surface meshing step that can fail on
irregular anatomy (the walls are stair-stepped at voxel size). Each nostril
inlet is a flat cut where that cavity's coronal cross-section first reaches
INLET_FRACTION of its median; the outlet is a flat cut where the pharynx's
axial cross-section, counted from below, first reaches OUTLET_FRACTION of its
widest. That keeps the narrow lower tail of the label out of the domain, so the
result is the nostril-to-nasopharynx resistance that rhinomanometry measures.
"""

import re
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

from .params import REGIONS, SIDES
from .resistance import AIR_DENSITY, CM3, REFERENCE_PRESSURE
from .simulate import AVERAGE_LAST, IMAGE, MAX_ITERATIONS, case_files, read_flux_history, run_case

DATASET = {"name": "NasalSeg", "zenodo": "https://zenodo.org/records/13893419",
           "doi": "10.1038/s41597-024-04176-1", "license": "CC BY 4.0"}
LABELS = {"right": 3, "left": 4, "pharynx": 5}
SINUS_LABELS = {"right": 1, "left": 2}
INLET_FRACTION = 0.3  # nostril cut: first coronal slice at this fraction of the cavity's median area
OUTLET_FRACTION = 0.5  # outlet cut: first axial slice (from below) at this fraction of the pharynx's widest
# Region boundaries as fractions of each cavity's length from its nostril cut.
# These follow the idealized model's names; they are positions, not landmarks.
REGION_FRACTIONS = (0.2, 0.35, 0.85)
INLETS = {"left": "inlet_left", "right": "inlet_right"}


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label volume, voxel spacing (mm) and origin (mm, LPS) from a NasalSeg .nrrd file."""
    import nrrd

    data, header = nrrd.read(str(path))
    directions = np.asarray(header["space directions"], dtype=float)
    if not np.allclose(directions, np.diag(np.diag(directions))) or header.get("space") != "left-posterior-superior":
        raise ValueError(f"{path}: expected axis-aligned LPS directions")
    data, step = np.asarray(data), np.diag(directions).copy()
    origin = np.asarray(header["space origin"], dtype=float).copy()
    for axis in np.nonzero(step < 0)[0]:  # some files store an axis reversed: flip it so indices grow along +LPS
        origin[axis] += (data.shape[axis] - 1) * step[axis]
        data = np.flip(data, axis=axis)
        step[axis] = -step[axis]
    return np.ascontiguousarray(data), step, origin


def check_labels(labels: np.ndarray) -> bool:
    """True when the label positions match LABELS: left cavity to the patient's
    left (+x in LPS) of the right one, sinuses outside both, pharynx behind (+y)."""
    centre = {}
    for name, lab in {**LABELS, **{f"sinus_{k}": v for k, v in SINUS_LABELS.items()}}.items():
        idx = np.argwhere(labels == lab)
        if not len(idx):
            return False
        centre[name] = idx.mean(axis=0)
    return bool(centre["left"][0] > centre["right"][0]
                and centre["sinus_left"][0] > centre["left"][0]
                and centre["sinus_right"][0] < centre["right"][0]
                and centre["pharynx"][1] > max(centre["left"][1], centre["right"][1]))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)  # 6-connectivity: cells must share a face
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _first_slice(counts: np.ndarray) -> int:
    nonzero = counts[counts > 0]
    return int(np.argmax(counts >= INLET_FRACTION * np.median(nonzero)))


def airway(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Tag volume (3 right cavity, 4 left, 5 pharynx, 0 solid) cut at the nostrils
    and the nasopharynx outlet, and the slice index of each cut."""
    tags = np.where(np.isin(labels, list(LABELS.values())), labels, 0).astype(np.int8)
    tags[~_largest_component(tags > 0)] = 0
    cuts = {}
    for side, inlet in INLETS.items():
        lab = LABELS[side]
        j = _first_slice((tags == lab).sum(axis=(0, 2)))  # axis 1 is posterior; low j is anterior
        tags[:, :j][tags[:, :j] == lab] = 0
        cuts[inlet] = j
    lab = LABELS["pharynx"]
    # Axis 2 is superior. The label's lower tail narrows (toward the velopharynx) and would
    # dominate the pressure drop, so the outlet sits where the nasopharynx is wide, which is
    # where rhinomanometry measures nasopharyngeal pressure.
    axial = (tags == lab).sum(axis=(0, 1))
    k = int(np.argmax(axial >= OUTLET_FRACTION * axial.max()))
    tags[:, :, :k][tags[:, :, :k] == lab] = 0
    cuts["outlet"] = k
    tags[~_largest_component(tags > 0)] = 0
    missing = [name for name, lab in LABELS.items() if not (tags == lab).any()]
    if missing:
        raise ValueError(f"airway has no {', '.join(missing)} connected to the rest after the cuts")
    return tags, cuts


def measure(tags: np.ndarray, spacing: np.ndarray) -> dict:
    """Per-side geometry measured from the cut airway (mm, mm^2, cm^3)."""
    out = {}
    for side in SIDES:
        m = tags == LABELS[side]
        counts = m.sum(axis=(0, 2)).astype(float)
        occupied = np.nonzero(counts)[0]
        area = counts[occupied] * spacing[0] * spacing[2]
        lo, hi = int(0.1 * len(area)), max(int(0.9 * len(area)), int(0.1 * len(area)) + 1)
        core = area[lo:hi]
        out.update({
            f"{side}_volume_cm3": float(m.sum() * spacing.prod() / 1000),
            f"{side}_length_mm": float(len(occupied) * spacing[1]),
            f"{side}_inlet_area_mm2": float(area[0]),
            f"{side}_min_area_mm2": float(core.min()),
            f"{side}_mean_area_mm2": float(core.mean()),
            f"{side}_min_area_pos": float((lo + int(np.argmin(core))) / len(area)),
        })
    out["pharynx_volume_cm3"] = float((tags == LABELS["pharynx"]).sum() * spacing.prod() / 1000)
    return out


def region_names(tags: np.ndarray) -> np.ndarray:
    """Label every airway voxel '<side>_<region>' using the same length fractions as
    `read_patient`; nasopharynx voxels count as choana of the nearer side."""
    names = np.full(tags.shape, "", dtype=object)
    for side in SIDES:
        m = tags == LABELS[side]
        js = np.nonzero(m.any(axis=(0, 2)))[0]
        j0, j1 = js.min(), js.max()
        bounds = [j0 + f * (j1 - j0) for f in REGION_FRACTIONS]
        jgrid = np.broadcast_to(np.arange(tags.shape[1])[None, :, None], tags.shape)
        idx = np.searchsorted(bounds, jgrid, side="right")
        for i, region in enumerate(REGIONS):
            names[m & (idx == i)] = f"{side}_{region}"
    pharynx = tags == LABELS["pharynx"]
    if pharynx.any():
        xs = {side: np.argwhere(tags == LABELS[side])[:, 0].mean() for side in SIDES}
        xgrid = np.broadcast_to(np.arange(tags.shape[0])[:, None, None], tags.shape)
        nearer_left = np.abs(xgrid - xs["left"]) < np.abs(xgrid - xs["right"])
        names[pharynx & nearer_left] = "left_choana"
        names[pharynx & ~nearer_left] = "right_choana"
    return names


# ---------------------------------------------------------------- voxel mesh

# Quad corners (as node offsets from the cell's own (i, j, k)) for each outward face direction,
# ordered so the right-hand normal points out of the cell.
QUADS = {
    (0, +1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, +1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (2, +1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}


def voxel_mesh(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict) -> dict:
    """OpenFOAM polyMesh arrays (points in metres) with one hex cell per airway voxel.
    Patches: inlet_left, inlet_right, outlet and wall."""
    cells = np.argwhere(tags > 0)  # C order, so a +axis neighbour always has a larger id
    shape = np.array(tags.shape)
    ids = -np.ones(tuple(shape + 2), dtype=np.int64)
    ids[tuple((cells + 1).T)] = np.arange(len(cells))
    node_dims = shape + 1

    def quad_nodes(c, key):
        corners = np.stack([c + np.array(o) for o in QUADS[key]], axis=1)  # (n, 4, 3)
        return (corners[..., 0] * node_dims[1] + corners[..., 1]) * node_dims[2] + corners[..., 2]

    internal, boundary = [], {"inlet_left": [], "inlet_right": [], "outlet": [], "wall": []}
    tag_of = tags[tuple(cells.T)]
    for axis in range(3):
        step = np.zeros(3, dtype=int); step[axis] = 1
        nb_plus = ids[tuple((cells + 1 + step).T)]
        nb_minus = ids[tuple((cells + 1 - step).T)]
        inner = nb_plus >= 0
        own = np.arange(len(cells))
        internal.append((own[inner], nb_plus[inner], quad_nodes(cells[inner], (axis, +1))))
        for sign, nb in ((+1, nb_plus), (-1, nb_minus)):
            out = nb < 0
            c, t = cells[out], tag_of[out]
            patch = np.full(len(c), "wall", dtype=object)
            if (axis, sign) == (1, -1):
                for side, inlet in INLETS.items():
                    patch[(t == LABELS[side]) & (c[:, 1] == cuts[inlet])] = inlet
            if (axis, sign) == (2, -1):
                patch[(t == LABELS["pharynx"]) & (c[:, 2] == cuts["outlet"])] = "outlet"
            q = quad_nodes(c, (axis, sign))
            for name in boundary:
                sel = patch == name
                boundary[name].append((own[out][sel], q[sel]))

    owner = np.concatenate([o for o, _, _ in internal])
    neighbour = np.concatenate([n for _, n, _ in internal])
    faces = np.concatenate([q for _, _, q in internal])
    order = np.lexsort((neighbour, owner))  # upper-triangular order
    owner, neighbour, faces = owner[order], neighbour[order], faces[order]
    patches = []
    for name, parts in boundary.items():
        o = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, int)
        q = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 4), int)
        patches.append({"name": name, "type": "wall" if name == "wall" else "patch",
                        "start": len(owner) + sum(len(p["owner"]) for p in patches), "owner": o, "faces": q})
    all_faces = np.concatenate([faces] + [p["faces"] for p in patches])
    used, inverse = np.unique(all_faces, return_inverse=True)
    all_faces = inverse.reshape(all_faces.shape)
    nodes = np.stack(np.unravel_index(used, tuple(node_dims)), axis=1)
    points = (origin + (nodes - 0.5) * spacing) * 1e-3
    return {
        "points": points, "faces": all_faces, "n_internal": len(owner), "n_cells": len(cells),
        "owner": np.concatenate([owner] + [p["owner"] for p in patches]), "neighbour": neighbour,
        "patches": [{k: p[k] for k in ("name", "type", "start")} | {"n": len(p["owner"])} for p in patches],
    }


def _foam(cls: str, obj: str, body: str) -> str:
    return (f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    location \"constant/polyMesh\";\n"
            f"    object {obj};\n}}\n\n{body}")


def write_polymesh(mesh: dict, poly_dir: Path) -> None:
    poly_dir.mkdir(parents=True, exist_ok=True)
    pts = "\n".join(f"({x:.9g} {y:.9g} {z:.9g})" for x, y, z in mesh["points"])
    (poly_dir / "points").write_text(_foam("vectorField", "points", f"{len(mesh['points'])}\n(\n{pts}\n)\n"), newline="\n")
    fcs = "\n".join(f"4({a} {b} {c} {d})" for a, b, c, d in mesh["faces"])
    (poly_dir / "faces").write_text(_foam("faceList", "faces", f"{len(mesh['faces'])}\n(\n{fcs}\n)\n"), newline="\n")
    for name, arr in (("owner", mesh["owner"]), ("neighbour", mesh["neighbour"])):
        body = f"{len(arr)}\n(\n" + "\n".join(map(str, arr)) + "\n)\n"
        (poly_dir / name).write_text(_foam("labelList", name, body), newline="\n")
    entries = "\n".join(f"    {p['name']}\n    {{\n        type {p['type']};\n        nFaces {p['n']};\n"
                        f"        startFace {p['start']};\n    }}" for p in mesh["patches"])
    (poly_dir / "boundary").write_text(
        _foam("polyBoundaryMesh", "boundary", f"{len(mesh['patches'])}\n(\n{entries}\n)\n"), newline="\n")


# ---------------------------------------------------------------- simulation

def write_patient_case(case_dir: Path, label_path: Path, processes: int = 4, turbulence: str = "kOmegaSST") -> dict:
    """Write a complete OpenFOAM case for one patient; returns its geometry record."""
    labels, spacing, origin = load_labels(label_path)
    tags, cuts = airway(labels)
    return {**write_airway_case(case_dir, tags, spacing, origin, cuts, processes, turbulence),
            "labels_ok": check_labels(labels)}


def write_airway_case(case_dir: Path, tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict,
                      processes: int = 4, turbulence: str = "kOmegaSST") -> dict:
    """Write a complete OpenFOAM case for an airway tag volume (for example after
    virtual surgery); returns its geometry record."""
    mesh = voxel_mesh(tags, spacing, origin, cuts)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for rel, text in case_files(processes, turbulence, inlets=tuple(INLETS.values()), sample_planes=False,
                                mesh_commands=[], write_format="ascii", reconstruct=processes > 1).items():
        path = case_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    write_polymesh(mesh, case_dir / "constant" / "polyMesh")
    np.savez_compressed(case_dir / "airway.npz", tags=tags, spacing=spacing, origin=origin,
                        cuts=np.array([cuts["inlet_left"], cuts["inlet_right"], cuts["outlet"]]))
    return {"cells": mesh["n_cells"], **measure(tags, spacing),
            **{f"{p['name']}_faces": p["n"] for p in mesh["patches"]}}


_NUMBER_LIST = re.compile(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\n?\(\s*(.*?)\)\s*;", re.S)


def read_internal_field(path: Path) -> np.ndarray:
    """Cell values from an ASCII OpenFOAM volScalarField or volVectorField."""
    match = _NUMBER_LIST.search(path.read_text())
    if not match:
        raise ValueError(f"no nonuniform internalField in {path}")
    kind, n, body = match.group(1), int(match.group(2)), match.group(3)
    values = np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float)
    return values.reshape(n, 3) if kind == "vector" else values


def _latest_time(case_dir: Path) -> Path:
    times = [d for d in case_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name) and d.name != "0"]
    return max(times, key=lambda d: float(d.name))


def read_patient(case_dir: Path, max_iterations: int = MAX_ITERATIONS) -> dict:
    """Per-side flow and region resistances (Pa*s/cm^3) from a finished patient case,
    with the same column names as `nasal.resistance.evaluate`."""
    saved = np.load(case_dir / "airway.npz")
    tags, spacing = saved["tags"], saved["spacing"]
    cells = np.argwhere(tags > 0)
    post = case_dir / "postProcessing"
    flows, checks = {}, {}
    for side, inlet in INLETS.items():
        hist = read_flux_history(next((post / f"{inlet}Flow").glob("*/surfaceFieldValue.dat")))
        iterations = int(hist[-1, 0])
        window = hist[hist[:, 0] > iterations - AVERAGE_LAST, 1]
        flows[side] = -float(np.mean(window)) if iterations >= max_iterations else -float(hist[-1, 1])
        checks[f"{side}_unsteadiness"] = float(np.std(window) / abs(np.mean(window)))
        checks["iterations"] = iterations
    converged = checks["iterations"] < max_iterations
    tdir = _latest_time(case_dir)
    p_name, u_name = ("p", "U") if converged else ("pMean", "UMean")
    p = read_internal_field(tdir / p_name)
    u = read_internal_field(tdir / u_name)
    pt = p + 0.5 * (u ** 2).sum(axis=1)
    record, total_flow = {}, 0.0
    for side in SIDES:
        on_side = tags[tuple(cells.T)] == LABELS[side]
        js = cells[on_side, 1]
        j0, j1 = js.min(), js.max()
        bounds = [int(round(j0 + f * (j1 - j0))) for f in REGION_FRACTIONS]
        pts = [REFERENCE_PRESSURE]
        for j in bounds:
            sel = on_side & (cells[:, 1] == j)
            flux = u[sel, 1]
            pts.append(AIR_DENSITY * float((pt[sel] * flux).sum() / flux.sum()))
        pts.append(0.0)
        q_cm3 = flows[side] / CM3
        total_flow += q_cm3
        record[f"{side}_flow_cm3_s"] = q_cm3
        record[f"{side}_resistance"] = REFERENCE_PRESSURE / q_cm3
        for i, region in enumerate(REGIONS):
            record[f"{side}_{region}_resistance"] = (pts[i] - pts[i + 1]) / q_cm3
    return {"total_resistance": REFERENCE_PRESSURE / total_flow, "total_flow_cm3_s": total_flow,
            **record, "converged": converged, **checks}


def simulate_patient(label_path: Path, case_dir: Path, processes: int = 4, turbulence: str = "kOmegaSST",
                     image: str = IMAGE) -> dict:
    """Build, run and read one patient; returns geometry plus flow record."""
    geometry = write_patient_case(case_dir, label_path, processes, turbulence)
    run_case(case_dir, image)
    return {**geometry, **read_patient(case_dir)}
