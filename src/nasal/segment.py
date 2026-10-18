"""CT to airway: segment the nasal airway from a raw CT volume, with no expert labels.

The output is a label volume in the NasalSeg convention (3 right nasal cavity, 4 left,
5 nasopharynx, 0 everything else), so `anatomy.airway`, meshing and OpenFOAM run on it
unchanged. The input is a CT volume (a stack of slices); a single 2D image cannot give a
3D airway.

Method:
1. Air is every voxel below `threshold` HU (default -300). Air outside the head is dropped:
   per axial slice, only air enclosed by tissue is kept.
2. The air is split into regions at its narrow necks (watershed on the distance to the
   wall), so a sinus joined to the nasal cavity through a thin opening becomes its own region.
3. A random forest, trained on the NasalSeg expert labels, decides from each region's
   position, thickness, size and intensity whether it is airway. It is evaluated with
   patient-grouped cross-validation (`nasal-segment eval`).
4. The largest connected group of airway regions is kept and labelled: in each coronal
   slice, air that spans the midline is nasopharynx, the rest the left or right cavity
   (LPS: +x is the patient's left).
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from .anatomy import LABELS, _largest_component, load_labels

AIR_THRESHOLD_HU = -300  # best of -200/-300/-400 in the grouped CV (mean airway Dice 0.82 to 0.83)
PEAK_MIN_DISTANCE = 3  # voxels between watershed seeds
MIDLINE_BAND_MM = 4.0
DUPLICATE_SCANS = {"P003": "P002", "P010": "P009", "P081": "P080"}  # identical geometry in NasalSeg
FEATURES = ("volume_mm3", "max_dt_mm", "mean_dt_mm", "dx_mm", "min_dx_mm", "y_rel", "z_rel", "y_mm", "z_mm",
            "mean_hu", "touches_midline", "neighbours")
SIX = ndimage.generate_binary_structure(3, 1)
MODEL_PATH = Path("data/segment_model.joblib")


def load_ct(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CT volume (HU), spacing (mm) and origin (mm, LPS); same layout and axis flips as the labels."""
    return load_labels(path)


def enclosed_air(air: np.ndarray) -> np.ndarray:
    """Air enclosed by tissue in each axial slice (axis 2), so air outside the head is dropped."""
    out = np.zeros_like(air)
    for k in range(air.shape[2]):
        out[:, :, k] = ndimage.binary_fill_holes(~air[:, :, k]) & air[:, :, k]
    return out


def midline(body: np.ndarray) -> float:
    """x index of the midsagittal plane: the centre of the head's left-right extent."""
    xs = np.nonzero(body.any(axis=(1, 2)))[0]
    return float(xs.mean()) if len(xs) else body.shape[0] / 2


def air_regions(ct: np.ndarray, spacing: np.ndarray, threshold: float = AIR_THRESHOLD_HU) -> dict:
    """Enclosed air split at its narrow necks: region ids, distance to wall (mm), midline."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    air = ct < threshold
    inside = enclosed_air(air)
    mid = midline(ndimage.binary_fill_holes(~air) | inside)
    dt = ndimage.distance_transform_edt(inside, sampling=spacing)
    peaks = peak_local_max(dt, min_distance=PEAK_MIN_DISTANCE, labels=ndimage.label(inside)[0], exclude_border=False)
    markers = np.zeros(ct.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return {"regions": watershed(-dt, markers, mask=inside), "dt": dt, "mid": mid, "inside": inside}


def region_features(ct: np.ndarray, spacing: np.ndarray, reg: dict) -> np.ndarray:
    """One row of FEATURES per region (region id i is row i - 1)."""
    lab, dt, mid, inside = reg["regions"], reg["dt"], reg["mid"], reg["inside"]
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    ids = np.arange(1, n + 1)
    ones = np.ones(lab.shape, np.int8)
    com = np.array(ndimage.center_of_mass(ones, lab, ids)).reshape(n, 3)
    dx = np.abs(np.arange(lab.shape[0]) - mid)[:, None, None] * spacing[0] * np.ones(lab.shape, np.float32)
    ys, zs = np.nonzero(inside.any(axis=(0, 2)))[0], np.nonzero(inside.any(axis=(0, 1)))[0]
    y0, y1, z0, z1 = ys.min(), ys.max(), zs.min(), zs.max()
    grown = ndimage.grey_dilation(lab, size=(3, 3, 3))
    edge = (grown != lab) & (lab > 0) & (grown > 0)
    pairs = np.unique(np.stack([lab[edge], grown[edge]], axis=1), axis=0)
    neighbours = np.bincount(pairs[:, 0], minlength=n + 1)[1:] if len(pairs) else np.zeros(n)
    return np.column_stack([
        ndimage.sum(ones, lab, ids) * spacing.prod(),
        ndimage.maximum(dt, lab, ids),
        ndimage.mean(dt, lab, ids),
        np.abs(com[:, 0] - mid) * spacing[0],
        ndimage.minimum(dx, lab, ids),
        (com[:, 1] - y0) / max(y1 - y0, 1),
        (com[:, 2] - z0) / max(z1 - z0, 1),
        (com[:, 1] - y0) * spacing[1],
        (com[:, 2] - z0) * spacing[2],
        ndimage.mean(ct, lab, ids),
        ndimage.sum(dx < MIDLINE_BAND_MM, lab, ids) > 0,
        neighbours,
    ])


def region_targets(reg: dict, expert: np.ndarray) -> np.ndarray:
    """Share of each region inside the expert airway (labels 3 to 5)."""
    n = int(reg["regions"].max())
    return ndimage.mean(np.isin(expert, list(LABELS.values())), reg["regions"], np.arange(1, n + 1))


def train(features: list[np.ndarray], targets: list[np.ndarray], seed: int = 0):
    """Random forest over regions; a region is airway when most of it lies in the expert airway.
    Regions are weighted by volume so the large ones decide the Dice."""
    from sklearn.ensemble import RandomForestClassifier

    X, share = np.vstack(features), np.concatenate(targets)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    clf.fit(X, share > 0.5, sample_weight=X[:, 0])
    return clf


MAJOR_SHARE = 0.15  # a coronal air region this share of the slice's largest counts as a main passage
JOIN_HEIGHT = 0.4  # cavities have joined when air between them spans this share of the region's height,
JOIN_SLICES = 3  # for this many coronal slices in a row (a thin bridge over the septum does not count)


def label_airway(airway: np.ndarray, mid: float) -> np.ndarray:
    """NasalSeg-style labels, walking coronal slices (axis 1) from the nostrils backwards.

    While the septum separates them, the two main air regions of a slice are the cavities:
    the one at larger x is the left (LPS), and smaller regions join the nearer side. The first
    slice where one region covers both cavities' positions starts the nasopharynx, and every
    slice behind it is nasopharynx. Sides come from the regions, not from a fixed midline, so a
    deviated septum or an off-centre head does not swap them; `mid` is only used before both
    cavities have appeared."""
    labels = np.zeros(airway.shape, dtype=np.int8)
    sides = None  # (right x, left x) of the cavities in the last slice that had both
    merged = False
    joined_run, pending = 0, []
    for j in range(airway.shape[1]):
        comps, n = ndimage.label(airway[:, j, :])
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(comps), comps, ids)
        cx = np.array([c[0] for c in ndimage.center_of_mass(np.ones_like(comps), comps, ids)])
        xmin = ndimage.minimum(np.indices(comps.shape)[0], comps, ids)
        xmax = ndimage.maximum(np.indices(comps.shape)[0], comps, ids)
        out = labels[:, j, :]
        if merged:
            out[comps > 0] = LABELS["pharynx"]
            continue
        major = [i for i in np.argsort(-size) if size[i] >= MAJOR_SHARE * size.max()]
        joined = False
        if len(major) == 1 and sides is not None and xmin[major[0]] <= sides[0] and xmax[major[0]] >= sides[1]:
            m = comps == major[0] + 1
            zs = np.nonzero(m.any(axis=0))[0]
            # rows where air runs unbroken from one cavity's centre to the other's: most of the
            # height at a choana, only a row or two where a thin septum leaves a gap at the floor
            span = m[int(round(sides[0])): int(round(sides[1])) + 1]
            joined = span.all(axis=0).sum() >= JOIN_HEIGHT * (zs.max() - zs.min() + 1)
        joined_run = joined_run + 1 if joined else 0
        pending = pending + [j] if joined else []
        if joined_run >= JOIN_SLICES:
            merged = True  # the cavities have joined: nasopharynx from the first slice of the run back
            for k in pending:
                labels[:, k, :][labels[:, k, :] > 0] = LABELS["pharynx"]
            out[comps > 0] = LABELS["pharynx"]
            continue
        if len(major) >= 2:
            r, l = sorted(major[:2], key=lambda i: cx[i])
            sides = (cx[r], cx[l])
        for i in range(n):
            if sides is None:
                side = "left" if cx[i] > mid else "right"
            else:
                side = "left" if abs(cx[i] - sides[1]) < abs(cx[i] - sides[0]) else "right"
            out[comps == i + 1] = LABELS[side]
    return labels


def segment(ct: np.ndarray, spacing: np.ndarray, model, threshold: float = AIR_THRESHOLD_HU) -> np.ndarray:
    """Raw CT (HU) -> NasalSeg-style airway labels (3 right, 4 left, 5 nasopharynx)."""
    reg = air_regions(ct, spacing, threshold)
    f = region_features(ct, spacing, reg)
    if not len(f):
        raise ValueError("no enclosed air found")
    keep = np.nonzero(model.predict_proba(f)[:, 1] > 0.5)[0] + 1
    airway = np.isin(reg["regions"], keep)
    if not airway.any():
        raise ValueError("no region was classified as airway")
    return label_airway(_largest_component(airway), reg["mid"])


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return float(2 * (a & b).sum() / total) if total else 1.0


def compare(pred: np.ndarray, expert: np.ndarray) -> dict:
    """Dice of the predicted airway against the expert labels: whole airway and per label."""
    out = {"dice_airway": dice(pred > 0, np.isin(expert, list(LABELS.values())))}
    for name, lab in LABELS.items():
        out[f"dice_{name}"] = dice(pred == lab, expert == lab)
    return out


def folds(pids: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Patient-grouped folds; duplicate scans share a fold."""
    groups = sorted({DUPLICATE_SCANS.get(p, p) for p in pids})
    order = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {g: int(order[i] % k) for i, g in enumerate(groups)}
    return {p: fold_of[DUPLICATE_SCANS.get(p, p)] for p in pids}


def prepare(ct_path: Path, label_path: Path | None, threshold: float = AIR_THRESHOLD_HU) -> dict:
    """Regions, features and (with expert labels) targets for one scan, without the large arrays
    that are only needed to build them."""
    ct, spacing, origin = load_ct(ct_path)
    reg = air_regions(ct, spacing, threshold)
    case = {"regions": reg["regions"], "mid": reg["mid"], "spacing": spacing, "origin": origin,
            "features": region_features(ct, spacing, reg)}
    if label_path is not None:
        expert, _, _ = load_labels(label_path)
        case["expert"] = expert.astype(np.int8)
        case["targets"] = region_targets(reg, expert)
    return case


def predict_case(case: dict, model) -> np.ndarray:
    keep = np.nonzero(model.predict_proba(case["features"])[:, 1] > 0.5)[0] + 1
    airway = np.isin(case["regions"], keep)
    if not airway.any():
        return np.zeros(airway.shape, np.int8)
    return label_airway(_largest_component(airway), case["mid"])


def domain_metrics(pred: np.ndarray, expert: np.ndarray, spacing: np.ndarray, surrogate=None) -> dict:
    """Compare what OpenFOAM would actually simulate: both airways after the nostril and outlet
    cuts, their measured geometry and, with a patient surrogate, their predicted resistance."""
    from .anatomy import airway, measure

    out = {}
    try:
        tags_e, _ = airway(expert)
        tags_p, _ = airway(pred)
    except ValueError as err:
        return {"domain_error": str(err)}
    out["dice_domain"] = dice(tags_p > 0, tags_e > 0)
    ge, gp = measure(tags_e, spacing), measure(tags_p, spacing)
    for key in ("volume_cm3", "min_area_mm2", "mean_area_mm2"):
        for side in ("left", "right"):
            k = f"{side}_{key}"
            out[f"{k}_expert"], out[f"{k}_auto"] = ge[k], gp[k]
    if surrogate is not None:
        re_, rp = surrogate.predict(ge), surrogate.predict(gp)
        for k in ("total_resistance", "left_resistance", "right_resistance"):
            out[f"{k}_expert"], out[f"{k}_auto"] = float(re_[k]), float(rp[k])
    return out


def cross_validate(cases: dict[str, dict], k: int = 5, seed: int = 0, surrogate=None) -> list[dict]:
    """Patient-grouped k-fold: train on the other folds, score each held-out scan."""
    fold_of = folds(list(cases), k, seed)
    rows = []
    for fold in range(k):
        train_ids = [p for p in cases if fold_of[p] != fold]
        model = train([cases[p]["features"] for p in train_ids], [cases[p]["targets"] for p in train_ids], seed)
        for pid in (p for p in cases if fold_of[p] == fold):
            c = cases[pid]
            pred = predict_case(c, model)
            row = {"patient": pid, "fold": fold, **compare(pred, c["expert"])}
            row.update(domain_metrics(pred, c["expert"], c["spacing"], surrogate))
            rows.append(row)
    return rows


def write_labels(path: Path, labels: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> None:
    """NRRD label file in the same LPS layout `anatomy.load_labels` reads."""
    import nrrd

    header = {"space": "left-posterior-superior", "space directions": np.diag(spacing),
              "space origin": origin, "encoding": "gzip"}
    nrrd.write(str(path), labels.astype(np.uint8), header)
