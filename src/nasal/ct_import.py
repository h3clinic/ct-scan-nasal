"""Open any CT the way a hospital or a dataset hands it over, and bring it into the layout the pipeline uses.

`load_ct_any(paths)` reads a DICOM series (a folder, a .zip, or the slice files themselves), NIfTI (.nii,
.nii.gz) or NRRD, and returns the volume in Hounsfield units with array axes running toward the patient's
left, posterior and superior (LPS, as `anatomy.load_labels` returns NasalSeg), positive spacing in mm and the
LPS position of the first voxel. Scans with a tilted gantry or oblique slices are refused rather than
resampled.

`crop_to_nose(hu, spacing, origin)` cuts a full-head scan down to a box around the nasal airway, placed from
two landmarks: the nose tip (the most anterior point of the head near the midline) and the hard palate (the
nasal floor, `segment.palate`). The box sizes cover the extent of the NasalSeg expert labels relative to those
landmarks over all 130 scans (side to side and front to back from the nose tip and midline, up and down from
the palate), plus a margin. NasalSeg scans themselves are already cropped and are left alone. The crop has
only been checked against NasalSeg's own geometry; on a full-head scan, look at the proposed box before
trusting the airway.

`prepare(paths)` does both and reports what it did.
"""

import tempfile
import zipfile
from pathlib import Path

import numpy as np

AXIS_TOLERANCE = 0.99  # |cosine| an image axis needs with a patient axis to count as aligned
# crop box around the landmarks, mm: NasalSeg labels reach +-50.7 mm from the midline, 130.7 mm behind the
# front of the face, 34.5 mm below and 75.0 mm above the palate (all 130 scans), plus a margin
CROP_HALF_WIDTH_MM = 55.0
CROP_BEHIND_TIP_MM = 135.0
CROP_FRONT_MARGIN_MM = 5.0
CROP_BELOW_PALATE_MM = 38.0
CROP_ABOVE_PALATE_MM = 80.0
FULL_HEAD_MM = {"x": 130.0, "z": 100.0}  # a scan wider or taller than this is treated as uncropped
TIP_BAND_MM = 15.0  # the nose tip is searched within this distance of the midline
PALATE_SEARCH_MM = (60.0, 5.0)  # the palate is searched this far below and above the nose tip's level
PALATE_WIDE_SEARCH_MM = (60.0, 40.0)  # ... then in this wider window


def to_lps(volume: np.ndarray, directions: np.ndarray, spacing: np.ndarray, origin: np.ndarray):
    """Permute and flip a volume so its axes run along +L, +P, +S.

    `directions[:, a]` is the LPS unit vector of array axis `a`, `spacing[a]` its voxel size (mm) and `origin`
    the LPS position (mm) of voxel (0, 0, 0). Returns (volume, spacing, origin) in the new layout."""
    directions = np.asarray(directions, dtype=float)
    patient_axis = np.abs(directions).argmax(axis=0)  # which of L, P, S each array axis follows
    if sorted(patient_axis) != [0, 1, 2] or np.abs(directions).max(axis=0).min() < AXIS_TOLERANCE:
        raise ValueError("the slices are oblique or the gantry was tilted; resample the scan to an axial grid first")
    order = np.argsort(patient_axis)  # array axis that follows L, then P, then S
    volume = np.transpose(volume, order)
    spacing = np.asarray(spacing, dtype=float)[order]
    signs = np.array([np.sign(directions[patient_axis[a], a]) for a in order])
    origin = np.asarray(origin, dtype=float).copy()
    for axis in range(3):
        if signs[axis] < 0:  # this axis runs toward R, A or I: flip it
            origin[axis] -= (volume.shape[axis] - 1) * spacing[axis]
            volume = np.flip(volume, axis=axis)
    return np.ascontiguousarray(volume), spacing, origin


def _dicom_files(paths: list[Path], workdir: Path) -> list[Path]:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file()]
        elif zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if not name.endswith("/") and ".." not in Path(name).parts:
                        files.append(Path(z.extract(name, workdir)))
        else:
            files.append(p)
    return files


def load_dicom(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The largest CT series among the files (a folder, a .zip or slice files): HU volume in LPS."""
    import pydicom

    with tempfile.TemporaryDirectory() as tmp:
        series: dict[str, list] = {}
        for f in _dicom_files(paths, Path(tmp)):
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=False)
            except Exception:
                continue  # not a DICOM file (DICOMDIR, readme, ...)
            if "ImagePositionPatient" not in ds or "PixelData" not in ds or getattr(ds, "Modality", "CT") != "CT":
                continue
            series.setdefault(str(getattr(ds, "SeriesInstanceUID", "")), []).append(ds)
        if not series:
            raise ValueError("no CT image slices found in the DICOM files")
        uid, slices = max(series.items(), key=lambda kv: len(kv[1]))
        if len(slices) < 10:
            raise ValueError(f"the largest CT series has only {len(slices)} slices; a 3D airway needs a stack of slices")
        row, col = (np.array(slices[0].ImageOrientationPatient[i:i + 3], dtype=float) for i in (0, 3))
        normal = np.cross(row, col)
        slices.sort(key=lambda ds: float(np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float))))
        positions = np.array([np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float)) for ds in slices])
        gaps = np.diff(positions)
        if np.any(gaps <= 0) or gaps.max() - gaps.min() > 0.05 * np.median(gaps):
            raise ValueError("the slices are not evenly spaced (duplicates or a varying slice spacing)")
        try:
            volume = np.stack([ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1))
                               + float(getattr(ds, "RescaleIntercept", 0)) for ds in slices], axis=-1)
        except Exception as err:  # compressed transfer syntaxes need extra decoders
            raise ValueError(f"could not decode the pixel data ({type(err).__name__}); export the series "
                             "uncompressed or as NIfTI") from err
        volume = np.transpose(volume, (1, 0, 2))  # pixel_array is (rows, columns): columns follow `row`
        pixel = [float(v) for v in slices[0].PixelSpacing]  # (between rows, between columns)
        spacing = np.array([pixel[1], pixel[0], float(np.median(gaps))])
        directions = np.column_stack([row, col, normal])
        origin = np.array(slices[0].ImagePositionPatient, dtype=float)
        hu, spacing, origin = to_lps(volume, directions, spacing, origin)
        info = {"format": "DICOM", "series": uid, "slices": len(slices), "series_found": len(series)}
    return hu, spacing, origin, info


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """NIfTI volume in LPS (NIfTI stores RAS coordinates; values are taken as HU, after the file's scaling)."""
    import nibabel as nib

    img = nib.load(str(path))
    volume = np.asarray(img.get_fdata(dtype=np.float32))
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {volume.ndim} dimensions")
    affine = img.affine.copy()
    affine[:2] *= -1  # RAS -> LPS
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    directions = affine[:3, :3] / spacing
    hu, spacing, origin = to_lps(volume, directions, spacing, affine[:3, 3])
    return hu, spacing, origin, {"format": "NIfTI", "slices": int(volume.shape[2])}


def load_ct_any(paths) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(HU volume in LPS, spacing mm, origin mm, info) from DICOM, NIfTI or NRRD; see the module docstring."""
    paths = [Path(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    if len(paths) == 1 and paths[0].suffix.lower() in (".nrrd", ".nhdr"):
        from .anatomy import load_labels

        hu, spacing, origin = load_labels(paths[0])
        return hu.astype(np.float32), spacing, origin, {"format": "NRRD", "slices": int(hu.shape[2])}
    if len(paths) == 1 and paths[0].name.lower().endswith((".nii", ".nii.gz")):
        return load_nifti(paths[0])
    return load_dicom(paths)


def is_full_head(hu: np.ndarray, spacing: np.ndarray) -> bool:
    return hu.shape[0] * spacing[0] > FULL_HEAD_MM["x"] or hu.shape[2] * spacing[2] > FULL_HEAD_MM["z"]


def landmarks(hu: np.ndarray, spacing: np.ndarray) -> dict:
    """Midline (x index), nose tip (y and z index) and hard palate (z index) of a head CT."""
    from scipy import ndimage

    from . import segment

    lab, n = ndimage.label(hu > -500)
    if not n:
        raise ValueError("no head found in the scan (nothing denser than -500 HU)")
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    mid = segment.midline(body)
    band = int(round(TIP_BAND_MM / spacing[0]))
    near = body[max(int(mid) - band, 0):int(mid) + band + 1]
    ys = np.nonzero(near.any(axis=(0, 2)))[0]
    tip_y = int(ys.min())
    tip_z = int(round(np.mean(np.nonzero(near[:, tip_y])[1])))
    # the palate is searched just below the nose tip first; when the tip's height is uncertain (a scan that
    # cuts through the face), in a wider window and then in the whole scan
    windows = [PALATE_SEARCH_MM, PALATE_WIDE_SEARCH_MM, None]
    for window in windows:
        if window is None:
            lo, hi = 0, hu.shape[2]
        else:
            lo = max(tip_z - int(round(window[0] / spacing[2])), 0)
            hi = min(tip_z + int(round(window[1] / spacing[2])) + 1, hu.shape[2])
        sub = hu[:, :, lo:hi]
        pal = segment.palate(sub, spacing, segment.enclosed_air(sub < segment.AIR_THRESHOLD_HU), mid)
        if pal["y_back"] >= 0:
            return {"midline_x": float(mid), "tip_y": tip_y, "tip_z": tip_z, "palate_z": lo + pal["z"],
                    "palate_search": "near the nose tip" if window is PALATE_SEARCH_MM else
                    "wider window" if window is not None else "whole scan"}
    raise ValueError("could not find the hard palate (bone with air above it near the midline)")


def crop_box(hu: np.ndarray, spacing: np.ndarray) -> tuple[tuple[slice, slice, slice], dict]:
    """Index box around the nasal airway from the landmarks, clipped to the scan."""
    mark = landmarks(hu, spacing)
    mm = lambda v, axis: int(round(v / spacing[axis]))  # noqa: E731
    x0 = int(mark["midline_x"]) - mm(CROP_HALF_WIDTH_MM, 0)
    x1 = int(mark["midline_x"]) + mm(CROP_HALF_WIDTH_MM, 0) + 1
    y0 = mark["tip_y"] - mm(CROP_FRONT_MARGIN_MM, 1)
    y1 = mark["tip_y"] + mm(CROP_BEHIND_TIP_MM, 1) + 1
    z0 = mark["palate_z"] - mm(CROP_BELOW_PALATE_MM, 2)
    z1 = mark["palate_z"] + mm(CROP_ABOVE_PALATE_MM, 2) + 1
    box = tuple(slice(max(a, 0), min(b, n)) for (a, b), n in zip(((x0, x1), (y0, y1), (z0, z1)), hu.shape))
    return box, mark


def crop_to_nose(hu: np.ndarray, spacing: np.ndarray, origin: np.ndarray):
    """(cropped HU, origin, report): the box around the nasal airway for a full-head scan, the scan unchanged
    when it is already cropped to the nose (as NasalSeg is)."""
    if not is_full_head(hu, spacing):
        return hu, origin, {"cropped": False}
    box, mark = crop_box(hu, spacing)
    start = np.array([s.start for s in box])
    report = {"cropped": True, "box": [(s.start, s.stop) for s in box], "landmarks": mark,
              "size_mm": [round((s.stop - s.start) * float(d), 1) for s, d in zip(box, spacing)]}
    return np.ascontiguousarray(hu[box]), origin + start * spacing, report


def prepare(paths) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load any CT and crop it to the nose when it is a full-head scan: (HU, spacing, origin, report)."""
    hu, spacing, origin, info = load_ct_any(paths)
    hu, origin, crop = crop_to_nose(hu, spacing, origin)
    return hu, spacing, origin, {**info, "shape": list(hu.shape),
                                 "spacing_mm": [round(float(s), 3) for s in spacing], **crop}
