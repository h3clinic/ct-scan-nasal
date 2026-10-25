"""CT import: DICOM, NIfTI and NRRD into the LPS layout, and the crop to the nose (synthetic volumes only)."""
import numpy as np
import pytest

from nasal import ct_import


def volume(shape=(12, 14, 16), seed=0):
    return np.random.default_rng(seed).integers(-1000, 1500, size=shape).astype(np.float32)


def test_to_lps_flips_and_permutes_back_to_the_same_points():
    vol = volume()
    spacing, origin = np.array([0.5, 0.6, 1.5]), np.array([-30.0, 10.0, 40.0])
    # the same scan stored with axes (z, x, y), z running inferior and y running anterior
    stored = np.flip(np.flip(np.transpose(vol, (2, 0, 1)), axis=0), axis=2)
    directions = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float).T  # columns: LPS vector of each stored axis
    s_spacing = spacing[[2, 0, 1]]
    s_origin = origin + np.array([0, (vol.shape[1] - 1) * spacing[1], (vol.shape[2] - 1) * spacing[2]])
    back, sp, org = ct_import.to_lps(stored, directions, s_spacing, s_origin)
    assert np.array_equal(back, vol) and np.allclose(sp, spacing) and np.allclose(org, origin)


def test_oblique_slices_are_refused():
    tilt = np.deg2rad(20)
    directions = np.array([[1, 0, 0], [0, np.cos(tilt), np.sin(tilt)], [0, -np.sin(tilt), np.cos(tilt)]]).T
    with pytest.raises(ValueError, match="oblique"):
        ct_import.to_lps(volume(), directions, np.ones(3), np.zeros(3))


def test_nifti_round_trip(tmp_path):
    nib = pytest.importorskip("nibabel")
    vol, spacing, origin = volume(), np.array([0.5, 0.6, 1.5]), np.array([-30.0, 10.0, 40.0])
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = origin
    affine[:2] *= -1  # LPS -> RAS, as a NIfTI file stores it
    nib.save(nib.Nifti1Image(vol, affine), tmp_path / "ct.nii.gz")
    back, sp, org, info = ct_import.load_ct_any(tmp_path / "ct.nii.gz")
    assert info["format"] == "NIfTI" and np.array_equal(back, vol)
    assert np.allclose(sp, spacing) and np.allclose(org, origin)


def write_dicom_series(folder, vol, spacing, origin, reverse=False):
    """One file per axial slice, rows along +P and columns along +L; slices stored in reverse when asked."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder.mkdir()
    series = generate_uid()
    order = range(vol.shape[2])[::-1] if reverse else range(vol.shape[2])
    for n, k in enumerate(order):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(folder / f"s{n}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Modality, ds.SeriesInstanceUID, ds.SOPInstanceUID = "CT", series, meta.MediaStorageSOPInstanceUID
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2] + k * spacing[2])]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        pixels = (vol[:, :, k].T + 1024).astype(np.uint16)  # rows follow y, columns follow x
        ds.Rows, ds.Columns = pixels.shape
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelData = pixels.tobytes()
        ds.save_as(folder / f"s{n}.dcm", enforce_file_format=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_dicom_series_round_trip_from_a_folder_or_a_zip(tmp_path, reverse):
    import shutil

    vol, spacing, origin = volume(), np.array([0.5, 0.6, 1.5]), np.array([-30.0, 10.0, 40.0])
    write_dicom_series(tmp_path / "series", vol, spacing, origin, reverse)
    back, sp, org, info = ct_import.load_ct_any(tmp_path / "series")
    assert info["format"] == "DICOM" and info["slices"] == vol.shape[2]
    assert np.array_equal(back, vol) and np.allclose(sp, spacing) and np.allclose(org, origin)
    zipped = shutil.make_archive(str(tmp_path / "series_zip"), "zip", tmp_path / "series")
    again, _, _, _ = ct_import.load_ct_any(zipped)
    assert np.array_equal(again, vol)


def test_too_few_slices_or_no_images_are_clear_errors(tmp_path):
    write_dicom_series(tmp_path / "short", volume((8, 8, 5)), np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="only 5 slices"):
        ct_import.load_ct_any(tmp_path / "short")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "readme.txt").write_text("not a scan")
    with pytest.raises(ValueError, match="no CT image slices"):
        ct_import.load_ct_any(tmp_path / "empty")


def head(shape=(260, 340, 120), spacing=(0.6, 0.6, 1.5)):
    """A crude full head: a tissue ellipsoid with a nose bump at the front, and a palate (bone) with air above it."""
    x, y, z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    mm = [g * s for g, s in zip((x, y, z), spacing)]
    cx, cz = shape[0] * spacing[0] / 2, shape[2] * spacing[2] / 2
    hu = np.full(shape, -1000.0, dtype=np.float32)
    inside = ((mm[0] - cx) / 70) ** 2 + ((mm[1] - 117) / 95) ** 2 + ((mm[2] - cz) / 85) ** 2 < 1  # front at y = 22 mm
    hu[inside] = 40.0
    nose = ((mm[0] - cx) / 12) ** 2 + ((mm[1] - 30) / 14) ** 2 + ((mm[2] - cz) / 18) ** 2 < 1  # tip at y = 16 mm
    hu[nose] = 40.0
    palate_z = int(round((cz - 25) / spacing[2]))
    xs = slice(int((cx - 15) / spacing[0]), int((cx + 15) / spacing[0]))
    ys = slice(int(45 / spacing[1]), int(90 / spacing[1]))
    hu[xs, ys, palate_z] = 600.0  # hard palate ...
    hu[xs, ys, palate_z + 1:palate_z + 30] = -1000.0  # ... with the nasal airway above it, inside the head
    return hu, np.array(spacing), palate_z


def test_full_head_scan_is_cropped_around_the_nose_and_palate():
    hu, spacing, palate_z = head()
    assert ct_import.is_full_head(hu, spacing)
    cropped, origin, report = ct_import.crop_to_nose(hu, spacing, np.zeros(3))
    mark = report["landmarks"]
    assert report["cropped"] and mark["palate_z"] == palate_z
    assert abs(mark["midline_x"] - hu.shape[0] / 2) < 3
    (x0, x1), (y0, y1), (z0, z1) = report["box"]
    assert y0 <= mark["tip_y"] and y1 - y0 <= round((ct_import.CROP_BEHIND_TIP_MM + ct_import.CROP_FRONT_MARGIN_MM) / 0.6) + 1
    assert z0 == palate_z - round(ct_import.CROP_BELOW_PALATE_MM / 1.5)
    assert np.allclose(origin, [x0 * 0.6, y0 * 0.6, z0 * 1.5]) and cropped.shape == (x1 - x0, y1 - y0, z1 - z0)


def test_a_scan_already_cropped_to_the_nose_is_left_alone():
    hu = volume((150, 180, 50))
    out, origin, report = ct_import.crop_to_nose(hu, np.array([0.6, 0.6, 1.5]), np.ones(3))
    assert out is hu and report == {"cropped": False} and np.array_equal(origin, np.ones(3))
