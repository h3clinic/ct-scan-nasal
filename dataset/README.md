---
license: cc-by-4.0
pretty_name: Nasal0 airflow simulations
language:
  - en
tags:
  - cfd
  - openfoam
  - medical
  - nasal-airway
  - surrogate-model
  - tabular
size_categories:
  - n<1K
configs:
  - config_name: patients
    data_files: patients/openfoam_results.csv
  - config_name: virtual_surgery
    data_files: patients/virtual_surgery_openfoam.csv
  - config_name: tubes
    data_files: tubes/openfoam_results.csv
  - config_name: tubes_formula_baseline
    data_files: tubes/cases_formula_baseline.csv
---

# Nasal0 airflow simulations

Simulated nasal airflow from [Nasal0](https://github.com/h3clinic/ct-scan-nasal): OpenFOAM results on real CT nasal airways and on idealized airways, virtual surgery runs and the surrogate models fitted to them.

Everything here is simulation output. No CT images or labels are included.

## Files

| File | Rows | What it is |
| --- | --- | --- |
| `patients/openfoam_results.csv` | 125 | One row per NasalSeg patient: airway geometry per side and OpenFOAM resistance per side and region |
| `patients/virtual_surgery_openfoam.csv` | 96 | 12 patients × 8 airways (untouched + 7 surgery plans), each run in OpenFOAM |
| `tubes/openfoam_results.csv` | 106 | Idealized airways with known valve angle, turbinate size and septum deviation, run in OpenFOAM |
| `tubes/cases_formula_baseline.csv` | 300 | Generated idealized cases with a friction-plus-narrowing formula estimate (no CFD) |
| `models/patient_surrogate.json` | | Power-law surrogate for patient resistance from 6 geometry numbers per side |
| `models/surrogate.pt` | | PyTorch surrogate for the idealized airways |
| `models/*_report.json` | | Cross-validated surrogate accuracy |
| `segmentation/cv_threshold_sweep.csv` | 520 | CT-to-airway front end: per scan and air threshold (−150, −200, −300, −400 HU), held-out Dice against the expert labels, measured geometry and surrogate resistance of the expert and automatic airways |
| `segmentation/cv_summary.json` | | Mean, median and worst Dice per threshold |

## Simulation setup

- **Solver:** OpenFOAM v2606 `simpleFoam`, steady, k-omega SST turbulence.
- **Driving pressure:** 150 Pa from nostrils to the nasopharynx (the rhinomanometry reference pressure).
- **Real airways:** NasalSeg labels (left cavity, right cavity, nasopharynx) turned into a voxel hex mesh. Inlets are cut at the nostrils, the outlet at the widest nasopharynx section. Walls are rigid.
- **Regions:** each side is split front to back into vestibule (nostril), valve, turbinate and choana (back opening).
- **Resistance:** pressure drop / flow, in Pa·s/cm³. Region resistances add up to the side total.
- **Virtual surgery:** turbinate reduction, valve repair and septoplasty, each a one-voxel (about 0.6 mm) change to the airway, in every combination.

## Main columns

- `patient` / `case_id`: patient id (P001 to P130, NasalSeg numbering) or idealized case id.
- `left_*`, `right_*` geometry: `volume_cm3`, `length_mm`, `inlet_area_mm2`, `min_area_mm2`, `mean_area_mm2`, `min_area_pos` (narrowest point, 0 front to 1 back).
- `total_resistance`, `left_resistance`, `right_resistance`: Pa·s/cm³ at 150 Pa.
- `{side}_{region}_resistance`: per region, regions `vestibule`, `valve`, `turbinate`, `choana`.
- `*_flow_cm3_s`: flow in cm³/s.
- Virtual surgery: `plan` (procedures joined by `+`, or `none`), `surrogate_total` (prediction for that plan by `models/patient_surrogate.json`, recomputed for every patient after the retraining), `standard_plan` (look-at-the-scan rule), `plus_plus_plan` (map-driven Workflow++ rule).

## Results in brief

- Median nasal resistance across 125 patients: 0.146 Pa·s/cm³.
- Patient surrogate: 16.2% mean side error (cross-validated, duplicate scans in one fold) against 30.1% for an average-patient guess.
- Virtual surgery on 10 distinct obstructed airways: the map-driven plan matched the best OpenFOAM plan for 10 of 10 patients, the look-at-the-scan rule for 1 of 10. The map-driven plan was turbinate reduction with valve repair for every patient, and that fixed plan alone does equally well.

## Limits

Rigid walls, one steady breath, no mucosa, voxel-resolution walls. The voxel septoplasty removes more airway than it adds, so it raises resistance in this model; that is a modeling artifact. NasalSeg contains duplicate scans (P002/P003, P009/P010, P080/P081).

**Fixed in `patients/openfoam_results.csv`:** 9 NasalSeg label files (P025, P042, P074, P089, P100, P101, P108, P118, P129) have a vertical axis that is flipped in their header but not in their voxels. The first release computed those patients' OpenFOAM rows on the upside-down airway, with the outlet cut at the wrong end of the nasopharynx. The 8 that have rows were re-run in the CT's orientation and their rows replaced; total resistance changed by −12% to +5%. P129 is also in `patients/virtual_surgery_openfoam.csv`; its 8 surgery rows were re-run the same way and replaced. P101 has no row, because its nasopharynx does not connect to the cavities in either orientation. The patient surrogate in `models/` was retrained on the corrected rows.

## Source and license

The real airways are derived from **NasalSeg** (Zenodo [13893419](https://zenodo.org/records/13893419), DOI [10.1038/s41597-024-04176-1](https://doi.org/10.1038/s41597-024-04176-1)), licensed CC BY 4.0. This dataset is released under CC BY 4.0 as well; please credit both NasalSeg and Nasal0.
