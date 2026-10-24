"""Look inside the nasal cavities of a surgeon-campaign case: where each synthetic pathology took away airway.

`case_figure(case)` rebuilds the case as the campaign does (`mutations.apply` on the patient's expert-label
airway) and draws coronal slices through the vestibule, valve, turbinate and choana, plus the slice each
pathology changed most, over the patient's CT. The healthy airway is outlined, the remaining airway is filled,
and the voxels each pathology removed are colored by pathology. Slices are drawn in the radiological
convention: the patient's right is on the left of each slice, top is superior.

    python -m nasal.case_view C00-00008 C00-00009        # -> data/figures/cases/<case>.png
"""

import json
from pathlib import Path

import numpy as np

from . import anatomy, mutations
from .anatomy import REGION_FRACTIONS

OUT_DIR = Path("data/figures/cases")
CASES = Path("data/campaign_pilot_voxel/cases.jsonl")
RESULTS = Path("data/campaign_pilot_voxel/results.jsonl")
COLORS = {"turbinate_hypertrophy": "#ff7f0e", "valve_collapse": "#d62728", "septal_deviation": "#e377c2",
          "choanal_narrowing": "#ffd92f"}
SLICES = (("vestibule", 0.10), ("valve", 0.275), ("turbinate, front", 0.45), ("turbinate, back", 0.70),
          ("choana", 0.925))


def removed_by_pathology(tags: np.ndarray, cuts: dict, spacing: np.ndarray, pathologies) -> tuple[np.ndarray, list]:
    """(deformed airway, [(pathology as applied, mask of the voxels it removed)]), applied in order as the
    campaign applies them."""
    current, parts = tags, []
    for p in pathologies:
        after, applied = mutations.apply(current, cuts, spacing, [p])
        parts.append((applied[0], (current > 0) & (after == 0)))
        current = after
    return current, parts


def slice_positions(tags: np.ndarray, parts: list) -> list[tuple[str, int]]:
    """(label, coronal index j) for the standard regions of the nasal cavities, then the slice each pathology
    changed most when it is not already shown."""
    cav = np.isin(tags, [anatomy.LABELS["left"], anatomy.LABELS["right"]])
    js = np.nonzero(cav.any(axis=(0, 2)))[0]
    j0, j1 = int(js.min()), int(js.max())
    out = [(name, int(round(j0 + f * (j1 - j0)))) for name, f in SLICES]
    for p, mask in parts:
        if mask.any():
            j = int(np.argmax(mask.sum(axis=(0, 2))))
            if min(abs(j - k) for _, k in out) > 2:
                out.append((f"most {p.kind.replace('_', ' ')}", j))
    return sorted(out, key=lambda t: t[1])


def case_figure(case: mutations.Case, ct: np.ndarray, labels: np.ndarray, spacing: np.ndarray,
                result: dict | None = None):
    """Matplotlib figure of the case's coronal slices (see the module docstring)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tags, cuts = anatomy.airway(labels)
    deformed, parts = removed_by_pathology(tags, cuts, spacing, case.pathologies)
    positions = slice_positions(tags, parts)
    air = tags > 0
    xs, zs = np.nonzero(air.any(axis=1))[0], np.nonzero(air.any(axis=0))[0]
    x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 12, tags.shape[0] - 1)
    z0, z1 = max(zs.min() - 5, 0), min(zs.max() + 5, tags.shape[2] - 1)
    extent = (0, (x1 - x0 + 1) * spacing[0], 0, (z1 - z0 + 1) * spacing[2])

    n = len(positions)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.2), facecolor="#1b1b1b")
    for ax, (name, j) in zip(np.atleast_1d(axes), positions):
        crop = (slice(x0, x1 + 1), j, slice(z0, z1 + 1))
        ax.imshow(np.clip(ct[crop].T, -1000, 400), cmap="gray", origin="lower", extent=extent, vmin=-1000, vmax=400)
        overlay = np.zeros((*ct[crop].T.shape, 4))
        overlay[deformed[crop].T > 0] = (0.35, 0.65, 1.0, 0.45)  # airway left after the pathologies
        for p, mask in parts:
            rgb = matplotlib.colors.to_rgb(COLORS[p.kind])
            overlay[mask[crop].T] = (*rgb, 0.95)
        ax.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
        if air[crop].any():
            ax.contour(air[crop].T.astype(float), levels=[0.5], colors="#7fffd4", linewidths=0.7, origin="lower",
                       extent=extent)
        depth_mm = (j - np.nonzero(air.any(axis=(0, 2)))[0].min()) * spacing[1]
        ax.set_title(f"{name}\n{depth_mm:.0f} mm from the nostril", color="#d0d0d0", fontsize=8)
        ax.text(0.03, 0.03, "R", color="#d0d0d0", fontsize=8, transform=ax.transAxes)
        ax.text(0.92, 0.03, "L", color="#d0d0d0", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444")

    applied = ", ".join(f"{p.kind.replace('_', ' ')} {p.side} ×{p.severity} ({int(m.sum())} voxels)" for p, m in parts)
    title = f"{case.case_id} · {case.base_patient} · {applied}"
    if result:
        best = max((a["drop_pct"] for a in result.get("attempts", [])), default=None)
        title += (f"\nbest known fix {result['oracle_drop_pct']:.1f}% lower"
                  + (f" · agent's best {best:.1f}% lower" if best is not None else "")
                  + (" · solved" if result.get("solved") else " · not solved"))
    fig.suptitle(title, color="#e0e0e0", fontsize=10)
    handles = [Patch(color="#7fffd4", label="healthy airway (outline)"), Patch(color="#59a6ff", label="airway left")]
    handles += [Patch(color=COLORS[p.kind], label=f"removed: {p.kind.replace('_', ' ')}") for p, _ in parts]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, labelcolor="#d0d0d0", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return fig


def load_cases(path: Path = CASES) -> dict[str, mutations.Case]:
    return {c.case_id: c for c in map(mutations.Case.from_json, Path(path).read_text(encoding="utf-8").splitlines())}


def load_results(path: Path = RESULTS) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    return {r["case_id"]: r for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .campaign import load_base
    from .segment import load_ct

    parser = argparse.ArgumentParser(description="Coronal slices of campaign cases: where each pathology removed airway.")
    parser.add_argument("cases", nargs="+", help="case ids such as C00-00008")
    parser.add_argument("--cases-file", type=Path, default=CASES)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    cases, results = load_cases(args.cases_file), load_results()
    args.out.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    from .campaign import DATA_ROOT

    for cid in args.cases:
        case = cases[cid]
        labels, spacing, _ = load_base(case.base_patient)
        ct, _, _ = load_ct(Path(DATA_ROOT) / f"images/{case.base_patient}_img.nrrd")
        fig = case_figure(case, ct, labels, spacing, results.get(cid))
        path = args.out / f"{cid}.png"
        fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
