"""OpenFOAM results for real patients' airways, untouched and after virtual surgery.

Stored runs come first (`nasal-patients simulate` and `nasal-patients surgery` tables). With `live=True`, a
missing airway or plan is meshed and run in OpenFOAM (Docker) with the fast settings below, and the result is
kept for the rest of the session. The agents and the studio read resistance from here, never from the
patient surrogate.
"""

import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import anatomy
from .decision import patient as ps

# Runs started here stop at 900 iterations instead of 3000: in the 13 stored 3000-iteration patient runs, total
# flow averaged up to iteration 600 was already within 0.11% of the final value, and P022 run at 900 iterations
# on 8 cores matched its stored run within 0.22% in every region (237 s, against about 25 minutes before).
FAST_ITERATIONS = 900
WORK_DIR = Path("data/studio")
_RUN_ID = itertools.count()


def run_settings(cores: int | None = None, procs: int | None = None) -> tuple[int, int]:
    """(processes per case, cases run at once) for this machine; leaves two cores free."""
    cores = max(1, (cores or os.cpu_count() or 4) - 2)
    procs = procs or min(8, cores)
    return procs, max(1, cores // procs)


def run_openfoam(tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict, case_dir: Path,
                 procs: int = 8, max_iterations: int = FAST_ITERATIONS, on_progress=None) -> dict:
    """Mesh the airway, run simpleFoam in Docker and read the result (a few minutes). With `on_progress`, the
    case also samples its fields while it runs and `on_progress(snapshot)` gets `nasal.live` snapshots."""
    from .simulate import run_case

    live_every = None
    if on_progress is not None:
        from .live import LIVE_EVERY, LiveCase

        live_every = LIVE_EVERY
    geometry = anatomy.write_airway_case(case_dir, tags, spacing, origin, cuts, procs, max_iterations=max_iterations,
                                         live_every=live_every)
    if on_progress is None:
        run_case(case_dir)
    else:
        reader = LiveCase(case_dir, max_iterations)
        run_case(case_dir, poll=lambda: on_progress(reader.poll()))
    return {**geometry, **anatomy.read_patient(case_dir, max_iterations=max_iterations)}


def plus_plus_from_totals(totals: dict[tuple, float]) -> tuple[str, ...]:
    """Workflow++ on simulated totals (`decision.patient.choose_plan`): the smallest plan near the best one, or
    no surgery below the minimum benefit. `totals` maps plans, () included."""
    from .decision.patient import choose_plan

    return choose_plan(totals)


class PatientFlow:
    """OpenFOAM records per (patient, plan label): stored runs, runs made this session, or new live runs."""

    def __init__(self, patients: pd.DataFrame | None = None, surgery: pd.DataFrame | None = None,
                 labels_dir: Path | None = None, live: bool = False, procs: int | None = None,
                 iterations: int = FAST_ITERATIONS, work_dir: Path = WORK_DIR):
        self.stored: dict[tuple[str, str], dict] = {}
        if patients is not None and "total_resistance" in patients:
            for r in patients.to_dict("records"):
                if not np.isnan(r.get("total_resistance", np.nan)):
                    self.stored[(r["patient"], "none")] = {**r, "source": "stored OpenFOAM run (3000 iterations)"}
        if surgery is not None:
            for r in surgery.to_dict("records"):  # the patient table's untouched run wins; they agree within 0.01%
                self.stored.setdefault((r["patient"], r["plan"]), {**r, "source": "stored OpenFOAM run (3000 iterations)"})
        self.runs: dict[tuple[str, str], dict] = {}
        self.labels_dir, self.live, self.iterations, self.work_dir = labels_dir, live, iterations, work_dir
        self.procs, self.parallel = run_settings(procs=procs)

    def known(self, pid: str, label: str) -> dict | None:
        return self.runs.get((pid, label)) or self.stored.get((pid, label))

    def airway(self, pid: str):
        """(tags, cuts, spacing, origin) of the patient's cut airway, in the CT's orientation."""
        files = sorted(Path(self.labels_dir).glob(f"{pid}_*seg.nrrd")) if self.labels_dir else []
        if not files:
            raise LookupError(f"{pid}: the CT labels are not loaded, so the airway cannot be simulated")
        labels, spacing, origin = anatomy.load_labels(files[0])
        tags, cuts = anatomy.airway(labels)
        return tags, cuts, spacing, origin

    def records(self, pid: str, plans: list[tuple[str, ...]]) -> dict[tuple[str, ...], dict]:
        """OpenFOAM record for each plan (() is the untouched airway), running the missing ones when live."""
        missing = [p for p in plans if self.known(pid, ps.label(p)) is None]
        if missing:
            if not self.live:
                names = ", ".join(ps.label(p) for p in missing)
                raise LookupError(f"{pid}: no OpenFOAM run for {names}; run it in nasal-studio or with nasal-patients")
            tags, cuts, spacing, origin = self.airway(pid)

            def work(plan):
                operated = ps.apply_patient_plan(tags, cuts, spacing, plan) if plan else tags
                case_dir = self.work_dir / pid / f"{ps.label(plan)}_{next(_RUN_ID)}"
                record = run_openfoam(operated, spacing, origin, cuts, case_dir, self.procs, self.iterations)
                self.runs[(pid, ps.label(plan))] = {**record, "source": f"OpenFOAM run now ({self.iterations} iterations)"}

            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                list(pool.map(work, missing))
        return {p: self.known(pid, ps.label(p)) for p in plans}
