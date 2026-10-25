"""Surgeon-agent campaign: thousands of synthetic problem airways, each solved by the agent and checked in OpenFOAM.

For every case (`nasal.mutations`):

1. the problem airway is built from its base patient and simulated;
2. the oracle plan, the approved operations that exactly undo the generated pathologies, is simulated: it is the
   reference for what is reachable;
3. the surgeon agent (`nasal.surgeon`) operates with the approved techniques and checks its plans in OpenFOAM.
   It succeeds when its kept plan's resistance drop reaches 90% of the best drop known for the case (the
   oracle's or the agent's own best; the same near-best placeholder as Workflow++). Otherwise it tries again,
   told what it tried and how far it fell short, up to `max_attempts`; a case still unsolved is flagged for
   review rather than retried forever;
4. every attempt goes into an experience library; before each case the agent is shown the most similar solved
   cases (by the pattern of regional OpenFOAM resistance) and the plans that worked.

Results are appended as cases finish, so a campaign resumes where it stopped. OpenFOAM runs go through an
executor: smooth walls on expert labels (the campaign's reconstruction), the faster voxel airway, or a remote
machine over SSH.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import anatomy, mutations, surgeon
from .decision import patient as ps
from .decision.rules import PLACEHOLDER_THRESHOLDS
from .params import REGIONS, SIDES

DATA_ROOT = Path(os.environ.get("NASALSEG_ROOT", "nasalseg"))  # NasalSeg folder with images/ and labels/
MAX_ATTEMPTS = 5
SUCCESS_SHARE = PLACEHOLDER_THRESHOLDS["near_best_gain"]  # 0.9 of the best known drop; a placeholder

INVERSE = {  # pathology -> the approved operation that undoes it
    "turbinate_hypertrophy": lambda p: {"technique": "turbinate_reduction", "side": p.side, "depth": p.severity},
    "valve_collapse": lambda p: {"technique": "valve_repair", "side": p.side, "depth": p.severity},
    "choanal_narrowing": lambda p: {"technique": "widen", "side": p.side, "depth": p.severity, "region": "choana",
                                    "direction": "all"},
    "septal_deviation": lambda p: {"technique": "septoplasty", "depth": p.severity},
}


class Executor(Protocol):
    def __call__(self, tags: np.ndarray, spacing: np.ndarray, origin: np.ndarray, cuts: dict, case_dir: Path) -> dict: ...


@dataclass
class LocalSmooth:
    """Smooth walls on the expert labels, snappyHexMesh and OpenFOAM in local Docker."""
    procs: int = 8
    iterations: int = 900

    def __call__(self, tags, spacing, origin, cuts, case_dir):
        from . import smooth
        from .simulate import run_case

        smooth.write_smooth_case(case_dir, tags, spacing, origin, cuts, self.procs, self.iterations)
        run_case(case_dir)
        return smooth.read_smooth(case_dir, self.iterations)


@dataclass
class LocalVoxel:
    """The voxel airway (about 7 times faster), for trying the campaign loop out."""
    procs: int = 8
    iterations: int = 900

    def __call__(self, tags, spacing, origin, cuts, case_dir):
        from .patient_flow import run_openfoam

        return run_openfoam(tags, spacing, origin, cuts, case_dir, self.procs, self.iterations)


@dataclass
class RemoteSSH:
    """The smooth case is written here, copied to a remote machine with Docker over SSH, run there, and its results
    copied back. `host` is an SSH name from the user's ~/.ssh/config; the remote needs Docker and the OpenFOAM
    image. Not yet run against a real machine."""
    host: str
    remote_dir: str = "~/nasal-runs"
    procs: int = 16
    iterations: int = 900

    def __call__(self, tags, spacing, origin, cuts, case_dir):
        import subprocess

        from . import smooth
        from .simulate import IMAGE

        smooth.write_smooth_case(case_dir, tags, spacing, origin, cuts, self.procs, self.iterations)
        remote = f"{self.remote_dir}/{case_dir.name}"
        run = lambda *cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: E731
        run("ssh", self.host, f"mkdir -p {remote}")
        run("scp", "-r", "-q", f"{case_dir}/.", f"{self.host}:{remote}/")
        run("ssh", self.host, f"cd {remote} && docker run --rm -v $PWD:/case {IMAGE} bash -lc 'cd /case && bash Allrun'")
        run("scp", "-r", "-q", f"{self.host}:{remote}/postProcessing", str(case_dir))
        return smooth.read_smooth(case_dir, self.iterations)


def oracle_plan(pathologies: list) -> list[dict]:
    """The operations that undo the applied pathologies, last applied first."""
    return [INVERSE[p.kind](p) for p in reversed(pathologies) if p.severity > 0]


def features(record: dict) -> np.ndarray:
    """Pattern of a problem airway for finding similar cases: each region's share of its side's resistance, and
    the log ratio of the sides."""
    shares = [record[f"{s}_{r}_resistance"] / record[f"{s}_resistance"] for s in SIDES for r in REGIONS]
    return np.array(shares + [np.log(record["left_resistance"] / record["right_resistance"])])


def drop_pct(before: dict, after: dict) -> float:
    return round(100 * (1 - after["total_resistance"] / before["total_resistance"]), 2)


class Experience:
    """Every attempt, kept on disk; similar cases are retrieved by `features`, solved ones first."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def similar(self, feats: np.ndarray, k: int = 3, k_failed: int = 2) -> list[dict]:
        """The k nearest solved attempts and the k_failed nearest attempts that fell short."""
        dist = lambda r: float(np.linalg.norm(np.array(r["features"]) - feats))  # noqa: E731
        solved = sorted((r for r in self.rows if r.get("success")), key=dist)[:k]
        failed = sorted((r for r in self.rows if not r.get("success")), key=dist)[:k_failed]
        return solved + failed


def lessons(similar: list[dict]) -> str:
    """Past attempts on similar airways, as the agent sees them: what worked and what fell short."""
    if not similar:
        return ""
    lines = []
    for r in similar:
        verdict = "worked" if r.get("success") else f"fell short (at least {r['best_known_pct']}% was reachable)"
        lines.append(f"- a case where {r['summary']}: {json.dumps(r['operations'])} gave {r['drop_pct']}% lower and {verdict}")
    return "Past attempts on similar airways (check your own plan in OpenFOAM):\n" + "\n".join(lines)


def diagnosis(before: dict, after: dict) -> str:
    """Where the resistance sits after a plan, and how each region changed: written by code for the feedback."""
    ranked = sorted(((s, r) for s in SIDES for r in REGIONS), key=lambda sr: -after[f"{sr[0]}_{sr[1]}_resistance"])
    parts = [f"{s} {r} {after[f'{s}_{r}_resistance']:.4f} ({100 * (after[f'{s}_{r}_resistance'] / before[f'{s}_{r}_resistance'] - 1):+.0f}%)"
             for s, r in ranked[:4]]
    return "After that plan the highest regional resistances (Pa*s/cm^3, change from before) were: " + "; ".join(parts) + "."


def describe(record: dict) -> str:
    """Short code-written pattern description, for the experience library."""
    hard = max(SIDES, key=lambda s: record[f"{s}_resistance"])
    region = max(((s, r) for s in SIDES for r in REGIONS), key=lambda sr: record[f"{sr[0]}_{sr[1]}_resistance"])
    return f"the {hard} side was harder and the {region[0]} {region[1]} held the most resistance"


def load_base(pid: str, root: Path | None = None):
    """(labels, spacing, origin) of a NasalSeg patient, in the CT's orientation."""
    root = Path(root or DATA_ROOT)
    return anatomy.load_labels(root / f"labels/{pid}_seg.nrrd", like=root / f"images/{pid}_img.nrrd")


def run_case(case: mutations.Case, execute: Executor, client, model: str, work: Path, experience: Experience,
             provider: str = "Custom (OpenAI-compatible)", max_attempts: int = MAX_ATTEMPTS,
             log: Callable[[str], None] = print, base: Callable[[str], tuple] = load_base) -> dict:
    """Build, simulate, solve and grade one case; returns its result row."""
    start = time.time()
    lab, spacing, origin = base(case.base_patient)
    healthy, cuts = anatomy.airway(lab)
    tags, applied = mutations.apply(healthy, cuts, spacing, case.pathologies)
    runs = iter(range(10_000))
    sim = lambda t: execute(t, spacing, origin, cuts, work / case.case_id / f"run_{next(runs)}")  # noqa: E731
    before = sim(tags)
    oracle = oracle_plan(applied)

    def plan_drop(ops):
        t = tags
        for op in ops:
            t = ps.operate(t, cuts, spacing, **op)
        return drop_pct(before, sim(t))

    oracle_drop = plan_drop(oracle) if oracle else 0.0
    # The voxel septoplasty removes more airway than it adds, so an oracle that includes it can raise resistance;
    # the same plan without it is a fairer reference then.
    without = [op for op in oracle if op["technique"] != "septoplasty"]
    if without and len(without) < len(oracle):
        oracle_drop = max(oracle_drop, plan_drop(without))
    feats = features(before)
    best_known, attempts, feedback, history = oracle_drop, [], "", []
    for attempt in range(1, max_attempts + 1):
        session = surgeon.SurgerySession(case.case_id, tags, cuts, spacing, origin, sim, baseline=before)
        session.tried_before = {h["key"]: h["drop"] for h in history}
        goal = "Lower the resistance as much as the approved techniques allow, with the least surgery."
        goal += ("\n" + lessons(experience.similar(feats))) if experience.rows else ""
        goal += ("\n" + feedback) if feedback else ""
        try:
            out = surgeon.run_surgeon(session, goal, client, model, provider)
        except Exception as err:  # a model or OpenFOAM failure counts as a failed attempt
            import traceback

            where = traceback.extract_tb(err.__traceback__)[-1]
            detail = f"{type(err).__name__}: {str(err)[:160]} ({Path(where.filename).name}:{where.lineno})"
            log(f"{case.case_id} attempt {attempt}: failed ({detail})")
            attempts.append({"attempt": attempt, "error": type(err).__name__, "detail": detail})
            continue
        tried = [drop_pct(before, r) for k, r in session.simulated.items() if k != session.key([])]
        best_known = float(max([best_known, *tried]))
        drop = out["result"].get("total_change_pct")
        drop = float(-drop) if drop is not None else 0.0
        success = bool(drop > 0 and drop >= SUCCESS_SHARE * best_known)
        row = {"case_id": case.case_id, "attempt": attempt, "features": feats.tolist(), "summary": describe(before),
               "operations": out["result"]["operations"], "drop_pct": round(drop, 2), "best_known_pct": best_known,
               "success": success, "openfoam_runs": session.runs}
        experience.add(row)
        attempts.append({k: row[k] for k in ("attempt", "operations", "drop_pct", "success", "openfoam_runs")})
        log(f"{case.case_id} attempt {attempt}: {drop:.1f}% lower, best known {best_known:.1f}% -> {'success' if success else 'retry'}")
        if success:
            break
        kept = session.simulated[session.key()]
        history.append({"key": session.key(), "drop": round(drop, 1), "operations": out["result"]["operations"]})
        earlier = "\n".join(f"- attempt {i}: {json.dumps(h['operations'])} -> {h['drop']}% lower"
                             for i, h in enumerate(history, start=1))
        feedback = (f"Earlier attempts on this airway (do not repeat them):\n{earlier}\n"
                    f"At least {best_known:.1f}% is reachable with the approved techniques; aim for "
                    f"{SUCCESS_SHARE * best_known:.1f}% or more. {diagnosis(before, kept)} Work on the regions that "
                    f"still hold the most resistance, with the operation that acts on each, deeper if needed.")
    return {"case_id": case.case_id, "base_patient": case.base_patient, "pathologies": [p.__dict__ for p in applied],
            "before_total": round(before["total_resistance"], 4), "oracle_operations": oracle,
            "oracle_drop_pct": oracle_drop, "best_known_pct": best_known, "attempts": attempts,
            "solved": any(a.get("success") for a in attempts), "minutes": round((time.time() - start) / 60, 1)}


def run(cases: list[mutations.Case], out_dir: Path, execute: Executor, client, model: str,
        provider: str = "Custom (OpenAI-compatible)", max_attempts: int = MAX_ATTEMPTS, limit: int | None = None,
        log: Callable[[str], None] = print, base: Callable[[str], tuple] = load_base) -> list[dict]:
    """Solve every case not yet in out_dir/results.jsonl; resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {json.loads(line)["case_id"] for line in results_path.open(encoding="utf-8")} if results_path.exists() else set()
    experience = Experience(out_dir / "experience.jsonl")
    todo = [c for c in cases if c.case_id not in done][:limit]
    new = []
    for case in todo:
        try:
            row = run_case(case, execute, client, model, out_dir / "cases", experience, provider, max_attempts, log, base)
        except Exception as err:  # a case that cannot even be built or simulated is recorded, not fatal
            row = {"case_id": case.case_id, "base_patient": case.base_patient, "error": f"{type(err).__name__}: {err}"[:300],
                   "solved": False}
            log(f"{case.case_id}: could not run ({row['error']})")
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        new.append(row)
    return new


def report(out_dir: Path) -> dict:
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").open(encoding="utf-8")]
    ok = [r for r in rows if "error" not in r]
    solved = [r for r in ok if r["solved"]]
    first = [r for r in solved if r["attempts"][0].get("success")]
    return {"cases": len(rows), "could_not_run": len(rows) - len(ok), "solved": len(solved),
            "solved_first_try": len(first), "unsolved": len(ok) - len(solved),
            "mean_attempts_when_solved": round(float(np.mean([len(r["attempts"]) for r in solved])), 2) if solved else None,
            "openfoam_runs": int(sum(2 + sum(a.get("openfoam_runs", 0) for a in r["attempts"]) for r in ok))}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Surgeon-agent campaign on synthetic problem airways.")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="write cases.jsonl")
    g.add_argument("--n", type=int, default=1000)
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run", help="solve the cases not yet done")
    r.add_argument("--executor", choices=["smooth", "voxel", "ssh"], default="smooth")
    r.add_argument("--host", help="SSH host for --executor ssh")
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    r.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    r.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    r.add_argument("--api-key", default="ollama")
    r.add_argument("--data", type=Path, default=None, help="NasalSeg folder with images/ and labels/ (or NASALSEG_ROOT)")
    sub.add_parser("report", help="summarize results.jsonl")
    for p in (g, r, sub.choices["report"]):
        p.add_argument("--out", type=Path, default=Path("data/campaign"))
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .agents import load_patient_table

        bases = list(load_patient_table().dropna(subset=["total_resistance"]).patient)
        args.out.mkdir(parents=True, exist_ok=True)
        cases = mutations.generate(args.n, bases, args.seed)
        (args.out / "cases.jsonl").write_text("".join(c.to_json() + "\n" for c in cases), encoding="utf-8")
        print(f"wrote {len(cases)} cases to {args.out / 'cases.jsonl'}")
    elif args.command == "run":
        cases = [mutations.Case.from_json(line) for line in (args.out / "cases.jsonl").open(encoding="utf-8")]
        execute = {"smooth": lambda: LocalSmooth(args.procs), "voxel": lambda: LocalVoxel(args.procs),
                   "ssh": lambda: RemoteSSH(args.host, procs=args.procs)}[args.executor]()
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        run(cases, args.out, execute, client, args.model, max_attempts=args.attempts, limit=args.limit,
            base=lambda pid: load_base(pid, args.data))
        print(json.dumps(report(args.out), indent=2))
    else:
        print(json.dumps(report(args.out), indent=2))


if __name__ == "__main__":
    main()
