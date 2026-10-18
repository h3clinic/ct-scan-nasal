import argparse
from pathlib import Path

import pandas as pd

from .cases import build_table, sample_cases, write_meshes
from .params import PARAM_SPECS, CaseParams


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic nasal airway cases.")
    parser.add_argument("-n", "--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--stl", action="store_true", help="also export one STL per side per case")
    args = parser.parse_args(argv)

    table = build_table(sample_cases(args.count, args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "cases.csv", index=False)
    if args.stl:
        write_meshes(table, args.out / "meshes")
    print(f"wrote {len(table)} cases to {args.out / 'cases.csv'}")


def select_cases(table: pd.DataFrame, ids: list[str] | None = None, first: int = 0,
                 extremes: bool = False, nominal: bool = False) -> pd.DataFrame:
    """Rows to simulate: named ids, the first N rows, the rows holding the
    lowest and highest value of each parameter, and/or the nominal case
    (CaseParams defaults, case_id "nominal")."""
    keep = set(ids or [])
    unknown = keep - set(table.case_id)
    if unknown:
        raise ValueError(f"unknown case ids: {', '.join(sorted(unknown))}")
    keep |= set(table.case_id.iloc[:first])
    if extremes:
        for name in PARAM_SPECS:
            keep |= {table.case_id[table[name].idxmin()], table.case_id[table[name].idxmax()]}
    selected = table[table.case_id.isin(keep)]
    if nominal:
        row = build_table([CaseParams()]).assign(case_id="nominal")
        selected = pd.concat([row, selected], ignore_index=True)
    return selected


def simulate_main(argv: list[str] | None = None) -> None:
    from . import simulate as sim

    parser = argparse.ArgumentParser(
        description="Run OpenFOAM (in Docker) on cases from cases.csv and write simulated.csv."
    )
    parser.add_argument("--cases", type=Path, default=Path("data/cases.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/simulated.csv"))
    parser.add_argument("--work", type=Path, default=Path("data/sim"), help="OpenFOAM case directories")
    parser.add_argument("--ids", nargs="*", help="case ids to simulate")
    parser.add_argument("--first", type=int, default=0, help="simulate the first N cases")
    parser.add_argument("--extremes", action="store_true",
                        help="simulate the cases with the lowest and highest value of each parameter")
    parser.add_argument("--nominal", action="store_true", help="also simulate the nominal case")
    parser.add_argument("--procs", type=int, default=8, help="MPI processes per run")
    parser.add_argument("--turbulence", choices=sim.TURBULENCE, default="kOmegaSST")
    parser.add_argument("--cells-across", type=int, default=sim.CELLS_ACROSS)
    parser.add_argument("--image", default=sim.IMAGE)
    args = parser.parse_args(argv)

    table = select_cases(pd.read_csv(args.cases), args.ids, args.first, args.extremes, args.nominal)
    if table.empty:
        parser.error("no cases selected: pass --ids, --first, --extremes or --nominal")
    done = pd.read_csv(args.out).to_dict("records") if args.out.exists() else []
    todo = table[~table.case_id.isin({r["case_id"] for r in done})]
    print(f"{len(table)} selected, {len(table) - len(todo)} already in {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    for row in todo.itertuples(index=False):
        params = CaseParams(**{k: getattr(row, k) for k in PARAM_SPECS})
        try:
            record = sim.simulate(params, args.work / row.case_id, args.procs, args.turbulence,
                                  args.cells_across, args.image)
        except RuntimeError as err:
            failed.append(row.case_id)
            print(f"{row.case_id}: FAILED, skipped\n{err}", flush=True)
            continue
        done.append({"case_id": row.case_id, **vars(params), "turbulence": args.turbulence, **record})
        pd.DataFrame(done).to_csv(args.out, index=False)
        print(f"{row.case_id}: total resistance {record['total_resistance']:.4f} Pa*s/cm^3 "
              f"(baseline {row.total_resistance:.4f})", flush=True)
    if done:
        ratio = sim.compare_to_baseline(pd.DataFrame(done)).set_index("case_id")
        print("\nsimulated / baseline resistance:")
        print(ratio.round(2).to_string())
    if failed:
        raise SystemExit(f"failed cases: {', '.join(failed)}")


def surrogate_main(argv: list[str] | None = None) -> None:
    import json

    from . import surrogate as sg

    parser = argparse.ArgumentParser(
        description="Train the fast surrogate on simulated cases and report its cross-validated error."
    )
    parser.add_argument("--sim", type=Path, nargs="+", default=[Path("data/simulated.csv")],
                        help="simulated.csv files to train on (merged by case_id)")
    parser.add_argument("--model", type=Path, default=Path("data/surrogate.pt"))
    parser.add_argument("--report", type=Path, default=Path("data/surrogate_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=sg.EPOCHS)
    parser.add_argument("--patients", action="store_true",
                        help="train the real-anatomy surrogate on patient_sim CSVs from nasal-patients")
    args = parser.parse_args(argv)

    if args.patients:
        table = pd.concat([pd.read_csv(p) for p in args.sim if Path(p).exists()],
                          ignore_index=True).drop_duplicates("patient", keep="last")
        targets = [c for c in table.columns if c.endswith("_resistance")]
        usable = table[targets].notna().all(axis=1) & (table[targets] > 0).all(axis=1)
        if (~usable).any():
            print(f"skipping {int((~usable).sum())} patients with a missing or non-positive region value: "
                  f"{', '.join(table.patient[~usable])}")
        table = table[usable].reset_index(drop=True)
        model_path = args.model if args.model != Path("data/surrogate.pt") else Path("data/patient_surrogate.json")
        report_path = args.report if args.report != Path("data/surrogate_report.json") else Path("data/patient_surrogate_report.json")
        result = sg.patient_report(table, args.folds)
        sg.PatientSurrogate.fit(table).save(model_path)
        table.to_csv(model_path.with_name("patient_sim.csv"), index=False)
        sg.write_report(result, report_path)
        print(json.dumps(result, indent=2))
        print(f"model: {model_path}, report: {report_path}, patients: {model_path.with_name('patient_sim.csv')}")
        return

    table = sg.load_simulated(args.sim)
    result = sg.report(table, args.folds, args.epochs)
    sg.Surrogate.fit(table, epochs=args.epochs).save(args.model)
    sg.write_report(result, args.report)
    print(json.dumps(result, indent=2))
    print(f"model: {args.model}, report: {args.report}")


def patients_main(argv: list[str] | None = None) -> None:
    from . import anatomy

    parser = argparse.ArgumentParser(description="Real patient airways from NasalSeg label volumes. Research only.")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="measure every patient's cut airway")
    s = sub.add_parser("simulate", help="run OpenFOAM on patient airways")
    for p in (m, s):
        p.add_argument("--labels", type=Path, required=True, help="NasalSeg labels folder (*_seg.nrrd)")
        p.add_argument("--ids", nargs="*", help="patient ids such as P001 (default: all)")
    m.add_argument("--out", type=Path, default=Path("data/patients.csv"))
    s.add_argument("--out", type=Path, default=Path("data/patient_sim.csv"))
    s.add_argument("--work", type=Path, default=Path("data/patient_sim"))
    s.add_argument("--procs", type=int, default=4)
    s.add_argument("--image", default=None)
    args = parser.parse_args(argv)

    files = sorted(args.labels.glob("*_seg.nrrd"))
    if args.ids:
        files = [f for f in files if f.name.split("_")[0] in set(args.ids)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(args.out).to_dict("records") if args.command == "simulate" and args.out.exists() else []
    seen = {r["patient"] for r in done}
    rows, failed = list(done), []
    for f in files:
        pid = f.name.split("_")[0]
        if pid in seen:
            continue
        try:
            if args.command == "measure":
                labels, spacing, _ = anatomy.load_labels(f)
                tags, _ = anatomy.airway(labels)
                row = {"patient": pid, "labels_ok": anatomy.check_labels(labels), **anatomy.measure(tags, spacing)}
            else:
                kwargs = {"image": args.image} if args.image else {}
                row = {"patient": pid, **anatomy.simulate_patient(f, args.work / pid, args.procs, **kwargs)}
        except Exception as err:  # one bad scan or diverged run must not stop the batch
            failed.append(pid)
            print(f"{pid}: FAILED {type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}", flush=True)
            continue
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        summary = (f"total resistance {row['total_resistance']:.4f} Pa*s/cm^3" if "total_resistance" in row
                   else f"min area L {row['left_min_area_mm2']:.0f} / R {row['right_min_area_mm2']:.0f} mm^2")
        print(f"{pid}: {summary}", flush=True)
    print(f"wrote {len(rows)} patients to {args.out}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


def agents_main(argv: list[str] | None = None) -> None:
    import json

    from . import agents

    parser = argparse.ArgumentParser(
        description="Run the Case, Planner, Critic and Explainer agents on a description. Research only."
    )
    parser.add_argument("request", help='e.g. "narrow nasal valve, septum bent to the left; what if the valve is widened?"')
    parser.add_argument("--surrogate", type=Path, default=agents.SURROGATE_PATH)
    parser.add_argument("--log", type=Path, default=Path("data/agent_log.json"))
    parser.add_argument("--thinking", action="store_true", help="turn on the model's reasoning mode")
    args = parser.parse_args(argv)

    patients, patient_surrogate = agents.load_patients()
    out = agents.run_pipeline(args.request, surrogate=agents.load_surrogate(args.surrogate), thinking=args.thinking,
                              patients=patients, patient_surrogate=patient_surrogate)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out["summary"])
    print(f"\n[{out['summary_source']}; {out['seconds']} s; log: {args.log}]")


if __name__ == "__main__":
    main()
