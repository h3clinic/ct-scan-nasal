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


if __name__ == "__main__":
    main()
