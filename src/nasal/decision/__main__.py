"""python -m nasal.decision: write the standard vs Workflow++ reclassification table."""

import argparse
from pathlib import Path

import pandas as pd

from .compare import reclassify, summarize


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=Path("data/cases.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/decisions.csv"))
    args = ap.parse_args(argv)
    table = reclassify(pd.read_csv(args.cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} cases, plan changed in {100 * table.changed.mean():.0f}% -> {args.out}")
    print(summarize(table).round(1).to_string())


if __name__ == "__main__":
    main()
