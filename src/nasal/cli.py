import argparse
from pathlib import Path

from .cases import build_table, sample_cases, write_meshes


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


if __name__ == "__main__":
    main()
