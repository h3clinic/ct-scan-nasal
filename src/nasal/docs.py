"""Keeps generated sections of README.md in sync with the code.

Run `python -m nasal.docs` to rewrite them; tests fail when they are stale.
"""

import re
from pathlib import Path

from .params import PARAM_SPECS

README = Path(__file__).resolve().parents[2] / "README.md"


def params_table() -> str:
    lines = ["| Parameter | Range | Unit | Meaning |", "| --- | --- | --- | --- |"]
    for name, s in PARAM_SPECS.items():
        lines.append(f"| `{name}` | {s.low:g} to {s.high:g} | {s.unit} | {s.description} |")
    return "\n".join(lines)


SECTIONS = {"params": params_table}


def render(text: str) -> str:
    for key, build in SECTIONS.items():
        pattern = re.compile(
            rf"(<!-- {key}:start -->\n).*?(<!-- {key}:end -->)", re.DOTALL
        )
        if not pattern.search(text):
            raise ValueError(f"README is missing the {key} markers")
        text = pattern.sub(lambda m: m.group(1) + build() + "\n" + m.group(2), text)
    return text


def main() -> None:
    text = README.read_text(encoding="utf-8")
    README.write_text(render(text), encoding="utf-8")


if __name__ == "__main__":
    main()
