"""Refuse to start E1-E4 until every reproduction gate in reproduction.md is PASS (plan Section 6.1).
An explained gap signed off by the authors is recorded as status 'PASS (explained)'."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def gate_status(md: str) -> dict[str, str]:
    out = {}
    for line in md.splitlines():
        m = re.match(r"\|\s*(G\d)\s*\|", line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out[m.group(1)] = cells[7] if len(cells) > 7 else ""
    return out


def all_pass(status: dict[str, str]) -> bool:
    return bool(status) and all(s.upper().startswith("PASS") for s in status.values())


if __name__ == "__main__":
    st = gate_status((REPO / "reproduction.md").read_text())
    for g, s in st.items():
        print(f"{g}: {s}")
    if not all_pass(st):
        sys.exit("Reproduction gates not all PASS: experiments E1-E4 are blocked (see reproduction.md).")
    print("All gates PASS.")
