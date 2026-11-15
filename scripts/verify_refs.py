"""Verify every DOI in a .bib file against the DOI registries (genuineness rule 5).

For each entry: GET https://api.crossref.org/works/<doi>; on 404 GET https://api.datacite.org/dois/<doi>.
Compare the registry title with the .bib title after lower-casing and stripping punctuation/LaTeX.
Also checks: exactly the expected number of entries (default 35) and year >= 2023.
Exit code 1 on any mismatch, unresolved DOI, count error or pre-2023 year.

  python scripts/verify_refs.py paper/refs.bib
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

UA = {"User-Agent": "ot-shield-verify-refs/1.0 (mailto:ot-shield@example.org)"}


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        body = m.group(3)
        fields = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body)}
        entries.append(dict(key=m.group(2).strip(), type=m.group(1), **fields))
    return entries


def norm(t: str) -> str:
    t = re.sub(r"\\[a-zA-Z]+\s*|[{}\\`'^\"~]", "", t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def registry_title(doi: str) -> tuple[str, str | None]:
    try:
        msg = _get(f"https://api.crossref.org/works/{doi}")["message"]
        return "crossref", (msg.get("title") or [None])[0]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    try:
        attrs = _get(f"https://api.datacite.org/dois/{doi}")["data"]["attributes"]
        return "datacite", (attrs.get("titles") or [{}])[0].get("title")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "unresolved", None
        raise


def check(entries, expected: int, lookup=registry_title):
    rows, ok = [], len(entries) == expected
    for e in entries:
        doi, title, year = e.get("doi", ""), e.get("title", ""), int(e.get("year", "0") or 0)
        reg, rtitle = lookup(doi) if doi else ("no-doi", None)
        match = rtitle is not None and (norm(rtitle) == norm(title) or norm(rtitle).startswith(norm(title))
                                        or norm(title).startswith(norm(rtitle)))
        good = match and year >= 2023
        ok &= good
        rows.append((e["key"], doi, reg, "OK" if good else ("YEAR<2023" if match else "MISMATCH"), rtitle or ""))
        time.sleep(0.2)
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--expected", type=int, default=35)
    a = ap.parse_args()
    entries = parse_bib(open(a.bib).read())
    ok, rows = check(entries, a.expected)
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print(f"\n{len(entries)} entries (expected {a.expected}); {'ALL VERIFIED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
