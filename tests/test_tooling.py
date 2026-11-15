"""Offline tests of the paper tooling and of E2 control flow (no network, no GPU)."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def test_bib_has_35_entries_all_with_doi_and_year_2023_on():
    vr = _load("verify_refs")
    entries = vr.parse_bib((REPO / "paper/refs.bib").read_text())
    assert len(entries) == 35
    assert all(e.get("doi") for e in entries)
    assert all(int(e["year"]) >= 2023 for e in entries)
    assert len({e["doi"].lower() for e in entries}) == 35


def test_verify_refs_logic_with_fake_registry():
    vr = _load("verify_refs")
    entries = vr.parse_bib((REPO / "paper/refs.bib").read_text())
    titles = {e["doi"]: e["title"] for e in entries}
    ok, rows = vr.check(entries, 35, lookup=lambda d: ("crossref", vr.norm(titles[d]).upper()))
    assert ok, [r for r in rows if r[3] != "OK"]
    bad = dict(titles)
    bad[entries[0]["doi"]] = "A completely different paper"
    ok, rows = vr.check(entries, 35, lookup=lambda d: ("crossref", bad[d]))
    assert not ok and rows[0][3] == "MISMATCH"


def test_gate_check_blocks_until_all_gates_pass(tmp_path):
    gc = _load("gate_check")
    status = gc.gate_status((REPO / "reproduction.md").read_text())
    assert set(status) == {"G1", "G2", "G3", "G4"} and not gc.all_pass(status)
    passed = (REPO / "reproduction.md").read_text().replace("| NOT RUN |", "| PASS |")
    assert gc.all_pass(gc.gate_status(passed))


def test_e2_attack_variants():
    e2 = _load("run_e2_rag")
    t = dict(id="q1", question="who wrote x?", adv_texts=["a b c d", "e f g h", "i", "j", "k"])
    cfg = dict(adv_per_query=5)
    filler = ["benign words here and there"] * 10
    assert e2.attack_texts("clean", t, cfg, filler) == []
    assert len(e2.attack_texts("poisonedrag_n2", t, cfg, filler)) == 2
    d = e2.attack_texts("poisonedrag_dilute_2x", t, cfg, filler)
    assert len(d[0].split()) == 4 + 8 and d[0].startswith("a b c d")
    assert "I don't know" in e2.attack_texts("jamming_explicit", t, cfg, filler)[0]
    assert e2.clean_str(" Paris. ") == "paris"


def test_memory_guard_stops_a_run_over_the_ceiling():
    import subprocess
    code = ("import time, sys; sys.path.insert(0, 'src'); from wts.hw import MemoryGuard\n"
            "MemoryGuard(max_gb=0.01, interval=0.1).start()\n"
            "try:\n    time.sleep(10); print('NOT STOPPED')\n"
            "except KeyboardInterrupt:\n    print('STOPPED')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=30).stdout
    assert "STOPPED" in out and "NOT STOPPED" not in out
