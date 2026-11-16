"""Offline tests of the paper tooling and of E2 control flow (no network, no GPU)."""
import re
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
    assert status["G1"] == "PASS" and status["G4"] == "PASS"                 # bold markdown is read as PASS
    md = (REPO / "reproduction.md").read_text()
    passed = re.sub(r"\| (NOT RUN|PENDING[^|]*) \|", "| PASS |", md)
    assert gc.all_pass(gc.gate_status(passed))


def test_e2_attack_variants():
    e2 = _load("run_e2_rag")
    t = dict(id="q1", question="q", adv_texts=["a b c d", "e f g h", "i", "j", "k"])
    cfg = dict(adv_per_query=5)
    filler = ["benign words here and there"] * 10
    assert e2.attack_texts("clean", t, cfg, filler) == []
    assert len(e2.attack_texts("poisonedrag_n2", t, cfg, filler)) == 2
    d = e2.attack_texts("poisonedrag_dilute_2x", t, cfg, filler)
    assert d[0].startswith("q.a b c d") and len(d[0].split()) == 4 + 8
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


def test_ensure_ollama_tag_pulls_and_creates_against_a_fake_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    sys.path.insert(0, str(REPO / "src"))
    from wts.hw import ensure_ollama_tag
    calls, tags = [], {"models": [{"name": "llama3.1:8b"}]}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps(tags).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            calls.append((self.path, json.loads(self.rfile.read(n))))
            self.send_response(200); self.end_headers()
            if self.path == "/api/show":   # what Ollama returns for mistral:7b-instruct-v0.2
                self.wfile.write(json.dumps({"template": "[INST] {{ .Prompt }} [/INST]",
                                             "parameters": 'stop                           "[INST]"\n'
                                                           'stop                           "[/INST]"'}).encode())
            else:
                self.wfile.write(b'{"status":"success"}\n')

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    assert ensure_ollama_tag("ots-a", "llama3.1:8b", 4096, url) is True       # base present: create only
    assert [c[0] for c in calls] == ["/api/create"] and calls[0][1]["parameters"]["num_ctx"] == 4096
    calls.clear()
    assert ensure_ollama_tag("ots-b", "mistral:7b", 8192, url) is True        # base missing: pull then create
    assert [c[0] for c in calls] == ["/api/pull", "/api/create"]
    tags["models"].append({"name": "ots-c:latest"})
    calls.clear()
    assert ensure_ollama_tag("ots-c", "llama3.1:8b", 4096, url) is False      # already there: nothing
    assert calls == []
    calls.clear()                                                              # GGUF without a chat template
    assert ensure_ollama_tag("ots-d", "hf.co/x/y-GGUF:Q4_K_M", 8192, url, template_from="llama3.1:8b") is True
    assert [c[0] for c in calls] == ["/api/pull", "/api/show", "/api/create"]
    created = calls[-1][1]
    assert created["template"] == "[INST] {{ .Prompt }} [/INST]"
    assert created["parameters"] == {"num_ctx": 8192, "temperature": 0, "stop": ["[INST]", "[/INST]"]}
    srv.shutdown()


def test_poisonedrag_passages_carry_the_question_prefix_like_the_benchmark():
    sys.path.insert(0, str(REPO / "src"))
    from wts import data as D
    t = D.poisonedrag_targets("nq")[0]
    adv = D.poisonedrag_attack_texts(t, 5)
    assert len(adv) == 5 and all(a.startswith(t["question"] + ".") for a in adv)
    assert adv[0] == t["question"] + "." + t["adv_texts"][0]   # = PoisonedRAG src/attack.py get_attack
    e2 = _load("run_e2_rag")
    assert e2.attack_texts("poisonedrag_blackbox", dict(t), dict(adv_per_query=5), []) == adv
