"""Resumability: journals, results-CSV helpers, AgentDojo log reuse, and an interrupted gate that restarts."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wts.resume import Journal, Progress, done_keys, purge_rows, tag_of  # noqa: E402
from wts.runlog import append_rows  # noqa: E402


def test_journal_survives_restart_and_ignores_a_torn_line(tmp_path):
    p = tmp_path / "j.jsonl"
    J = Journal(p)
    J.add("a", {"x": 1})
    J.add("b", {"x": 2})
    with p.open("a") as f:
        f.write('{"_key": "c", "x": ')          # crash in the middle of writing item c
    J2 = Journal(p)
    assert "a" in J2 and "b" in J2 and "c" not in J2 and J2.get("b")["x"] == 2


def test_tag_changes_with_config_or_version():
    assert tag_of({"a": 1}, 1) == tag_of({"a": 1}, 1)
    assert tag_of({"a": 1}, 1) != tag_of({"a": 1}, 2) != tag_of({"a": 2}, 2)


def test_done_keys_and_purge(tmp_path):
    p = tmp_path / "r.csv"
    append_rows(p, [dict(seed=0, q="x", v=1), dict(seed=0, q="y", v=2), dict(seed=1, q="x", v=3)])
    assert done_keys(p, ["seed", "q"]) == {("0", "x"), ("0", "y"), ("1", "x")}
    assert purge_rows(p, {"seed": 0}) == 2
    assert done_keys(p, ["seed", "q"]) == {("1", "x")}


def test_progress_line_format(capsys):
    P = Progress("demo", 4, resumed=1, every_s=0)
    P.step(); P.step(); P.step()
    out = capsys.readouterr().out
    assert "resuming: 1/4" in out and "[demo] 4/4 (100%)" in out and "left" in out


def test_agentdojo_resume_reloads_finished_tasks_and_calibration_reads_logs(tmp_path):
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.benchmark import benchmark_suite_without_injections
    from agentdojo.functions_runtime import EmptyEnv, FunctionCall
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.types import ChatAssistantMessage, text_content_block_from_string

    import run_e3_agentdojo as E3
    from wts.agentdojo_hook import QuietLogger, build_pipeline

    suite = get_suite("v1.2.2", "banking")
    tool = next(f for f in suite.tools if not f.parameters.model_json_schema().get("required"))

    class FakeLLM(BasePipelineElement):
        """Calls one argument-free tool, then answers. Counts calls so we can see what a resume skips."""
        name = "local-fake"
        calls = 0

        def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
            FakeLLM.calls += 1
            if messages[-1]["role"] == "tool":
                msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Done.")], tool_calls=None)
            else:
                msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Checking.")],
                                           tool_calls=[FunctionCall(function=tool.name, args={}, id="c1")])
            return query, runtime, env, [*messages, msg], extra_args

    pe = E3.ProgressElement()
    pipe = build_pipeline(FakeLLM(), "local-fake-calib-seed0", first=[pe])
    tasks = sorted(suite.user_tasks)[:3]
    pe.reset("first run", len(tasks), 0)
    with QuietLogger(tmp_path):
        benchmark_suite_without_injections(pipe, suite, tmp_path, False, user_tasks=tasks, benchmark_version="v1.2.2")
    first_calls = FakeLLM.calls
    assert first_calls >= 6 and E3.count_logged(tmp_path, pipe.name, "banking", "none") == 3
    # a task interrupted by an API error leaves a log without utility/security: not counted, not harvested
    broken = tmp_path / pipe.name / "banking" / "user_task_99" / "none" / "none.json"
    broken.parent.mkdir(parents=True)
    broken.write_text('{"suite_name": "banking", "messages": [], "error": null, "duration": 0.1}')
    assert E3.count_logged(tmp_path, pipe.name, "banking", "none") == 3
    assert E3.harvest_calibration(tmp_path, pipe.name, "banking", ["user_task_99"]) == ({}, [])

    pe.reset("second run", len(tasks), E3.count_logged(tmp_path, pipe.name, "banking", "none"))
    with QuietLogger(tmp_path):
        res = benchmark_suite_without_injections(pipe, suite, tmp_path, False, user_tasks=tasks, benchmark_version="v1.2.2")
    assert FakeLLM.calls == first_calls                  # nothing re-run: every task reloaded from the logs
    assert len(res["utility_results"]) == 3

    outputs, pairs = E3.harvest_calibration(tmp_path, pipe.name, "banking", tasks)
    assert sum(len(v) for v in outputs.values()) == 3 and tool.name in outputs
    assert len(pairs) == 3 and all("CALL " + tool.name in a for a, _ in pairs)

    # attack run: AgentDojo first runs every injection task benignly, then each (user task, injection task) pair
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import benchmark_suite_with_injections
    attack = load_attack("direct", suite, pipe)
    n_inj = len(suite.injection_tasks)
    E3.reset_for_attack(pe, "attack", tmp_path, pipe.name, suite, attack, "direct", 2)
    assert pe.p.total == n_inj + 2 * n_inj and pe.p.resumed == 0
    with QuietLogger(tmp_path):
        benchmark_suite_with_injections(pipe, suite, attack, tmp_path, False, user_tasks=tasks[:2],
                                        benchmark_version="v1.2.2", verbose=False)
    assert pe.p.done == pe.p.total
    calls = FakeLLM.calls
    E3.reset_for_attack(pe, "attack again", tmp_path, pipe.name, suite, attack, "direct", 2)
    assert pe.p.resumed == pe.p.total
    with QuietLogger(tmp_path):
        benchmark_suite_with_injections(pipe, suite, attack, tmp_path, False, user_tasks=tasks[:2],
                                        benchmark_version="v1.2.2", verbose=False)
    assert FakeLLM.calls == calls


def test_g3_interrupted_then_resumed_finishes_without_redoing_work(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_gates", REPO / "scripts/run_gates.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    monkeypatch.setattr(G, "OUT", tmp_path)

    class Stub:
        model = "stub"
        calls = 0
        limit = None

        def chat(self, messages, **kw):
            Stub.calls += 1
            if Stub.limit is not None and Stub.calls > Stub.limit:
                raise KeyboardInterrupt            # simulate Ctrl+C / reboot part way through
            user = messages[-1]["content"]
            m = re.search(r"tools available to you are: \[([^\]]+)\]", user)
            if not m:
                return '```json\n{"result": "simulated"}\n```'
            tools = re.findall(r"'([^']+)'", m.group(1))
            step2 = len(tools) > 2 and f"Action: {tools[1]}" in user.split("Now begin your task!")[-1]
            return (f"Thought: The user asked for this so I will use the tool now.\n"
                    f"Action: {tools[2] if step2 else tools[1]}\nAction Input: {{\"x\": \"y\"}}\n")

    monkeypatch.setattr(G, "build_llm", lambda c: Stub())
    cfg = {"setting": "base", "llm": {"model": "stub"}}
    Stub.limit = 300
    with pytest.raises(KeyboardInterrupt):
        G.g3(cfg)
    partial = len(G.journal("G3", cfg))
    assert 100 < partial < 1054
    Stub.limit, Stub.calls = None, 0
    G.g3(cfg)
    assert len(G.journal("G3", cfg)) == 1054
    assert Stub.calls < 1054 + 600 - partial          # only the remaining cases were sent to the model
    summary = [json.loads(l) for l in (tmp_path / "summary.jsonl").read_text().splitlines()]
    assert summary[-1]["asr_valid_total"] == 1.0 and summary[-1]["gate"] == "G3"
    assert len(pd.read_csv(tmp_path / "G3.csv")) == 1054


def test_append_rows_widens_header_for_optional_columns(tmp_path):
    p = tmp_path / "items.csv"
    append_rows(p, [dict(qid=1, defense="none")])
    append_rows(p, [dict(qid=2, defense="wts", n_flagged=3)])
    append_rows(p, [dict(qid=3, defense="para", paraphrase="x, y")])
    df = pd.read_csv(p, keep_default_na=False, dtype=str)
    assert list(df.columns) == ["qid", "defense", "n_flagged", "paraphrase"]
    assert df.n_flagged.tolist() == ["", "3", ""] and df.paraphrase.tolist() == ["", "", "x, y"]


def test_filter_rows_drops_partial_items(tmp_path):
    from wts.resume import filter_rows
    p = tmp_path / "dilution.csv"
    append_rows(p, [dict(seed=0, item="a", f=1), dict(seed=0, item="b", f=1), dict(seed=0, item="b", f=2)])
    assert filter_rows(p, lambda r: (r["seed"], r["item"]) in {("0", "a")}) == 2
    assert pd.read_csv(p).item.tolist() == ["a"]


def test_resumable_map_only_computes_missing_items(tmp_path):
    from wts.resume import resumable_map
    calls = []

    def fn(batch):
        calls.extend(batch)
        if "boom" in batch:
            raise KeyboardInterrupt
        return [len(x) for x in batch]
    keys = [f"k{i}" for i in range(6)]
    items = ["a", "bb", "ccc", "boom", "eeeee", "f"]
    with pytest.raises(KeyboardInterrupt):
        resumable_map("t", keys, items, fn, tmp_path / "j.jsonl", chunk=2)
    items[3] = "dddd"
    calls.clear()
    assert resumable_map("t", keys, items, fn, tmp_path / "j.jsonl", chunk=2) == [1, 2, 3, 4, 5, 1]
    assert calls == ["ccc", "dddd", "eeeee", "f"]   # chunk [a, bb] finished before the crash and was kept


def test_encode_cache_recovers_from_a_torn_file(tmp_path):
    import numpy as np

    from wts.cache import _key, encode_cached
    from wts.encoders import HashingEncoder
    enc = HashingEncoder(dim=16)
    texts = ["one two three", "four five six seven"]
    a = encode_cached(enc, texts, cache_dir=tmp_path)
    f = tmp_path / enc.name.replace("/", "_") / f"{_key(texts[1])}.npy"
    f.write_bytes(f.read_bytes()[:20])          # simulate a crash in the middle of writing
    b = encode_cached(enc, texts, cache_dir=tmp_path)
    assert all(np.allclose(x, y, atol=1e-2) for x, y in zip(a, b))


def test_g3_sensitivity_gate_writes_its_own_files(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_gates", REPO / "scripts/run_gates.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    monkeypatch.setattr(G, "OUT", tmp_path)
    (tmp_path / "G3.csv").write_text("split,case,eval,sim_response\ndh,0,succ,\n")   # must not be imported
    monkeypatch.setattr(G, "build_llm", lambda c: type("L", (), {"model": "s", "chat": lambda self, m, **k: "nothing"})())
    cfg = {"setting": "base", "llm": {"model": "v01", "ollama_base": "hf.co/x:Q4_K_M"}}
    G.g3(cfg, "G3_v01")
    assert len(pd.read_csv(tmp_path / "G3_v01.csv")) == 1054 and len(G.journal("G3_v01", cfg)) == 1054
    assert (tmp_path / "G3.csv").read_text().count("\n") == 2                      # untouched
    last = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[-1])
    assert last["gate"] == "G3_v01" and last["weights"] == "hf.co/x:Q4_K_M"


def test_g3_redoes_cases_simulated_by_another_model(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_gates", REPO / "scripts/run_gates.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    monkeypatch.setattr(G, "OUT", tmp_path)
    cfg = {"setting": "base", "llm": {"model": "agent"}, "sim_llm": {"model": "fixed-sim"}}
    J = G.journal("G3_v01", {k: v for k, v in cfg.items() if k != "sim_llm"})
    J.add("ds|0", dict(split="ds", case=0, eval="succ", sim_response="local:agent"))       # simulated by the agent
    J.add("ds|1", dict(split="ds", case=1, eval="unsucc", sim_response="cache"))
    seen = []
    monkeypatch.setattr(G, "build_llm", lambda c: type("L", (), {"model": c["model"],
                                                                 "chat": lambda self, m, **k: seen.append(1) or "x"})())
    G.g3(cfg, "G3_v01")
    df = pd.read_csv(tmp_path / "G3_v01.csv")
    assert len(df) == 1054 and len(seen) == 1054 - 1      # everything but the cached case ds|1 was (re)done
    last = json.loads((tmp_path / "summary.jsonl").read_text().splitlines()[-1])
    assert last["simulator"] == "fixed-sim" and last["cases_with_repeat_abort"] == 0
