"""Dataset loaders for E1-E4. Every loader returns plain Python records and records its source revision
so run logs can cite it. Hugging Face datasets are loaded with an explicit revision when the config pins one.

Gated data (LMSYS-Chat-1M) is never written to the repository: only row ids are saved.
"""
from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "ext"


def _hf(name, split, revision=None, config=None):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, revision=revision)


# ----------------------------------------------------------------------------------------- E1 benign
def alpaca_prompts(n: int, seed: int, revision=None) -> list[dict]:
    ds = _hf("tatsu-lab/alpaca", "train", revision)
    rows = [dict(id=f"alpaca-{i}", text=(r["instruction"] + ("\n" + r["input"] if r["input"] else "")).strip(),
                 source="alpaca", label=0, family="benign") for i, r in enumerate(ds)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def lmsys_prompts(n: int, seed: int, revision=None) -> list[dict]:
    """First user turn of English, non-flagged LMSYS-Chat-1M conversations (benign shift set)."""
    ds = _hf("lmsys/lmsys-chat-1m", "train", revision)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        r = ds[i]
        if r["language"] != "English" or any(m.get("flagged") for m in r["openai_moderation"]):
            continue
        text = r["conversation"][0]["content"].strip()
        if 3 <= len(text.split()) <= 400:
            rows.append(dict(id=f"lmsys-{r['conversation_id']}", text=text, source="lmsys", label=0, family="benign"))
        if len(rows) >= n:
            break
    return rows


# ----------------------------------------------------------------------------------------- E1 attacks
def deepset_injections(revision=None) -> list[dict]:
    rows = []
    for split in ("train", "test"):
        for i, r in enumerate(_hf("deepset/prompt-injections", split, revision)):
            if r["label"] == 1:
                rows.append(dict(id=f"deepset-{split}-{i}", text=r["text"], source="deepset", label=1,
                                 family="prompt_injection"))
    return rows


def in_the_wild_jailbreaks(n: int, seed: int, config="jailbreak_2023_12_25", revision=None) -> list[dict]:
    ds = _hf("TrustAIRLab/in-the-wild-jailbreak-prompts", "train", revision, config)
    rows = [dict(id=f"itw-{i}", text=r["prompt"], source="in_the_wild", label=1, family="jailbreak_template")
            for i, r in enumerate(ds)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def harmbench_test_cases(path: str | Path, method: str) -> list[dict]:
    """Test cases produced by HarmBench's pipeline (step 1: generate_test_cases) for one attack method,
    e.g. results/GCG/<model>/test_cases/test_cases.json  ->  {behavior_id: [test_case, ...]}"""
    data = json.loads(Path(path).read_text())
    rows = []
    for bid, cases in data.items():
        for j, c in enumerate(cases):
            rows.append(dict(id=f"hb-{method}-{bid}-{j}", text=c, source=f"harmbench-{method}", label=1,
                             family=f"jailbreak_{method.lower()}", behavior_id=bid))
    return rows


def harmbench_classifier_prompt() -> str:
    """HarmBench's LLAMA2_CLS_PROMPT['prompt'], read from eval_utils.py without importing it (that module imports
    spacy and datasketch, which the prompt does not need). The string is used exactly as released."""
    import ast
    src = (EXT / "HarmBench/eval_utils.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LLAMA2_CLS_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)["prompt"]
    raise RuntimeError("LLAMA2_CLS_PROMPT not found in HarmBench eval_utils.py")


def harmbench_direct_requests() -> list[dict]:
    """HarmBench standard text behaviors as plain requests (no attack wrapper)."""
    path = EXT / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["FunctionalCategory"] == "standard":
                rows.append(dict(id=f"hb-direct-{r['BehaviorID']}", text=r["Behavior"], source="harmbench-direct",
                                 label=1, family="harmful_direct", behavior_id=r["BehaviorID"]))
    return rows


# ----------------------------------------------------------------------------------------- E2 RAG
def poisonedrag_targets(dataset: str) -> list[dict]:
    p = EXT / f"PoisonedRAG/results/adv_targeted_results/{dataset}.json"
    out = []
    for v in json.loads(p.read_text()).values():
        adv = v["adv_texts"]
        if isinstance(adv, str):
            adv = ast.literal_eval(adv)
        out.append(dict(id=v["id"], question=v["question"], correct=v["correct answer"],
                        incorrect=v["incorrect answer"], adv_texts=list(adv)))
    return out


def poisonedrag_attack_texts(target: dict, n: int = 5) -> list[str]:
    """PoisonedRAG black-box ('LM_targeted') passages exactly as the benchmark injects them: the target question,
    a period, then the released generated text (ext/PoisonedRAG/src/attack.py, get_attack, lines 91-93). The
    question prefix is what makes the passages retrievable; the released texts alone rank far lower."""
    return [target["question"] + "." + a for a in target["adv_texts"][:n]]


def poisonedrag_retrieval(dataset: str, retriever: str = "contriever") -> dict:
    """Precomputed BEIR top-100 retrieval released with PoisonedRAG: {query_id: {doc_id: score}}."""
    p = EXT / f"PoisonedRAG/results/beir_results/{dataset}-{retriever}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run PoisonedRAG's evaluate_beir.py for retriever '{retriever}'")
    return json.loads(p.read_text())


def beir_corpus(dataset: str, data_dir: str | Path = REPO / "data/raw/beir"):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    split = "train" if dataset == "msmarco" else "test"
    path = Path(data_dir) / dataset
    if not path.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        util.download_and_unzip(url, str(data_dir))
    return GenericDataLoader(str(path)).load(split=split)  # corpus, queries, qrels


# ----------------------------------------------------------------------------------------- E3 InjecAgent
def injecagent_cases(split: str, setting: str = "base") -> list[dict]:
    """split in {'dh', 'ds'}; setting in {'base', 'enhanced'}."""
    p = EXT / f"InjecAgent/data/test_cases_{split}_{setting}.json"
    return json.loads(p.read_text())


def injecagent_benign_response(case: dict, filler: str) -> str:
    """Benign counterpart of an attacked tool response: the same template with the attacker slot filled by
    benign text drawn from the benign pool (the dataset ships no benign slot content)."""
    return case["Tool Response Template"].replace("<Attacker Instruction>", filler)
