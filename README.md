# OT-Shield: an optimal-transport trust score for GenAI, RAG and agent security

Code for the paper *Generative AI, Agentic AI and RAG: Emerging Security Threats and Defense Mechanisms*.

The **Wasserstein Trust Score (WTS)** treats any untrusted text unit (a user prompt, a retrieved chunk, a tool
output, an agent plan step) as a cloud of token embeddings and measures its Wasserstein-2 distance from a
benign reference: closed-form Bures–Wasserstein with location and shape terms, a sliced-W2 check, a length
adjustment, and split conformal calibration with a finite-sample false-alarm guarantee.

> **Status:** library, tests and all experiment harnesses are done; no experimental results yet.
> See [`docs/status.md`](docs/status.md). Experiments are blocked by design until the reproduction gates in
> [`reproduction.md`](reproduction.md) pass.

## Layout

| Path | What |
| --- | --- |
| `hypotheses.md` | Pre-registered hypotheses H1–H6 (frozen) + dated addenda |
| `reproduction.md` | Reproduction gates G1–G4 (must PASS before E1–E4) |
| `environment.lock.md` | Pinned benchmark commits, models, datasets |
| `src/wts/` | `measures` (W2, sliced W2, baselines), `length`, `conformal`, `guards`, `encoders`, `baselines`, `rag_baselines`, `agentdojo_hook`, `stats`, `runlog` |
| `scripts/run_e1_detection.py` | E1 prompt-channel detection, calibration study (RQ1, RQ4) |
| `scripts/run_e2_rag.py` | E2 PoisonedRAG / jamming with defenses (RQ2) |
| `scripts/run_e3_agentdojo.py`, `run_e3_injecagent.py` | E3 agents inside each benchmark's own harness (RQ3) |
| `scripts/run_e4_adaptive.py` | E4 dilution, paraphrase, GCG with WTS in the loss (RQ5) |
| `scripts/bench_latency.py` | E5 overhead (RQ6) |
| `scripts/make_tables.py`, `make_figures.py` | the only way numbers reach `paper/`; also writes `results/verdicts.md` |
| `scripts/verify_refs.py` | checks all 35 DOIs in `paper/refs.bib` against Crossref / DataCite |
| `paper/` | manuscript skeleton (`main.tex`) and bibliography |

## Quick start

```bash
make install && make benchmarks
make test      # 43 unit tests, CPU
make smoke     # every pipeline end to end with a test-only encoder, CPU, no downloads -> results/_smoke/
```

GPU machine (one A100/H100-class GPU, vLLM for the open models):

```bash
make install-gpu
# fill reproduction.md (published values from the cited tables), run gates G1-G4, then:
make gate-check
make e1 e2 e3 e4 latency
make tables figures refs
```

## Genuineness rules

Numbers in the paper come only from `results/*.csv` via the generators; smoke results and the test-only
encoder are refused by every results path. Every run writes a JSON record (config, seed, git commit,
versions). Negative results are reported. See Section 6 of the research plan.
