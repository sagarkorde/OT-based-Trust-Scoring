# Status — 2026-11-15

Everything the plan specifies that can be built and checked without a GPU or model downloads is done and
tested. **No experimental result exists yet.** The next step is Phase 2 (reproduction gates) on a GPU
machine; `make e1`..`make e4` refuse to run until `reproduction.md` shows all four gates PASS.

## Phase gates

| Phase | Deliverable | Gate | Status |
| --- | --- | --- | --- |
| 0 | Repo, pinned benchmarks (`environment.lock.md`), Makefile | `make smoke` runs end to end | PASS (CPU, test-only encoder) |
| 1 | `hypotheses.md`, `reproduction.md` committed before any WTS code | committed | PASS — commit 99c0c82; addenda A1, A2 appended before any real data |
| 2 | Reproduce G1–G4 within ±5 points | `make gate-check` | NOT RUN — needs GPU, Hugging Face, BEIR, vLLM |
| 3 | WTS library + unit tests | tests pass; latency file written | PASS — 43 tests; CPU latency in `results/_smoke/e5_cpu.csv` (not the H6 result) |
| 4 | E1–E3 harnesses | CSVs + tables with CIs | Harnesses written; E1, E3 (AgentDojo ground truth, InjecAgent detection) smoke-tested on real benchmark text; E2 and agent runs need GPU |
| 5 | E4 adaptive attacks | Table 5, Figure 6 | Dilution smoke-tested; paraphrase and GCG need GPU |
| 6 | Manuscript | traceable numbers, refs pass | Skeleton with threat model, method, propositions and proofs; compiles (tested with `article` class; install `elsarticle`) |
| 7 | Independent re-run, Zenodo | — | NOT STARTED |

## Why Phase 2 did not run here

The development workspace has 2 CPU cores, 7 GB RAM and no GPU, and its network allowlist blocks Hugging
Face, the BEIR download server, LLM APIs and the DOI registries. Running any model or dataset there would
have meant inventing numbers, which the plan forbids.

## Decisions made during implementation (all before any real data)

1. **Length confound (addendum A1).** Every distance — ours and the baselines' — falls with unit length at
   fixed content (`tests/test_length.py`). All scores now also have a length-adjusted form; guards decide
   on it. Without this, PoisonedRAG's short adversarial passages would be "detected" for being short.
2. **AgentDojo calibration pooled across suites (addendum A2)**, because one suite gives fewer than the 39
   calibration outputs a conformal threshold needs at alpha/2.
3. **Per-tool models only when they can flag**: a tool needs >= min_n_for_alpha(alpha/n_tests) calibration
   outputs, else the pooled model is used (found when InjecAgent smoke runs could never flag).
4. **Injected-output labels in AgentDojo** are matched on whitespace-normalised injection lines, because
   AgentDojo's tool-result formatting collapses blank lines.
5. **InjecAgent benign responses**: the dataset ships no benign slot content, so benign responses are the
   same templates with the attacker slot filled by Alpaca text; reference/calibration fillers are disjoint
   from benign-test fillers.
6. **E1 split** is ref 500 / calibration pool 1,000 / test 500 per seed so the RQ4 study can draw n_cal up
   to 1,000. PCA for the LLM-layer encoder is fit on 1,000 extra Alpaca rows never used elsewhere.
7. **Agent guard** flags when min(p_output, p_drift) <= alpha/2 (Bonferroni), keeping per-step FPR <= alpha.

## Re-implementations that must be validated before their numbers are reported

- RobustRAG keyword aggregation (`src/wts/rag_baselines.py`): compare with the authors' code on 20 questions.
- PIShield-style probe (`src/wts/baselines.py`): layer chosen on validation; compare with the released code if available.
- Paraphrase defense re-ranks the top-100 + adversarial passages (no full-corpus re-retrieval); state this in the paper.
- Jamming uses an explicit-instruction blocker; Shafran et al.'s optimized blockers can be plugged in via `jamming_optimized_path`.

## Observations from the development machine (not results)

- Gaussian W2 at d = 768 takes ~95 ms per unit on 2 CPU cores (`numpy`); the eigendecomposition dominates.
  H6 is defined on GPU; `bench_latency.py --device cuda` times the torch path, whose Ledoit-Wolf matches
  scikit-learn to 1e-10 (tested).
- Exploratory idea, not pre-registered: AgentDojo tool outputs often bundle many records with the injection
  in one; a windowed max score could help. Report only as exploratory if tried.

## Before Phase 2 on the GPU machine

1. `make install-gpu && make benchmarks && make test && make smoke`
2. Record model and dataset revisions in `environment.lock.md`; `pip freeze > docs/pip-freeze-gpu.txt`.
3. Fill the "Published" column of `reproduction.md` from the cited tables (never from memory).
4. Serve LLMs with vLLM on ports 8000/8001/8002 (see configs); run G1–G4; log every attempt.
5. `make gate-check`, then `make e1 e2 e3 e4 latency tables figures refs`.
