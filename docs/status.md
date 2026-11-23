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

## Laptop profile (HP Omen: i9-13th gen, 64 GB RAM, RTX 4060 8 GB, Windows 11) — added 2026-11-15

`scripts/windows/setup.ps1` installs everything; `scripts/windows/run.ps1 <task>` runs a task with
`--profile laptop` (configs in `configs/laptop/`). Deviations from the plan's hardware, all recorded per run
in the JSON log (`hardware` field):

| Item | Plan | Laptop profile | Why |
| --- | --- | --- | --- |
| Open LLMs | bf16 via vLLM | Ollama Q4 (llama3.1:8b, qwen2.5:7b; llama2:7b-chat and mistral:7b-instruct-v0.2 for gates G1, G4) | 8 GB VRAM; vLLM has no Windows build |
| Context | model default | fixed tags: 16k (agents), 4k (RAG, short prompts); flash attention + q8 KV cache | Ollama's default context silently truncates AgentDojo prompts |
| White-box encoder, Llama Guard, probe, GCG | bf16 | 4-bit NF4 (bitsandbytes) | 8 GB VRAM |
| HarmBench classifier (13B) | GPU bf16 | CPU bf16 (~26 GB RAM), unquantized | keeps the official judge exact |
| RAM | — | hard ceiling 50 GB per run (MemoryGuard) | user's limit |
| Scope | 3 LLMs, 5 seeds | E1 5 seeds; E2 2 LLMs x 3 seeds; E3 1 LLM x 3 seeds; E4 3 seeds, GCG 15 behaviours x lambda {0, 1} | plan Section 8 drop order (never below 3 seeds; E4 and RQ4 kept) |
| HarmBench GCG/PAIR/AutoDAN test cases in E1 | yes | no; GCG detection measured in E4 | test-case generation needs GPU-days on large GPUs |
| E3 agent runs | every (defense, seed, case) run | replay (2026-11-19): seed-free baselines shared by seeds; WTS cases copy the no-defense result when no guard element would act (checked with the elements' own code on the saved trajectory), otherwise run; a random sample of copies is rerun for real (`*_replay_check.csv`) | ~8–9 days of agent runs on this laptop; exact under temperature-0 determinism, which the check measures |

Gates on this hardware: G1 (PoisonedRAG, LLaMA-2-7B: 0.97, Table 1), G3 (InjecAgent, prompted Mistral-7B: 16.7 %,
Table 3) and G4 (HarmBench DirectRequest, Mistral 7B: 46.0 %, Table 6) have published comparators that fit 8 GB.
AgentDojo reports no such model, so G2 is measured for the record and can pass only as an explained gap signed off
by the authors, unless an API model (e.g. gpt-4o-mini with a key) is used for it.

## E3 observations (recorded 2026-11-19, from results/e3; not yet in the manuscript)

- AgentDojo ground-truth tool outputs (benchmark v1.2.2, BGE, 3 seeds): the guard was calibrated on 188 benign
  tool outputs over ~40 tool types; no tool reached the per-tool minimum (39 calibration outputs at alpha/2), so
  every output was scored against the pooled `__all__` reference with the length-adjusted Gaussian W2. Pooled
  AUROC (seed 0) 0.41; per (suite, tool) with both classes, injected outputs score lower than their clean versions
  in 8 of 14 groups. Not a labelling bug: positives are outputs that contain the injection (line match), negatives
  are the same tools' outputs in clean runs. Likely mechanism: against a pooled reference of mixed tool formats,
  appended natural-language instructions make an output more typical, and the length adjustment lowers the score of
  longer outputs. Reported as a negative result for the tool-output part of H3; any per-tool or unadjusted variant
  would be exploratory (not pre-registered) and must be labelled so.
- InjecAgent (base, seed 0): AUROC 0.70, TPR 0.16 at FPR 0.056 (alpha 0.05).
- InjecAgent agents finished 2026-11-19 23:21 (12,648 rows: 1 LLM x 3 seeds x 2 settings x 2 defenses x 1,054
  cases; no duplicates). Replay: WTS cases run for real 69-176 per (seed, setting), the rest copied. Detection
  AUROC by seed (base / enhanced): 0.70 / 0.54, 0.72 / 0.60, 0.59 / 0.38.
- Determinism check (InjecAgent, 30 reruns per setting): base 30/30, enhanced 28/30. Both mismatches: recorded
  "invalid", rerun "succ" (dh 208, ds 312). So Ollama at temperature 0 is not perfectly repeatable: about 3 % of
  reruns (2/60) change outcome. A copied WTS row therefore equals one realisation of the no-defense run rather than
  an independent rerun; the paper reports the 58/60 figure and treats run-to-run variation as a limitation of the
  replay procedure (and of every temperature-0 result on this stack).
- AgentDojo PI-detector baseline (2026-11-20): AgentDojo's DeBERTa detector classifies whole tool outputs without
  truncation (up to ~8k tokens). On the 8 GB RTX 4060 this spilled into shared system memory and pushed Llama out
  of VRAM: workspace user_task_26 took 9 s without the detector and 144 s with it (same 3 steps), and three runs of
  user_task_38 took 25-52 min. From 2026-11-20 the detector runs on the CPU (`pi_detector_device: cpu`); code, model
  and threshold unchanged, decisions identical up to floating-point differences near the 0.5 threshold. Seed-0
  workspace PI-detector runs before the switch were computed on the GPU and are kept.
- Context limit: Ollama runs Llama with a 16k-token context and silently drops the oldest tokens when a trajectory
  grows beyond it (seen in a 15-step loop with ~30k-character tool outputs); AgentDojo's API backends would raise
  context_length_exceeded instead. Affects only very long trajectories, identically for every defense; reported as
  a limitation.
- Devices (authors asked on 2026-11-20 for everything on the GPU): the E3 WTS encoder (BGE) moved from CPU to GPU
  for runs started after commit of this note; InjecAgent and the AgentDojo ground-truth detection used CPU
  embeddings. The embedding cache stores fp16, so CPU- and GPU-computed embeddings agree to the stored precision.
  Still on the CPU: the HarmBench Llama-2-13B classifier (G4/E4 judge; 26 GB in bf16 cannot fit 8 GB, and a
  quantised judge would no longer be the classifier validated in G4) and AgentDojo's DeBERTa PI detector (authors'
  choice 2026-11-20: CPU, same decisions at normal speed; GPU spilled into system memory, see above).
- Request timeouts (2026-11-21): some AgentDojo runs took exactly ~30 min 11 s per run (e.g. all PI-detector runs of
  workspace user_task_21): Ollama's default reply length is unlimited, a degenerate reply never ended, the 30-min
  client timeout fired, and AgentDojo's LocalLLM turned the error into an empty reply (task failed). Fix: the agent
  model's Ollama tag stores a reply cap `num_predict: 4096` (tool-call replies are a few hundred tokens; vLLM, used
  by the comparator, also stops generation at the context limit rather than never), and E3 moves logs of runs that
  hit the timeout to `agentdojo_logs/_retried/` at start-up, reopening the affected blocks (and the WTS blocks
  replayed from an affected no-defense log) so exactly those runs are redone with the cap. Runs whose replies
  ended on their own are unaffected.


### AgentDojo replay determinism check (finished 2026-11-21 22:51)
From results/e3/done/agentdojo_replay_check_ots-llama3.1-8b-16k.ok: n = 30 re-runs of logged cases;
same outcome (utility / attack success) in 29/30; identical message trajectory in only 4/30.
Outcomes are stable under replay; exact wording is not (Ollama temperature-0 nondeterminism). To be reported in the paper.

### E3 AgentDojo complete (2026-11-22 18:18)
results/e3/agentdojo_items.csv: 9,468 rows = 3 seeds x 4 suites x 6 defenses x (benign + important_instructions), no duplicates.
Preliminary (mean over 3 seeds, %; full tables to be generated by the analysis script):
benign utility: none 21.1, spotlighting 29.9, PI detector 20.4, wts_drift_confirm 21.1, wts_full 19.7, wts_output_block 19.7.
under attack, utility / ASR: none 17.7/2.4, spotlighting 19.6/4.6, PI detector 13.8/1.5,
wts_drift_confirm 17.0/2.2, wts_full 17.7/2.3, wts_output_block 17.7/2.4.
NEGATIVE RESULT: the WTS agent guard does not reduce ASR measurably on AgentDojo with Llama-3.1-8B
(97-100 % of attacked cases unflagged, so replayed unchanged); consistent with AUROC < 0.5 on the GT tool outputs.
The undefended ASR is already low (2.4 %), which limits headroom for any defense. To be reported prominently.

### E1 Llama layer encoder: VRAM overflow fixed (2026-11-22)
First run: the second encoding pass (LMSYS reference texts, long) slowed to ~4.5 h estimated and crashed with
CUBLAS_STATUS_INTERNAL_ERROR. Cause: the full forward computed batch x 1024 x 128k-vocab logits and kept all 33 hidden
states, overflowing the 8 GB GPU. Fix: stop the forward at block 16 (hook), no LM head. Output values are identical
(tests/test_llm_layer_encoder.py checks against the full forward). Already cached encodings are kept.
PIShield probe: features now from the base model without the LM head (same hidden states).

### E1 Llama Guard baseline: CUDA out of memory at 4544/5463 (2026-11-23)
The Llama encoder finished all 5 seeds (15/15 encoder x seed runs). Llama Guard then ran out of GPU memory on a long
(untruncated) input: the full forward built a length x 128k-vocab logits tensor plus a KV cache. Fix: logits of the
last position only (logits_to_keep=1, use_cache=False); identical values (tested). Inputs stay untruncated, as for the
4,544 items already scored (journaled, kept). Because this pass was interrupted, it records no Llama Guard timing row;
judge latency comes from the latency step instead.
