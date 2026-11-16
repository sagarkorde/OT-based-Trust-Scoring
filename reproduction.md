# Reproduction gates (Phase 2)

No WTS experiment (E1–E4) may start until every gate below is PASS or has an explained gap
signed off by the authors. Tolerance: ±5 absolute percentage points.

Rule: the "Published" column is filled by reading the cited table in the cited paper, and the
table/figure number is written next to the value. Never fill it from memory. Leave `TO FILL` until done.

| Gate | What | Setting | Published (source table) | Reproduced (mean over seeds) | Seeds | Diff | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | PoisonedRAG ASR, NQ | Black-box attack (released adversarial texts), N = 5 passages per question, Contriever, top-k = 5, dot product, 100 target questions; LLM LLaMA-2-7B (laptop: Ollama llama2:7b-chat Q4) | 0.97 (PoisonedRAG [14], Table 1, NQ black-box, LLaMa-2-7B) | 0.96 (96/100; 95 % CI 0.90-0.98) | 1 (temperature 0) | −0.01 | **PASS** | Paper uses temperature 0.1 (we use 0) and fp16 weights (we use Q4); ASR by substring match as in the paper |
| G2 | AgentDojo utility under attack and ASR, no defense | Llama 3.1 8B Instruct (Ollama Q4_K_M, local), prompt-based LocalLLM (temperature 0, top-p 0.9), benchmark v1.1.2, difficulty-1 user tasks of all 4 suites (330 cases), `important_instructions` (assumed) | Utility under attack 24.85 % (20.61-29.70), ASR 4.24 % (2.42-6.67) (Mu et al. 2025, arXiv 2502.12197, Table 3, Llama 3.1 8B Instruct "As-is"; setup from their fork normster/agentdojo @ d8477eaf) | — | — | — | NOT RUN | Attack name not stated in the paper; n = 330 inferred and confirmed (82/330 and 14/330 reproduce both percentages). They served bf16 with vLLM; we use Q4 |
| G3 | InjecAgent ASR-valid, base setting | Prompted (ReAct) agent, DH and DS, InjecAgent's own evaluator; laptop: Ollama mistral:7b-instruct-v0.2 Q4, 8k context | 16.7 % (InjecAgent [22], Table 3, prompted Mistral-7B, base setting, Total) | 16.0 % with v0.1 (127/792 valid; 95 % CI 13.6-18.8) · 22.1 % with v0.2 | 1 (temperature 0) | −0.7 (v0.1) / +5.4 (v0.2) | **PASS** (v0.1) | Check which Mistral-7B release the paper used (appendix model list) before comparing |
| G4 | HarmBench ASR, DirectRequest | 200 standard text behaviours (all splits), HarmBench Llama-2-13B classifier (bf16, CPU on laptop); model Mistral 7B Instruct v0.2 (laptop: Ollama Q4) | 46.0 % (HarmBench [10], Table 6, standard behaviours, all splits, DR, Mistral 7B); test split only: 44.7 % (Table 7) | 42.0 % (84/200; 95 % CI 35.4-48.9) | 1 (temperature 0) | −4.0 | **PASS** | HarmBench lists Mistral 7B as Mistral 7B Instruct v0.2 (Appendix C.2) |

## Sources of published values (read 2026-11-15)

- PoisonedRAG, arXiv:2402.07867 (HTML), Table 1 and Section 5 defaults: Contriever, N = 5, top-k = 5, dot product, 100 target questions, substring-match ASR, PaLM 2 default at temperature 0.1.
- HarmBench, arXiv:2402.04249 (HTML), Table 6 (all behaviours) and Table 7 (test behaviours), DirectRequest column.
- InjecAgent, arXiv:2403.02691 (HTML), Table 3, ASR-valid (%), prompted agents, base setting, Total.

## Log

Append one entry per attempt: date, gate, command, commit hash, result, explanation of any gap.

- 2026-11-15 21:15 · G1 · `run.ps1 gates G1 G3 G4` · commit 7979197 · ASR 0.76 (published 0.97), mean poisoned
  passages in top-5 = 2.4 · **FAIL**. Cause: harness bug, not the model. PoisonedRAG prepends the target question
  ("question" + ".") to each released adversarial text (src/attack.py, get_attack, lines 91-93); our harness used
  the released texts alone, so fewer reached the top-5. Fixed in the next commit (wts.data.poisonedrag_attack_texts,
  used by G1, E2 and E4). G1 to be re-run.

- 2026-11-15 23:09 · G3 · same run · Mistral-7B-Instruct-v0.2 (Ollama Q4_K_M, 8k context, temperature 0),
  InjecAgent prompt, base setting · ASR-valid Total **22.1 %** (n valid = 655 of 1,054; 95 % CI 19.0-25.3;
  DH 15.7 %, DS 29.3 %; valid rate 62.1 %) vs published 16.7 % (Table 3) · diff +5.4 · **outside tolerance by
  0.4 points**. Score recomputed independently from G3.csv with InjecAgent's get_score definitions (identical).
  Known differences: (1) the paper does not name its Mistral-7B release (repo and paper checked); (2) Q4
  quantization vs full precision; (3) 11 DS step-2 attacker responses were simulated by the local model instead of
  GPT-4-0613 (cache misses); 8 of those 11 succeeded, and counting all 8 as failures gives 20.9 % (inside
  tolerance). Temperature (0) and output budget (512) match the paper. Status: **awaiting author decision**
  (accept as explained gap, or run the sensitivity checks: Mistral-7B-Instruct-v0.1, and the 11 cases
  re-simulated another way).

- 2026-11-16 12:15 · G4 · `run.ps1 gates G4` · generations from the 2026-11-15 run (commit 7979197), judged at
  commit 672e6d5 over three sessions (resumed; judge journal has 200 unique behaviours, no duplicates) ·
  Mistral-7B-Instruct-v0.2 (Ollama Q4_K_M, temperature 0, 512 new tokens), DirectRequest, 200 standard text
  behaviours, judge cais/HarmBench-Llama-2-13b-cls in bf16 on CPU with HarmBench's own classifier prompt (peak RSS
  25.7 GiB) · ASR **42.0 %** (84/200; Wilson 95 % CI 35.4-48.9) vs published 46.0 % (Table 6) · diff −4.0 ·
  **PASS** (inside ±5; the published value lies inside the CI). ASR recomputed independently from G4.csv.
  First attempt crashed before judging (missing sentencepiece/protobuf, fixed in acfc1c0); no item was judged twice.

- 2026-11-16 12:24 · G1 (attempt 2, gate version 2) · `run.ps1 gates G1` · commit 3e0388f · LLaMA-2-7B-chat
  (Ollama Q4, temperature 0, 150 new tokens), NQ, Contriever, top-5, 5 adversarial passages per question with the
  question prefix as in PoisonedRAG's get_attack · ASR **0.96** (96/100; Wilson 95 % CI 0.90-0.98) vs published 0.97
  (Table 1) · diff −0.01 · **PASS**. Mean poisoned passages in top-5 = 4.79 (5/5 for 86 questions; was 2.4 in
  attempt 1), confirming the attempt-1 gap was the missing prefix. ASR recomputed independently from G1.csv.

- 2026-11-16 12:40 · Author decisions. G3: run the sensitivity check before deciding (gate `G3_v01`: identical
  run with Mistral-7B-Instruct-v0.1, Q4_K_M GGUF of the official weights, chat template and stop words copied
  from the v0.2 Ollama model; results in G3_v01.csv, the v0.2 results stay as recorded above). G2: run with an
  OpenAI API model (API credits of the authors' ChatGPT Pro subscription); the published comparator is to be read
  from the AgentDojo results table for that exact model snapshot before the run.

- 2026-11-16 22:40 · G3_v01 attempt 1 · stopped after 765/1054 cases (all journaled): Ollama aborted a reply with
  "token repeat limit reached" (one token repeated > 30 times) while Mistral-7B-Instruct-v0.1 simulated an attacker
  tool response for a DS cache miss. Changes before resuming: (a) the simulator for cache misses is fixed to the G3
  run's model (v0.2), so G3 and G3_v01 differ only in the agent model; cases already simulated by v0.1 are redone;
  (b) a repeat-limit abort no longer crashes the run: the reply is streamed and the text produced before the abort
  is scored (other servers would return the same text followed by the repeated token up to max_tokens). Cases with
  such an abort are flagged and counted in the summary (`cases_with_repeat_abort`).

- 2026-11-17 00:10 · G2 decision revised by the authors: no API credit is available (ChatGPT Plus has no API
  credits; an OpenAI key returned 429 "no credits" twice, no task ran), and no free hosted model has a published
  AgentDojo number on the same setup. G2 is reproduced instead against Mu et al. 2025 (Table 3), whose model
  (Llama 3.1 8B Instruct) runs locally. The comparator's setup was reconstructed from the authors' AgentDojo fork
  (normster/agentdojo @ d8477eaf): its LocalLLM prompt is identical to the pinned AgentDojo's; one difference was
  found and removed: the pinned LocalLLM serialises tool results as content-block lists, the fork as plain text
  (wts.agentdojo_hook.text_tool_results_local_llm, also used by E3). Assumption to report: the attack
  (important_instructions) is not named in the paper.

- 2026-11-17 00:55 · G3_v01 (sensitivity check) · resumed at commit a525235 (763/1054 kept, 2 cases redone because
  v0.1 had simulated their attacker response) · Mistral-7B-Instruct-v0.1 (TheBloke Q4_K_M GGUF, v0.2 chat template),
  attacker-response simulator for DS cache misses = v0.2 as in G3 · ASR-valid **16.0 %** (127/792 valid cases;
  Wilson 95 % CI 13.6-18.8; DH 15.4 %, DS 16.8 %; valid rate 75.1 %) vs published 16.7 % · diff −0.7 · inside ±5.
  7 DS step-2 responses were simulated locally (5 valid, 4 succ); counting those 4 as failures gives 15.5 %, still
  inside. No repeat-limit aborts. Recomputed independently from G3_v01.csv.
  Conclusion recorded by the authors: InjecAgent's unnamed "Mistral-7B" behaves like Instruct v0.1 (−0.7) rather
  than v0.2 (+5.4); G3 is marked PASS on v0.1 and both numbers are reported in the paper as release sensitivity.

