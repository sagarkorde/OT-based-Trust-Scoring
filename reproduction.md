# Reproduction gates (Phase 2)

No WTS experiment (E1–E4) may start until every gate below is PASS or has an explained gap
signed off by the authors. Tolerance: ±5 absolute percentage points.

Rule: the "Published" column is filled by reading the cited table in the cited paper, and the
table/figure number is written next to the value. Never fill it from memory. Leave `TO FILL` until done.

| Gate | What | Setting | Published (source table) | Reproduced (mean over seeds) | Seeds | Diff | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | PoisonedRAG ASR, NQ | Black-box attack (released adversarial texts), N = 5 passages per question, Contriever, top-k = 5, dot product, 100 target questions; LLM LLaMA-2-7B (laptop: Ollama llama2:7b-chat Q4) | 0.97 (PoisonedRAG [14], Table 1, NQ black-box, LLaMa-2-7B) | 0.96 (96/100; 95 % CI 0.90-0.98) | 1 (temperature 0) | −0.01 | **PASS** | Paper uses temperature 0.1 (we use 0) and fp16 weights (we use Q4); ASR by substring match as in the paper |
| G2 | AgentDojo targeted ASR and utility, no defense | `important_instructions` attack, all 4 suites, benchmark v1, gpt-4o-mini-2024-07-18 via OpenAI API (temperature 0) | Targeted ASR 27.19 %; utility 68.04 %; utility under attack 49.92 % (AgentDojo results table, docs/results-table.html at pinned commit 089ed468: openai, gpt-4o-mini-2024-07-18, defense None, important_instructions, dated 2024-07-19) | — | — | — | NOT RUN | Row predates benchmark v1.1, so v1 is assumed; the gate is judged on targeted ASR, utility reported alongside |
| G3 | InjecAgent ASR-valid, base setting | Prompted (ReAct) agent, DH and DS, InjecAgent's own evaluator; laptop: Ollama mistral:7b-instruct-v0.2 Q4, 8k context | 16.7 % (InjecAgent [22], Table 3, prompted Mistral-7B, base setting, Total) | 22.1 % (v0.2; sensitivity run with v0.1 pending) | 1 (temperature 0) | +5.4 | PENDING (sensitivity check) | Check which Mistral-7B release the paper used (appendix model list) before comparing |
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
