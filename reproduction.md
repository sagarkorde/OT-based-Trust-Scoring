# Reproduction gates (Phase 2)

No WTS experiment (E1–E4) may start until every gate below is PASS or has an explained gap
signed off by the authors. Tolerance: ±5 absolute percentage points.

Rule: the "Published" column is filled by reading the cited table in the cited paper, and the
table/figure number is written next to the value. Never fill it from memory. Leave `TO FILL` until done.

| Gate | What | Setting | Published (source table) | Reproduced (mean over seeds) | Seeds | Diff | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | PoisonedRAG ASR, NQ | Black-box attack (LM_targeted), 5 adv passages/question, Contriever, top-k = 5, dot score, prompt_id 4; one open LLM the paper reports | TO FILL (PoisonedRAG [14], main results table) | — | — | — | NOT RUN | Released adv texts: `results/adv_targeted_results/nq.json` at PoisonedRAG commit pinned in environment.lock.md |
| G2 | AgentDojo targeted ASR and utility, no defense | `important_instructions` attack, all 4 suites, benchmark version pinned in environment.lock.md; model closest to one reported by AgentDojo [21] | TO FILL (AgentDojo leaderboard / paper results table, same benchmark version) | — | — | — | NOT RUN | AgentDojo numbers change between benchmark versions; compare like with like |
| G3 | InjecAgent ASR-valid, base setting | Prompted agent, DH and DS splits | TO FILL (InjecAgent [22], main results table) | — | — | — | NOT RUN | Use the repo's `evaluate_prompted_agent.py` evaluator unchanged |
| G4 | HarmBench ASR for one attack/model pair | e.g. GCG or PAIR test cases against a model HarmBench reports, HarmBench classifier as judge | TO FILL (HarmBench [10], results table) | — | — | — | NOT RUN | Use the HarmBench classifier, never an ad-hoc judge |

## Log

Append one entry per attempt: date, gate, command, commit hash, result, explanation of any gap.
