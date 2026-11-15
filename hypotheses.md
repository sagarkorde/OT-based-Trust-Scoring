# Pre-registered hypotheses

Status: FROZEN at commit time. This file must not be edited after it is committed.
Any change goes in a dated addendum at the bottom, below the line marked ADDENDA,
and the paper reports both the original and the addendum.

Copied verbatim from Section 4 of the research plan
("Research Plan — GenAI, Agentic AI and RAG Security: OT-based Trust Scoring", 15 Nov 2026).

Pre-registration date: 2026-11-15. No WTS score had been computed on any benchmark item when this file was committed.

| RQ | Question | Hypothesis | Pass criterion | Fails gracefully to |
| --- | --- | --- | --- | --- |
| RQ1 | Does the shape term of W_2 add detection power over location-only scores for direct injection and jailbreaks? | H1: Gaussian W_2 beats cosine, Euclidean and Mahalanobis at matched FPR | AUROC gain >= 3 points over the best location-only baseline on at least 2 of 3 encoders, 95% bootstrap CI excluding 0 | Report that the signal is mostly in the mean; keep the unified framework and calibration claims |
| RQ2 | Does per-chunk WTS filtering cut RAG poisoning and jamming success while keeping clean accuracy? | H2: ASR falls by at least half relative to no defense | Relative ASR reduction >= 50% on PoisonedRAG (NQ, HotpotQA, MS-MARCO) with clean accuracy loss <= 2 points | Report the trade-off curve; compare to RobustRAG and paraphrase defenses at equal utility |
| RQ3 | Does tool-output scoring plus intent drift reduce targeted attack success in agents without hurting utility? | H3: targeted ASR falls by >= 40% relative on AgentDojo and InjecAgent | ASR reduction >= 40% with benign utility loss <= 5 points, beating or matching spotlighting and the DeBERTa injection detector baked into AgentDojo | Report which of the two signals (output score or drift) carried the result |
| RQ4 | Do conformal thresholds deliver the promised false-alarm rate, and how do they degrade under shift? | H4: empirical FPR <= alpha + 1 point in distribution | Measured at alpha in {0.01, 0.05, 0.10} with n_cal in {50, 100, 200, 500, 1000}; cross-corpus shift (Alpaca -> LMSYS-Chat) reported as a coverage gap | The gap becomes a quantified limitation and a motivation for per-channel recalibration |
| RQ5 | How much attack success does a defense-aware attacker recover? | H5 (expected partial evasion): dilution and paraphrase attacks recover at least half the lost ASR | Report recovered ASR and the attacker's cost in added tokens; relate cost to the Proposition 2 bound | Either outcome is informative; the paper reports the evasion cost curve |
| RQ6 | Is the score cheap enough to run on every input and every agent step? | H6: < 20 ms per unit for Gaussian W_2 at d = 768 on one GPU; >= 10x cheaper than an 8B-parameter LLM judge | Median and p95 latency over 1,000 units, same hardware for all methods | If sliced W_2 is needed for accuracy, report its higher cost honestly |

Secondary analyses (no hypothesis, exploratory, labelled as such in the paper): which LLM layer gives the
best white-box score; how the location and shape terms split by attack family; whether scores transfer
across LLMs when the encoder is external.

## Operational definitions fixed at pre-registration

These pin down terms in the table so that the criteria cannot drift later.

- "Location-only baselines" (H1) = cosine distance to reference centroid, Euclidean distance to reference
  centroid, Mahalanobis distance of the mean-pooled unit embedding. Kernel MMD is reported but is not a
  location-only baseline.
- "3 encoders" (H1) = bge-base-en-v1.5 tokens, all-MiniLM-L6-v2 tokens, Llama-3.1-8B-Instruct layer 16
  projected to 768 dims by PCA fit on reference data only.
- "AUROC gain" (H1) = AUROC(Gaussian W_2) − max over location-only baselines of AUROC, pooled over the E1
  attack families, paired bootstrap over items, 10,000 resamples.
- "ASR" (H2) = PoisonedRAG rule: cleaned incorrect answer is a substring of the cleaned response.
  "Clean accuracy" (H2) = same rule with the correct answer, on the same 100 target questions with no
  poisoned passages injected.
- "Targeted ASR" (H3) = AgentDojo security result (injection task goal achieved), and InjecAgent
  ASR-valid as defined by its evaluator. "Benign utility" = AgentDojo utility without injections.
- "Relative reduction" = (ASR_none − ASR_defended) / ASR_none.
- "In distribution" (H4) = calibration and test benign items drawn from disjoint random splits of the same
  benign source.
- "Recovered ASR" (H5) = (ASR_adaptive − ASR_defended) / (ASR_none − ASR_defended).
- Significance level for all tests: 0.05, two-sided, Holm-corrected within each results table.
- Default hyper-parameters, fixed now: alpha = 0.05 for all main results; Ledoit–Wolf shrinkage;
  sliced W_2 with L = 128 projections; RAG leave-one-out pool k' = 20, corpus mixing weight lambda = 0.5;
  agent guard combines output and drift p-values by Bonferroni (flag if min p <= alpha/2).

---
ADDENDA (dated; appended only)
