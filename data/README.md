# Data

Nothing in this folder is committed except this file. Downloads land in `data/raw/`, encoder caches in
`data/cache/` (both git-ignored).

| Dataset | Loader (`src/wts/data.py`) | Used in | Licence / access | Notes |
| --- | --- | --- | --- | --- |
| tatsu-lab/alpaca | `alpaca_prompts` | E1 benign (ref / cal / test), PCA pool, fillers | CC BY-NC 4.0 | 2,000 rows + 1,000 PCA rows per sampling seed |
| lmsys/lmsys-chat-1m | `lmsys_prompts` | E1 benign shift set (RQ4) | gated; do not redistribute | only conversation ids are saved |
| deepset/prompt-injections | `deepset_injections` | E1 attacks; PIShield-probe training (train split only) | check dataset card | label 1 rows |
| TrustAIRLab/in-the-wild-jailbreak-prompts | `in_the_wild_jailbreaks` | E1 attacks | see dataset card | config jailbreak_2023_12_25 |
| HarmBench behaviors + test cases | `harmbench_direct_requests`, `harmbench_test_cases` | E1 attacks, E4 GCG | check repository licence | test cases generated with HarmBench step 1 |
| PoisonedRAG adversarial texts + BEIR top-100 | `poisonedrag_targets`, `poisonedrag_retrieval` | E2 | see repo | from pinned commit |
| BEIR nq, hotpotqa, msmarco | `beir_corpus` | E2 corpora | see BEIR | downloaded on first use |
| AgentDojo suites v1.2.2 | via `agentdojo` | E3 | check repository licence | |
| InjecAgent test cases | `injecagent_cases` | E3, E4 | check repository licence | |

Record the exact revision of each in `environment.lock.md` and the sample counts actually obtained here
after the first GPU run (do not copy published sizes).
