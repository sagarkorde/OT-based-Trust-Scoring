# Environment lock

Recorded 2026-11-15. Update this file (never silently) whenever anything below changes, and cite the
release commit of this repository in the paper.

## Benchmark repositories (pinned by commit)

| Repo | URL | Commit | Used for |
| --- | --- | --- | --- |
| AgentDojo (package version 0.1.35 at this commit) | https://github.com/ethz-spylab/agentdojo | 089ed468cf3ed0322acc66b0211f26d9d90dbf60 | E3, G2 |
| InjecAgent | https://github.com/uiuc-kang-lab/InjecAgent | f19c9f2c79a41046eb13c03c51a24c567a8ffa07 | E3, G3 |
| PoisonedRAG | https://github.com/sleeepeer/PoisonedRAG | f660d72174f06b13fae5163ce656e7b235db858f | E2, G1 |
| HarmBench | https://github.com/centerforaisafety/HarmBench | 8e1604d1171fe8a48d8febecd22f600e462bdcdd | E1, G4 |

Clone with `make benchmarks` (checks out exactly these commits into `ext/`).
AgentDojo benchmark version for all runs: `v1.2.2`, the latest at the pinned commit (set in `configs/e3.yaml`; record here if changed).

## Models (record the Hugging Face revision hash when first downloaded)

| Role | Model id | Revision | Notes |
| --- | --- | --- | --- |
| Open LLM 1 | meta-llama/Llama-3.1-8B-Instruct | TO RECORD | gated; accept license |
| Open LLM 2 | Qwen/Qwen2.5-7B-Instruct | TO RECORD | |
| Open LLM 3 | mistralai/Mistral-7B-Instruct-v0.3 | TO RECORD | first to drop if compute is short |
| API LLM | gpt-4o-mini (or cheapest current function-calling model) | dated model string | record exact string |
| Encoder 1 | BAAI/bge-base-en-v1.5 | TO RECORD | |
| Encoder 2 | sentence-transformers/all-MiniLM-L6-v2 | TO RECORD | |
| Judge baseline | meta-llama/Llama-Guard-3-8B | TO RECORD | |
| AgentDojo detector baseline | protectai/deberta-v3-base-prompt-injection-v2 | TO RECORD | used by AgentDojo's `transformers_pi_detector` |

## Datasets (record HF revision or file hash)

| Dataset | Source | Revision / hash |
| --- | --- | --- |
| deepset/prompt-injections | Hugging Face | TO RECORD |
| tatsu-lab/alpaca | Hugging Face | TO RECORD |
| lmsys/lmsys-chat-1m | Hugging Face (gated; do not redistribute) | TO RECORD |
| TrustAIRLab/in-the-wild-jailbreak-prompts | Hugging Face | TO RECORD |
| HarmBench behaviors + test cases | HarmBench repo / release | commit above |
| BEIR nq, hotpotqa, msmarco | BEIR download server | TO RECORD (zip sha256) |

## Python (development machine, 2026-11-15)

Python 3.13; numpy 2.5.3; scipy 1.18.1; scikit-learn 1.9.1; POT 0.9.7.post1; pandas 3.0.5; matplotlib 3.11.2.
GPU machine: record `pip freeze > docs/pip-freeze-gpu.txt` and CUDA / driver versions before Phase 2.

## Hardware

Development (this commit): 2 CPU cores, 7 GB RAM, no GPU. Used only for library code and unit tests.
Experiments: one A100/H100-class GPU (record model, memory, driver).
