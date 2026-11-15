PY ?= python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: help install install-gpu benchmarks test smoke latency gate-check e1 e2 e3 e4 tables figures refs paper all-results

help:
	@echo "install       CPU deps (library + tests)"
	@echo "install-gpu   GPU experiment deps"
	@echo "benchmarks    clone pinned benchmark repos into ext/"
	@echo "test          unit tests (CPU)"
	@echo "smoke         end-to-end plumbing check with the test-only hashing encoder (CPU, no downloads)"
	@echo "latency       E5 latency benchmark (use --device cuda on the GPU machine)"
	@echo "gate-check    refuse to run E1-E4 unless reproduction.md gates all PASS"
	@echo "e1..e4        experiments (GPU machine)"
	@echo "tables        results/*.csv -> paper/tables/*.tex"
	@echo "figures       results/*.csv -> paper/figures/*.pdf"
	@echo "refs          verify every DOI in paper/refs.bib against Crossref/DataCite"

install:
	$(PY) -m pip install -r requirements.txt

install-gpu:
	$(PY) -m pip install -r requirements-gpu.txt

benchmarks:
	mkdir -p ext
	test -d ext/agentdojo   || git clone https://github.com/ethz-spylab/agentdojo ext/agentdojo
	test -d ext/InjecAgent  || git clone https://github.com/uiuc-kang-lab/InjecAgent ext/InjecAgent
	test -d ext/PoisonedRAG || git clone https://github.com/sleeepeer/PoisonedRAG ext/PoisonedRAG
	test -d ext/HarmBench   || git clone https://github.com/centerforaisafety/HarmBench ext/HarmBench
	cd ext/agentdojo   && git checkout -q 089ed468cf3ed0322acc66b0211f26d9d90dbf60
	cd ext/InjecAgent  && git checkout -q f19c9f2c79a41046eb13c03c51a24c567a8ffa07
	cd ext/PoisonedRAG && git checkout -q f660d72174f06b13fae5163ce656e7b235db858f
	cd ext/HarmBench   && git checkout -q 8e1604d1171fe8a48d8febecd22f600e462bdcdd

test:
	$(PY) -m pytest -q tests

smoke:
	$(PY) scripts/smoke.py

latency:
	$(PY) scripts/bench_latency.py --out results/e5_overhead.csv

gate-check:
	$(PY) scripts/gate_check.py

e1: gate-check
	$(PY) scripts/run_e1_detection.py --config configs/e1.yaml

e2: gate-check
	$(PY) scripts/run_e2_rag.py --config configs/e2.yaml

e3: gate-check
	$(PY) scripts/run_e3_agentdojo.py --config configs/e3.yaml
	$(PY) scripts/run_e3_injecagent.py --config configs/e3.yaml

e4: gate-check
	$(PY) scripts/run_e4_adaptive.py --config configs/e4.yaml

tables:
	$(PY) scripts/make_tables.py --results results --out paper/tables

figures:
	$(PY) scripts/make_figures.py --results results --out paper/figures

refs:
	$(PY) scripts/verify_refs.py paper/refs.bib
