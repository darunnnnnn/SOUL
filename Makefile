# Convenience wrappers. Everything here is also a plain `python -m ctxlayer.cli`
# call -- see README.md -- so make is never required.

PY ?= python
SUBSET ?= 500
SETTING ?= unified
LIMIT ?=

.PHONY: help install test reproduce download ingest index eval ablate report \
        diagnose walkthrough clean clean-all

help:
	@echo "make install      - editable install + dependencies"
	@echo "make test         - run the test suite (no network, ~8s)"
	@echo "make reproduce    - THE one command: data -> indexes -> runs -> EVAL.md"
	@echo "make eval         - run the systems (SETTING=unified LIMIT=100)"
	@echo "make ablate       - run the ablation suite"
	@echo "make report       - rebuild all results from raw outputs, offline"
	@echo "make diagnose     - reproduce the design measurements"
	@echo "make walkthrough  - scripted 5-10 minute demo"

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

# Hard Requirement #4: one command reruns the evaluation and rebuilds the
# results from raw outputs.
reproduce:
	$(PY) -m ctxlayer.cli reproduce --subset $(SUBSET) --setting $(SETTING)

download:
	$(PY) -m ctxlayer.cli download

ingest:
	$(PY) -m ctxlayer.cli ingest --subset $(SUBSET)

index:
	$(PY) -m ctxlayer.cli index

eval:
	$(PY) -m ctxlayer.cli eval --setting $(SETTING) $(if $(LIMIT),--limit $(LIMIT),)

ablate:
	$(PY) -m ctxlayer.cli ablate --setting $(SETTING) $(if $(LIMIT),--limit $(LIMIT),)

report:
	$(PY) -m ctxlayer.cli report --setting $(SETTING) --write-md

diagnose:
	$(PY) -m ctxlayer.cli diagnose corpus
	$(PY) -m ctxlayer.cli diagnose anchors
	$(PY) -m ctxlayer.cli diagnose reachability

walkthrough:
	$(PY) scripts/walkthrough.py $(SETTING)

# Remove derived artefacts but KEEP eval/raw_outputs -- those are the evidence
# the report is rebuilt from, and are committed.
clean:
	rm -rf data/index data/cache data/processed eval/results

clean-all: clean
	rm -rf data/raw
