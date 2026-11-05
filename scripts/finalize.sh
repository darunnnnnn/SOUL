#!/usr/bin/env bash
# Everything after the headline run, in order, unattended.
#
# Each step is a separate process, which matters: the environment fingerprint
# (and therefore the git commit recorded in the raw outputs) is captured once
# per process, and report.py refuses to pair runs from different commits. So
# every output DIRECTORY stays internally consistent even if the working tree
# is committed between steps.
#
#   bash scripts/finalize.sh
#
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

log() { printf '\n==== %s  (%s) ====\n' "$1" "$(date +%H:%M:%S)"; }

# 1. What the iteration bought: hop at which each gold passage was found.
log "trace analysis"
python -u scripts/analyze_traces.py unified ours || echo "FAILED: traces"

# 2. Ablation suite, on the DEVELOPMENT split (the first 150 questions, the
#    same ones hyperparameter selection used), so any decision informed by the
#    ablation table leaves the held-out split untouched.
log "ablations (dev split, n=150)"
python -u -m ctxlayer.cli ablate --setting unified --limit 150 \
  || echo "FAILED: ablations"

# 3. The standard 20-paragraph MuSiQue setting, for comparability with
#    published work. Fewer questions: this is a secondary table.
log "pool setting (comparability, n=300)"
python -u -m ctxlayer.cli eval --setting pool --limit 300 \
  || echo "FAILED: pool"

# 4. Honest latency. Must be last: it wants an idle machine, and it bypasses
#    the caches that every step above relies on.
log "cold-cache latency"
python -u scripts/bench_latency.py 25 unified || echo "FAILED: latency"

# 5. Rebuild every published number from the raw outputs, and regenerate
#    EVAL.md. Offline; reads nothing but eval/raw_outputs/.
log "report + EVAL.md"
python -u -m ctxlayer.cli report --setting unified --write-md || echo "FAILED: report"
python -u -m ctxlayer.cli report --setting unified --holdout  || echo "FAILED: holdout"
python -u -m ctxlayer.cli report --setting pool                || echo "FAILED: pool report"

# 6. Fail loudly if any hand-written figure disagrees with its artefact.
log "documentation claims vs artefacts"
python -u scripts/check_doc_claims.py || echo "FAILED: doc claims"

log "done"
