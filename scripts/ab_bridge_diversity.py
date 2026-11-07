"""Measure the bridge-diversity variant that was tried and rejected.

Nested bridge candidates ("North Holland" and "Holland" extracted from one
passage) can crowd a round's queries onto a single entity. The variant capped
bridges per passage at 2 and discarded any candidate nested inside a
higher-ranked sibling:

    max_bridges_per_passage = 2
    dedupe_nested_bridges   = True

It lost. This script re-derives the paired comparison from the two committed
runs, so the negative result in EVAL.md section 9 can be checked rather than
taken on trust.

    python scripts/ab_bridge_diversity.py

Shipped run  : eval/raw_outputs/unified/ours.jsonl       (defaults, no-op values)
Variant run  : eval/raw_outputs/unified_div/ours.jsonl   (the two settings above)

Both runs cover the identical 500 questions, so every comparison is paired
per question and tested by a paired permutation test -- the same procedure
the main report uses.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "eval" / "raw_outputs" / "unified" / "ours.jsonl"
VARIANT = ROOT / "eval" / "raw_outputs" / "unified_div" / "ours.jsonl"

METRICS = ["ctx_all_docs", "ctx_doc_coverage", "ctx_support_all", "answer_em", "answer_f1"]
TUNING_SPLIT = 150          # first 150 qids by sorted order are development data
RESAMPLES = 10_000
SEED = 20251105


def load(path: Path) -> tuple[dict, dict]:
    """Return (header, {qid: record}). The first line is the run header."""
    with path.open(encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        records = {}
        for line in fh:
            rec = json.loads(line)
            records[rec["qid"]] = rec
    return header, records


def value(rec: dict, key: str) -> float:
    return float(rec.get("metrics", rec)[key])


def permutation_p(deltas: list[float], resamples: int = RESAMPLES, seed: int = SEED) -> float:
    """Two-sided paired permutation test: flip the sign of each pair at random."""
    observed = abs(sum(deltas) / len(deltas))
    rng = random.Random(seed)
    hits = 0
    for _ in range(resamples):
        total = sum(d if rng.random() < 0.5 else -d for d in deltas)
        if abs(total / len(deltas)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (resamples + 1)


def describe_config(header: dict) -> str:
    cfg = header.get("config", {})
    it = cfg.get("iterative", cfg)
    keys = ("max_bridges_per_passage", "dedupe_nested_bridges")
    return "  ".join(f"{k}={it.get(k)}" for k in keys)


def main() -> int:
    for path in (SHIPPED, VARIANT):
        if not path.exists():
            print(f"missing run: {path.relative_to(ROOT)}")
            return 1

    h_ship, ship = load(SHIPPED)
    h_var, var = load(VARIANT)

    qids = sorted(set(ship) & set(var))
    if not qids:
        print("the two runs share no questions")
        return 1

    print("shipped : commit %s  %s" % (h_ship["environment"]["git_commit"][:10], describe_config(h_ship)))
    print("variant : commit %s  %s" % (h_var["environment"]["git_commit"][:10], describe_config(h_var)))
    print()
    print(f"paired on {len(qids)} questions"
          f" ({len(ship)} and {len(var)} in the two runs)")
    print(f"paired permutation test, {RESAMPLES:,} resamples, seed {SEED}")

    splits = [("all questions", qids), (f"held out (drop first {TUNING_SPLIT})", qids[TUNING_SPLIT:])]

    for label, ids in splits:
        print()
        print(f"--- {label}, n = {len(ids)} ---")
        print(f"{'metric':<18} {'shipped':>8} {'variant':>8} {'delta':>8} {'p':>8}")
        for key in METRICS:
            a = [value(ship[q], key) for q in ids]
            b = [value(var[q], key) for q in ids]
            deltas = [y - x for x, y in zip(a, b)]
            delta = sum(deltas) / len(deltas) * 100
            p = permutation_p(deltas)
            mark = "  <-- worse" if delta < 0 and p < 0.05 else ""
            print(f"{key:<18} {sum(a)/len(a)*100:8.1f} {sum(b)/len(b)*100:8.1f} "
                  f"{delta:+8.1f} {p:8.4f}{mark}")

    print()
    print("A negative delta means the diversity variant retrieved less gold")
    print("evidence than the shipped configuration. It was reverted; the")
    print("defaults in src/ctxlayer/config.py are the no-op values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
