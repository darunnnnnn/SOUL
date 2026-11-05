"""Trace analysis over the raw evaluation outputs.

Answers three questions that the headline table cannot:

1. **How much of the gold evidence is reachable only by a bridged hop?**
   Each retrieved passage carries the hop at which it entered the pool. Gold
   passages that entered at hop >= 1 were, by construction, not proposed by the
   round-0 query -- i.e. they are the passages no single-shot retriever can
   reach. This is the direct quantification of the submission's central claim.

2. **Does the adaptive stop actually fire?** The stop compares a per-round
   min-max normalised score against the running best, and round 0's
   normalisation pins that best at 1.0 -- so the criterion may be inert. The
   traces say how often it triggers, and what it costs in queries.

3. **Where does the query budget go?** Queries per question by hop type, which
   is the cost side of the gain.

Pure analysis: reads only committed raw outputs, no model, no retrieval.

    python scripts/analyze_traces.py [label] [system]
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import ctxlayer
from ctxlayer.config import PathsConfig

LABEL = sys.argv[1] if len(sys.argv) > 1 else "unified"
SYSTEM = sys.argv[2] if len(sys.argv) > 2 else "ours"


def load(label: str, system: str):
    p = PathsConfig().resolve("raw_outputs_dir") / label / f"{system}.jsonl"
    if not p.exists():
        raise SystemExit(f"not found: {p}\nRun the evaluation first.")
    header, rows = {}, []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        (rows.append(r) if r.get("_record") != "header" else header.update(r))
    return header, rows


def main() -> int:
    header, rows = load(LABEL, SYSTEM)
    n = len(rows)
    print(f"\n{'='*86}")
    print(f"TRACE ANALYSIS  --  system '{SYSTEM}', label '{LABEL}', n={n}")
    print(f"{'='*86}")

    # ---------------------------------------------------------------- 1
    # Gold passages, by the hop at which they were retrieved.
    hop_of_gold: collections.Counter = collections.Counter()
    gold_found = gold_total = 0
    only_bridged_q = 0
    per_hoptype: dict[str, list[float]] = collections.defaultdict(list)

    for r in rows:
        gold = set(r["gold_passage_ids"])
        gold_total += len(gold)
        ranked = r["ranked_passage_ids"]
        hops = r.get("ranked_hops") or [0] * len(ranked)
        hop_by_pid = dict(zip(ranked, hops))
        found = gold & set(ranked)
        gold_found += len(found)
        bridged = 0
        for pid in found:
            h = hop_by_pid.get(pid, 0)
            hop_of_gold[h] += 1
            if h >= 1:
                bridged += 1
        if bridged:
            only_bridged_q += 1
        per_hoptype[r["hop_type"]].append(bridged / max(1, len(gold)))

    print("\n1. WHERE GOLD EVIDENCE WAS FOUND  (hop at which it entered the pool)")
    print("   hop 0 = proposed by the original question, i.e. reachable by a")
    print("   single-shot retriever. hop >= 1 = reached only by a bridged query.\n")
    tot = sum(hop_of_gold.values())
    for h in sorted(hop_of_gold):
        share = hop_of_gold[h] / tot * 100 if tot else 0
        label = "round 0 (single-shot reachable)" if h == 0 else f"hop {h} (bridged)"
        print(f"     {label:36s} {hop_of_gold[h]:5d}  {share:5.1f}%")
    bridged_total = sum(v for k, v in hop_of_gold.items() if k >= 1)
    print(f"\n   gold passages retrieved           : {gold_found}/{gold_total}")
    if tot:
        print(f"   of those, reached ONLY by a bridge: {bridged_total} "
              f"({bridged_total/tot*100:.1f}%)")
    print(f"   questions with >=1 bridged gold    : {only_bridged_q}/{n} "
          f"({only_bridged_q/max(1,n)*100:.1f}%)")

    print("\n   share of a question's gold evidence that was bridged, by hop type:")
    for k in sorted(per_hoptype):
        v = per_hoptype[k]
        print(f"     {k:8s} {statistics.mean(v)*100:5.1f}%   (n={len(v)})")

    # ---------------------------------------------------------------- 2
    print("\n2. ADAPTIVE STOP")
    stops: collections.Counter = collections.Counter()
    rounds_used: list[int] = []
    for r in rows:
        rds = r["trace"]["rounds"]
        reasons = [d.get("stopped") for d in rds if d.get("stopped")]
        stops[reasons[0] if reasons else "ran_full_budget"] += 1
        rounds_used.append(max((d.get("round", 0) for d in rds), default=0))
    for k, v in stops.most_common():
        print(f"     {k:28s} {v:5d}  ({v/max(1,n)*100:5.1f}%)")
    if rounds_used:
        print(f"     mean rounds reached        : {statistics.mean(rounds_used):.2f}")

    # ---------------------------------------------------------------- 3
    print("\n3. QUERY BUDGET")
    by_hop: dict[str, list[int]] = collections.defaultdict(list)
    allq = []
    for r in rows:
        q = r["trace"]["n_queries"]
        allq.append(q)
        by_hop[r["hop_type"]].append(q)
    if allq:
        print(f"     queries/question  mean {statistics.mean(allq):.2f}  "
              f"median {statistics.median(allq):.0f}  max {max(allq)}")
    for k in sorted(by_hop):
        print(f"       {k:8s} {statistics.mean(by_hop[k]):.2f}")
    lat = [r["context_ms"] for r in rows]
    if lat:
        print(f"     context latency   median {statistics.median(lat):.0f} ms  "
              f"p95 {sorted(lat)[int(0.95*len(lat))]:.0f} ms")

    # ---------------------------------------------------------------- 4
    print("\n4. BRIDGE ENTITIES  (most frequently followed)")
    ents: collections.Counter = collections.Counter()
    for r in rows:
        ents.update(r["trace"]["bridge_entities"])
    for e, c in ents.most_common(12):
        print(f"     {c:4d}  {e}")
    print(f"     distinct bridge entities followed: {len(ents)}")

    out = PathsConfig().resolve("results_dir") / f"trace_analysis_{LABEL}_{SYSTEM}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "system": SYSTEM,
                "label": LABEL,
                "n": n,
                "gold_by_hop": {str(k): v for k, v in sorted(hop_of_gold.items())},
                "gold_retrieved": gold_found,
                "gold_total": gold_total,
                "bridged_share_of_retrieved_gold": (
                    bridged_total / tot if tot else 0.0
                ),
                "questions_with_bridged_gold": only_bridged_q,
                "stop_reasons": dict(stops),
                "queries_per_question_mean": (
                    statistics.mean(allq) if allq else 0.0
                ),
                "bridged_share_by_hop_type": {
                    k: statistics.mean(v) for k, v in per_hoptype.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
