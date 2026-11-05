"""A/B the assembly relevance floor, in both retrieval settings.

Why
---
The 20-paragraph `pool` setting exposed a weakness: our assembly is selective
(relevance floor + per-document cap), which is a win when retrieval is hard and
the token budget is genuinely contested, but looks like pure loss when the
candidate pool is 20 clean paragraphs and you can simply keep half of them.

The floor's *purpose* is to stop the deep tail being pulled up by the diversity
and coverage terms. It was never meant to reject head candidates. So the
candidate fix is to exempt anything already within the top `max_passages` by
relevance -- the floor then targets the tail only.

Retrieval-only (no reader): the floor affects which passages reach the context,
and support/document coverage measure exactly that. Skipping the reader makes
this ~20x cheaper, so the decision is made on a few hundred questions rather
than a handful.

    python scripts/ab_assembly.py [n] [settings]
"""

from __future__ import annotations

import collections
import statistics
import sys
import time

import ctxlayer
from ctxlayer.config import PRESETS, PathsConfig
from ctxlayer.eval.metrics import context_metrics, retrieval_metrics
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.ingest import load_processed
from ctxlayer.pipeline import ContextLayer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SETTINGS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["unified", "pool"])

VARIANTS = {
    # The baseline to beat, in each setting.
    "hybrid_rerank": PRESETS["hybrid_rerank"],
    # Current behaviour.
    "ours_floor_all": PRESETS["ours"],
    # Floor disabled entirely -- the upper bound on "selectivity costs recall".
    "ours_no_floor": PRESETS["ours"].replace(**{"assemble.min_rel_ratio": 0.0}),
    # Candidate fix: floor applies to the tail only (implemented behind
    # `floor_exempt_top_k`; see AssembleConfig).
    "ours_floor_tail": PRESETS["ours"].replace(
        **{"assemble.floor_exempt_top_k": True}
    ),
    # Also let more of one document through, to separate the two constraints.
    "ours_floor_tail_cap3": PRESETS["ours"].replace(
        **{"assemble.floor_exempt_top_k": True, "assemble.max_per_doc": 3}
    ),
}

COLS = [
    "ctx_support_all",
    "ctx_all_docs",
    "ctx_support_recall",
    "ctx_doc_coverage",
    "ctx_words",
    "ctx_n_docs",
]


def main() -> int:
    import json

    paths = PathsConfig()
    res = load_processed(paths)
    kb = KnowledgeBase.load(res, PRESETS["ours"], paths)
    kb.enable_score_cache(paths, PRESETS["ours"])
    p2d = {p.passage_id: p.doc_id for p in kb.passages.values()}
    qs = res.questions[:N]

    print(f"\ncorpus {kb.n_passages} passages | n={len(qs)} questions "
          f"| retrieval-only (no reader)\n")

    all_results: dict[str, dict] = {}
    for setting in SETTINGS:
        print("=" * 104)
        print(f"SETTING: {setting}")
        print("=" * 104)
        print(f"{'variant':24s}" + "".join(f"{c.replace('ctx_',''):>14s}" for c in COLS)
              + f"{'sec/q':>9s}")
        print("-" * 104)
        rows = {}
        for name, cfg in VARIANTS.items():
            layer = ContextLayer(kb, cfg)
            agg = collections.defaultdict(list)
            t0 = time.time()
            for q in qs:
                ctx, ranked = layer.build_context_for(q, setting=setting)
                m = retrieval_metrics(
                    [s.passage_id for s in ranked], q.gold_passage_ids,
                    q.gold_doc_ids, p2d, (10,),
                )
                m.update(context_metrics(ctx.passage_ids, q.gold_passage_ids,
                                         q.gold_doc_ids, p2d, ctx.n_words))
                for k, v in m.items():
                    agg[k].append(v)
            dt = time.time() - t0
            rows[name] = {k: statistics.mean(v) for k, v in agg.items()}
            cells = "".join(
                f"{rows[name][c]:14.1f}" if c in ("ctx_words", "ctx_n_docs")
                else f"{rows[name][c]*100:14.1f}"
                for c in COLS
            )
            print(f"{name:24s}{cells}{dt/len(qs):9.2f}")
        print()
        all_results[setting] = rows

    # Persist, so the table quoted in DESIGN.md can be checked against the
    # measurement that produced it rather than trusted
    # (scripts/check_doc_claims.py).
    out = paths.resolve("results_dir") / "ab_assembly.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": len(qs), "settings": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
