"""Cold- and warm-cache latency, measured honestly.

Why a separate script
---------------------
The `context_ms` recorded in the raw eval outputs is measured with the
cross-encoder score cache and the query-embedding cache **warm**, because those
caches are what make a 22-ablation sweep affordable. That number is the right
one for "what does an ablation sweep cost", and the wrong one for "what does a
single user query cost" -- reporting only it would understate real latency by a
large factor.

So this reports both, per system, on the same questions:

  cold   every cache bypassed: the true cost of answering a fresh question
  warm   caches populated: the amortised cost inside a sweep or a re-run

Run it on an otherwise idle machine; concurrent work inflates both numbers.

    python scripts/bench_latency.py [n] [setting]
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import ctxlayer
from ctxlayer.config import PRESETS, PathsConfig
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.ingest import load_processed
from ctxlayer.pipeline import ContextLayer
from ctxlayer.synth.reader import ExtractiveReader

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SETTING = sys.argv[2] if len(sys.argv) > 2 else "unified"
SYSTEMS = ["bm25", "dense", "hybrid", "hybrid_rerank", "ours"]


def bench(kb, cfg, questions, reader, warm: bool) -> dict[str, float]:
    layer = ContextLayer(kb, cfg)
    ctx_ms: list[float] = []
    read_ms: list[float] = []
    for q in questions:
        if not warm:
            # Bypass every memoisation so this measures real work.
            kb._score_cache = None
            kb._rerankers.clear()
            if kb.dense is not None:
                kb.dense._qcache.clear()
            layer = ContextLayer(kb, cfg)
        t0 = time.perf_counter()
        ctx, _ = layer.build_context_for(q, setting=SETTING)
        ctx_ms.append((time.perf_counter() - t0) * 1000)
        if reader is not None:
            t1 = time.perf_counter()
            reader.read(q.question, ctx)
            read_ms.append((time.perf_counter() - t1) * 1000)
    out = {
        "ctx_median_ms": statistics.median(ctx_ms),
        "ctx_p95_ms": sorted(ctx_ms)[min(len(ctx_ms) - 1, int(0.95 * len(ctx_ms)))],
    }
    if read_ms:
        out["read_median_ms"] = statistics.median(read_ms)
        out["total_median_ms"] = statistics.median(
            [a + b for a, b in zip(ctx_ms, read_ms)]
        )
    return out


def main() -> int:
    paths = PathsConfig()
    res = load_processed(paths)
    base = PRESETS["ours"]
    kb = KnowledgeBase.load(res, base, paths)
    qs = res.questions[:N]
    # Reader with NO cache: we are measuring the reader, not the cache.
    reader = ExtractiveReader(base.reader)

    print(f"\nlatency, {SETTING} setting, n={N} questions, CPU\n")
    print(f"{'system':16s}{'cold ctx ms':>14s}{'cold p95':>11s}"
          f"{'warm ctx ms':>14s}{'read ms':>10s}{'cold total':>12s}")
    print("-" * 77)
    results = {}
    for name in SYSTEMS:
        cfg = PRESETS[name]
        kb.enable_score_cache(paths, cfg)         # warm pass uses the real cache
        warm = bench(kb, cfg, qs, reader, warm=True)
        cold = bench(kb, cfg, qs, reader, warm=False)
        kb.enable_score_cache(paths, cfg)
        results[name] = {"cold": cold, "warm": warm}
        print(f"{name:16s}{cold['ctx_median_ms']:14.0f}{cold['ctx_p95_ms']:11.0f}"
              f"{warm['ctx_median_ms']:14.1f}"
              f"{cold.get('read_median_ms', 0):10.0f}"
              f"{cold.get('total_median_ms', 0):12.0f}")

    out = paths.resolve("results_dir") / f"latency_{SETTING}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": N, "setting": SETTING, "systems": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\ncold = every cache bypassed (cost of a fresh query)")
    print("warm = caches populated (amortised cost inside a sweep or re-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
