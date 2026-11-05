"""Compare reader modes on the ORACLE condition (gold paragraphs only).

The oracle condition isolates the reader: the context is exactly the gold
evidence, so any failure is a reading failure. This probe decides the reading
protocol on evidence rather than preference.
"""
import sys, time, statistics
import ctxlayer
from ctxlayer.config import PRESETS, PathsConfig, EvalConfig
from ctxlayer.ingest import load_processed
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.pipeline import ContextLayer
from ctxlayer.synth.reader import ExtractiveReader
from ctxlayer.eval.metrics import best_over_aliases

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
paths = PathsConfig()
res = load_processed(paths)
kb = KnowledgeBase.load(res, PRESETS["ours"], paths)
qs = res.questions[:N]

for system in ("oracle", "ours"):
    for mode in ("per_passage", "concat"):
        cfg = PRESETS[system].replace(**{"reader.mode": mode})
        layer = ContextLayer(kb, cfg)
        reader = ExtractiveReader(cfg.reader)
        ems, f1s = [], []
        t = time.time()
        for q in qs:
            ctx, _ = layer.build_context_for(q, setting="unified")
            a = reader.read(q.question, ctx)
            em, f1 = best_over_aliases(a.answer, q.all_answers())
            ems.append(em); f1s.append(f1)
        print(f"{system:8s} {mode:12s} EM={statistics.mean(ems)*100:5.1f} "
              f"F1={statistics.mean(f1s)*100:5.1f}  ({time.time()-t:.0f}s)", flush=True)
