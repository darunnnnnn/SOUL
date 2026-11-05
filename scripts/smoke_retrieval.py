"""Quick retrieval-only comparison on a small sample (development smoke test)."""
import sys, time, collections, statistics
sys.path.insert(0, "src")
import ctxlayer
from ctxlayer.config import IngestConfig, PathsConfig, PRESETS, EvalConfig
from ctxlayer.ingest import ingest_musique
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.pipeline import ContextLayer
from ctxlayer.eval.metrics import retrieval_metrics, context_metrics

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SYSTEMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bm25","dense","hybrid","hybrid_rerank","ours"]
SETTING = sys.argv[3] if len(sys.argv) > 3 else "unified"

paths = PathsConfig()
ing = ingest_musique(IngestConfig(subset_size=500), paths)
kb = KnowledgeBase.build(ing, PRESETS["ours"], paths, verbose=True)
p2d = {p.passage_id: p.doc_id for p in kb.passages.values()}
qs = ing.questions[:N]
print(f"\ncorpus={kb.n_passages} passages / {kb.n_documents} docs | {len(qs)} questions | setting={SETTING}\n")

rows = {}
for name in SYSTEMS:
    cfg = PRESETS[name]
    cl = ContextLayer(kb, cfg)
    agg = collections.defaultdict(list); t0 = time.time()
    for q in qs:
        ctx, ranked = cl.build_context_for(q, setting=SETTING)
        rids = [s.passage_id for s in ranked]
        m = retrieval_metrics(rids, q.gold_passage_ids, q.gold_doc_ids, p2d, (2,5,10,20))
        m.update(context_metrics(ctx.passage_ids, q.gold_passage_ids, q.gold_doc_ids, p2d, ctx.n_words))
        for k,v in m.items(): agg[k].append(v)
    dt = time.time()-t0
    rows[name] = {k: statistics.mean(v) for k,v in agg.items()}
    rows[name]["sec_per_q"] = dt/len(qs)
    print(f"{name:16s} done in {dt:.1f}s ({dt/len(qs):.2f}s/q)")

cols = ["support_recall@10","support_all@10","all_docs@10","ctx_support_recall","ctx_support_all","ctx_all_docs","ctx_n_docs","ctx_words","sec_per_q"]
print("\n" + "="*132)
print(f"{'system':16s}" + "".join(f"{c[-17:]:>14s}" for c in cols))
print("-"*132)
for name in SYSTEMS:
    print(f"{name:16s}" + "".join(f"{rows[name][c]:14.3f}" for c in cols))
print("="*132)
