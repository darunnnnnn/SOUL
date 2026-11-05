"""Build and persist the knowledge base (ingest -> indexes -> disk)."""
import sys, json, time
sys.path.insert(0, "src")
import ctxlayer
from ctxlayer.config import IngestConfig, PathsConfig, PRESETS
from ctxlayer.ingest import ingest_musique, save_processed
from ctxlayer.index.store import KnowledgeBase

n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
paths = PathsConfig()
t = time.time()
ing = ingest_musique(IngestConfig(subset_size=n), paths)
save_processed(ing, paths)
print(f"ingested {ing.stats['n_questions']}q / {ing.stats['n_passages']}p "
      f"/ {ing.stats['n_documents']}d in {time.time()-t:.1f}s", flush=True)
kb = KnowledgeBase.build(ing, PRESETS["ours"], paths, verbose=True)
kb.save(paths, PRESETS["ours"])
print("\nindex stats:", json.dumps(kb.stats["index"], indent=2), flush=True)
