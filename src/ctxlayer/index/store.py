"""``KnowledgeBase``: the single object every retriever is given.

It owns the corpus, the three indexes (BM25F, dense, entity mentions), and the
anchor table used to resolve citations. Retrievers are stateless functions over
it, which is what lets the eval runner build the knowledge base **once** and
then evaluate seven systems against the identical artefact -- no chance of two
systems accidentally seeing differently-built indexes.

Both retrieval settings are served from this one object:

``unified``  search the whole knowledge base (our headline setting)
``pool``     search restricted to a question's own 20-paragraph candidate pool
             (the standard MuSiQue distractor setting, for comparability with
             published numbers)

The setting is a *per-query argument*, not a separate build, so the two numbers
in the results table are guaranteed to come from the same index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PathsConfig, SystemConfig
from ..ingest.musique import IngestResult
from ..schema import Anchor, Document, Passage
from .bm25 import BM25FIndex
from .dense import DenseIndex
from .entity_index import EntityIndex
from .rerank import CrossEncoderReranker
from .score_cache import ScoreCache

__all__ = ["KnowledgeBase"]


class KnowledgeBase:
    def __init__(
        self,
        passages: dict[str, Passage],
        documents: dict[str, Document],
        bm25: BM25FIndex | None = None,
        dense: DenseIndex | None = None,
        entities: EntityIndex | None = None,
        stats: dict | None = None,
    ) -> None:
        self.passages = passages
        self.documents = documents
        self.bm25 = bm25
        self.dense = dense
        self.entities = entities
        self.stats = stats or {}
        self.anchors: dict[str, Anchor] = {
            a.anchor_id: a for p in passages.values() for a in p.anchors
        }
        self._rerankers: dict[str, CrossEncoderReranker] = {}
        self._score_cache: ScoreCache | None = None
        self._cache_path: Path | None = None

    # ----------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        ingest: IngestResult,
        cfg: SystemConfig,
        paths: PathsConfig,
        verbose: bool = True,
    ) -> "KnowledgeBase":
        """Build every index. Dense embeddings are cached on disk by content.

        All three indexes are built regardless of which are *enabled* for a
        given system: the cost is small, and building once means every baseline
        and ablation shares byte-identical indexes.
        """
        ordered = [ingest.passages[p] for p in sorted(ingest.passages)]
        timings: dict[str, float] = {}

        t = time.time()
        bm25 = BM25FIndex(cfg.bm25).build(ordered)
        timings["bm25_s"] = round(time.time() - t, 2)
        if verbose:
            print(f"  bm25F: {bm25.n_docs} passages, {len(bm25._vocab)} terms "
                  f"({timings['bm25_s']}s)")

        t = time.time()
        dense = DenseIndex(cfg.dense).build(
            ordered, cache_dir=paths.resolve("cache_dir"), verbose=verbose
        )
        timings["dense_s"] = round(time.time() - t, 2)

        t = time.time()
        entities = EntityIndex(cfg.graph).build(ordered, verbose=verbose)
        timings["entities_s"] = round(time.time() - t, 2)
        if verbose:
            print(f"  entities: {entities.stats.get('n_entity_keys')} keys "
                  f"({timings['entities_s']}s)")

        stats = dict(ingest.stats)
        stats["index"] = {
            "bm25_terms": len(bm25._vocab),
            "dense_dim": int(dense.emb.shape[1]) if dense.emb.size else 0,
            "dense_device": dense.device,
            **entities.stats,
            "build_timings": timings,
        }
        return cls(ingest.passages, ingest.documents, bm25, dense, entities, stats)

    # ------------------------------------------------------------- accessors
    def get(self, passage_id: str) -> Passage:
        return self.passages[passage_id]

    def many(self, passage_ids: Iterable[str]) -> list[Passage]:
        return [self.passages[p] for p in passage_ids if p in self.passages]

    def enable_score_cache(self, paths: PathsConfig, cfg: SystemConfig) -> None:
        """Attach the persistent cross-encoder cache (see score_cache.py)."""
        self._cache_path = paths.resolve("cache_dir") / "rerank_scores.sqlite"
        self._score_cache = ScoreCache(
            self._cache_path, cfg.rerank.model, cfg.rerank.max_length
        )
        self._rerankers.clear()

    def cache_stats(self) -> dict[str, float]:
        if self._score_cache is None:
            return {}
        out = dict(self._score_cache.stats())
        out["n_cross_encoder_calls"] = float(
            sum(r.n_scored for r in self._rerankers.values())
        )
        return out

    def reranker(self, cfg: SystemConfig) -> CrossEncoderReranker:
        key = f"{cfg.rerank.model}:{cfg.rerank.max_length}:{cfg.rerank.top_k_in}"
        if key not in self._rerankers:
            cache = self._score_cache
            if cache is not None and (
                cache.model != cfg.rerank.model
                or cache.max_length != cfg.rerank.max_length
            ):
                # A differently-configured reranker must not read this cache.
                cache = ScoreCache(self._cache_path, cfg.rerank.model,
                                   cfg.rerank.max_length)
            self._rerankers[key] = CrossEncoderReranker(cfg.rerank, cache=cache)
        return self._rerankers[key]

    def resolve_anchor(self, anchor_id: str) -> Anchor | None:
        return self.anchors.get(anchor_id)

    def verify_anchor(self, anchor_id: str) -> bool:
        """Re-resolve an anchor's char span against the live passage text.

        This is the mechanical check behind our provenance claim: a citation is
        only valid if the exact substring it points at is still there.
        """
        a = self.anchors.get(anchor_id)
        if a is None:
            return False
        p = self.passages.get(a.passage_id)
        if p is None:
            return False
        return p.text[a.char_start : a.char_end] == a.text

    @property
    def n_passages(self) -> int:
        return len(self.passages)

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    # ----------------------------------------------------------- persistence
    def save(self, paths: PathsConfig, cfg: SystemConfig) -> Path:
        out = paths.resolve("index_dir")
        out.mkdir(parents=True, exist_ok=True)
        if self.bm25 is not None:
            self.bm25.save(out / "bm25f.pkl")
        if self.dense is not None:
            self.dense.save(out / "dense.npz")
        if self.entities is not None:
            self.entities.save(out / "entities.pkl")
        (out / "index_manifest.json").write_text(
            json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    @classmethod
    def load(
        cls, ingest: IngestResult, cfg: SystemConfig, paths: PathsConfig
    ) -> "KnowledgeBase":
        out = paths.resolve("index_dir")
        man = out / "index_manifest.json"
        if not man.exists():
            raise FileNotFoundError(
                f"No index at {out}. Run:  python -m ctxlayer.cli index"
            )
        bm25 = BM25FIndex.load(out / "bm25f.pkl")
        dense = DenseIndex.load(out / "dense.npz", cfg.dense)
        entities = EntityIndex.load(out / "entities.pkl")
        stats = json.loads(man.read_text(encoding="utf-8"))
        kb = cls(ingest.passages, ingest.documents, bm25, dense, entities, stats)
        # Guard against evaluating against an index built from a different
        # subset -- a silent version skew would invalidate every number.
        if set(bm25.passage_ids) != set(ingest.passages):
            raise RuntimeError(
                "Index/corpus mismatch: the index was built from a different "
                "subset. Rebuild with:  python -m ctxlayer.cli index --force"
            )
        return kb
