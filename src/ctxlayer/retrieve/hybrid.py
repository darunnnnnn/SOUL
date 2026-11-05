"""Single-shot hybrid retrieval: the shared primitive and the baselines.

``SingleShotRetriever`` implements *all* of our retrieval baselines -- BM25F
only, dense only, hybrid RRF, and hybrid + cross-encoder -- selected purely by
config. One code path for every baseline means a baseline cannot be
accidentally crippled relative to our system: they execute the same fusion,
the same candidate depth and the same reranker.

It is also the primitive our iterative retriever calls once per round, so
improvements in the shared machinery help the baselines exactly as much as they
help us.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from ..config import SystemConfig
from ..index.fusion import fuse
from ..index.store import KnowledgeBase
from ..schema import RetrievalTrace, ScoredPassage

__all__ = ["SingleShotRetriever"]


class SingleShotRetriever:
    def __init__(self, kb: KnowledgeBase, cfg: SystemConfig) -> None:
        self.kb = kb
        self.cfg = cfg

    def _signals(
        self, query: str, restrict_to: Sequence[str] | None
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        cfg = self.cfg
        depth = cfg.fusion.candidates_per_signal
        bm25_ranked: list[tuple[str, float]] = []
        dense_ranked: list[tuple[str, float]] = []
        if cfg.bm25.enabled and self.kb.bm25 is not None:
            bm25_ranked = self.kb.bm25.search(query, top_k=depth, restrict_to=restrict_to)
        if cfg.dense.enabled and self.kb.dense is not None:
            dense_ranked = self.kb.dense.search(
                query, top_k=depth, restrict_to=restrict_to
            )
        return bm25_ranked, dense_ranked

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        restrict_to: Sequence[str] | None = None,
        rerank_query: str | None = None,
        rerank_depth: int | None = None,
        trace: RetrievalTrace | None = None,
    ) -> list[ScoredPassage]:
        """Fused (optionally reranked) ranking for one query.

        ``rerank_query`` lets a caller rerank against a different string than
        the one used for recall -- the iterative retriever needs this, because a
        hop-2 passage must be judged against the hop-2 bridge query, not against
        the original question it barely overlaps with.
        """
        cfg = self.cfg
        t0 = time.time()
        bm25_ranked, dense_ranked = self._signals(query, restrict_to)
        t_recall = (time.time() - t0) * 1000

        # Fuse deeper than top_k when a reranker will consume the list: recall
        # at the rerank depth is what bounds final quality.
        depth = rerank_depth if rerank_depth is not None else cfg.rerank.top_k_in
        fuse_k = max(top_k, depth) if cfg.rerank.enabled else top_k
        candidates = fuse(cfg.fusion, bm25_ranked, dense_ranked, top_k=fuse_k)

        t1 = time.time()
        if cfg.rerank.enabled and candidates:
            reranker = self.kb.reranker(cfg)
            candidates = reranker.rerank(
                rerank_query or query,
                candidates,
                self.kb.passages,
                depth=depth,
            )
        t_rerank = (time.time() - t1) * 1000

        if trace is not None:
            trace.queries_issued.append(query)
            trace.timings_ms["recall_ms"] = trace.timings_ms.get("recall_ms", 0.0) + t_recall
            trace.timings_ms["rerank_ms"] = trace.timings_ms.get("rerank_ms", 0.0) + t_rerank
        return candidates[:top_k]
