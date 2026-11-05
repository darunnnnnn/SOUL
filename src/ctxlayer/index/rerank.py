"""Cross-encoder reranking.

Model choice
------------
``cross-encoder/ms-marco-MiniLM-L-6-v2``: 22M params, 90 MB, trained on MS
MARCO passage ranking. A cross-encoder reads the (query, passage) pair jointly,
so unlike a bi-encoder it can judge whether the passage answers *this* query
rather than whether it is topically near it. On BEIR-style benchmarks reranking
the top ~50 of a hybrid list is worth several points of nDCG@10, which is why
"hybrid + cross-encoder rerank" is the right *strong* baseline to beat rather
than a straw-man BM25.

It is shared identically by our system and the ``hybrid_rerank`` baseline, so it
cannot be the source of any measured gain.

Scores are raw logits (higher = more relevant); they are not probabilities and
are not comparable across queries. Everything downstream either ranks with them
or normalises per query, never compares them across questions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import RerankConfig
from ..schema import Passage, ScoredPassage
from .dense import resolve_device
from .score_cache import ScoreCache

__all__ = ["CrossEncoderReranker"]


class CrossEncoderReranker:
    _cache: dict[tuple[str, str], tuple[object, object]] = {}

    def __init__(self, cfg: RerankConfig, cache: "ScoreCache | None" = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.cache = cache
        self.n_scored = 0
        self._tok = None
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        key = (self.cfg.model, self.device)
        if key not in self._cache:
            tok = AutoTokenizer.from_pretrained(self.cfg.model)
            model = (
                AutoModelForSequenceClassification.from_pretrained(self.cfg.model)
                .to(self.device)
                .eval()
            )
            for p in model.parameters():
                p.requires_grad_(False)
            self._cache[key] = (tok, model)
        self._tok, self._model = self._cache[key]
        self._torch = torch

    def score(self, query: str, passages: Sequence[Passage]) -> np.ndarray:
        """Relevance logits for (query, passage) pairs, cache-aware.

        Only cache misses reach the model. Pairs are then batched in
        length-sorted order, because with dynamic padding a batch costs its
        longest member.
        """
        if not passages:
            return np.zeros(0, dtype=np.float32)
        texts = [p.with_title() for p in passages]
        scores: list[float | None] = [None] * len(texts)

        if self.cache is not None and self.cache.enabled:
            keys = [self.cache.key(query, t) for t in texts]
            found = self.cache.get_many(list(dict.fromkeys(keys)))
            for i, k in enumerate(keys):
                if k in found:
                    scores[i] = found[k]
            todo = [i for i, s in enumerate(scores) if s is None]
        else:
            keys = []
            todo = list(range(len(texts)))

        if todo:
            self._ensure()
            torch = self._torch
            order = sorted(todo, key=lambda i: len(texts[i]))
            with torch.inference_mode():
                for i in range(0, len(order), self.cfg.batch_size):
                    idx = order[i : i + self.cfg.batch_size]
                    enc = self._tok(  # type: ignore[misc]
                        [query] * len(idx),
                        [texts[j] for j in idx],
                        padding=True,
                        truncation=True,
                        max_length=self.cfg.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self._model(**enc).logits  # type: ignore[misc]
                    # ms-marco cross-encoders are single-logit regressors; guard
                    # against a 2-logit checkpoint being swapped in by config.
                    if logits.shape[-1] == 1:
                        vals = logits[:, 0]
                    else:
                        vals = logits[:, -1] - logits[:, 0]
                    arr = vals.float().cpu().numpy()
                    for row, j in enumerate(idx):
                        scores[j] = float(arr[row])
            self.n_scored += len(todo)
            if self.cache is not None and self.cache.enabled and keys:
                self.cache.put_many(
                    {keys[j]: float(scores[j]) for j in todo}  # type: ignore[arg-type]
                )

        return np.array([s if s is not None else 0.0 for s in scores], dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredPassage],
        passage_lookup: dict[str, Passage],
        top_k: int | None = None,
        depth: int | None = None,
    ) -> list[ScoredPassage]:
        """Rerank the top ``cfg.top_k_in`` candidates, keeping the tail order.

        Candidates beyond ``top_k_in`` are appended *below* every reranked item
        rather than discarded, so increasing ``top_k_in`` can only change the
        head of the ranking. That keeps the rerank-depth ablation interpretable.
        """
        if not candidates:
            return []
        d = depth if depth is not None else self.cfg.top_k_in
        head = list(candidates[:d])
        tail = list(candidates[d:])
        passages = [passage_lookup[c.passage_id] for c in head]
        scores = self.score(query, passages)

        for c, s in zip(head, scores):
            c.components["rerank"] = float(s)
            c.components["pre_rerank_score"] = float(c.score)
            c.score = float(s)
            c.reason = (c.reason + "+rerank").lstrip("+")
        head.sort(key=lambda c: (-c.score, c.passage_id))

        # Push the un-reranked tail strictly below the reranked head.
        if tail:
            floor = min((c.score for c in head), default=0.0)
            for j, c in enumerate(tail):
                c.components["rerank"] = float("nan")
                c.score = floor - 1.0 - j * 1e-6
        merged = head + tail
        return merged[:top_k] if top_k else merged
