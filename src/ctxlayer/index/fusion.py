"""Rank fusion for the hybrid retriever.

Why rank-space fusion rather than score interpolation
-----------------------------------------------------
BM25 scores are unbounded and corpus-dependent (they scale with idf and query
length); cosine similarities from a normalised bi-encoder live in [-1, 1] and
are heavily concentrated near 0.6-0.9. Any fixed ``a*bm25 + b*cosine`` therefore
needs per-corpus calibration and silently re-weights itself as the corpus
changes. Reciprocal Rank Fusion (Cormack et al., SIGIR 2009) depends only on
*rank*, needs no calibration, and is empirically hard to beat with a tuned
linear combination.

We use the weighted form::

    RRF(d) = sum_s  w_s / (k + rank_s(d))

``k`` (default 60, the value from the original paper) controls how sharply the
top of each list is favoured; higher ``k`` flattens the contribution so deeper
ranks still matter. Documents missing from a signal's candidate list simply
contribute nothing from that signal, which is the correct behaviour -- absence
is weak evidence, not negative evidence.

``linear`` fusion is implemented too, but only so the ablation in EVAL.md can
show *why* RRF was chosen rather than asserting it.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..config import FusionConfig
from ..schema import ScoredPassage

__all__ = ["fuse", "rrf", "minmax"]


def minmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalise a score map to [0, 1] (used by `linear` fusion)."""
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rrf(
    ranked_lists: Mapping[str, Sequence[tuple[str, float]]],
    weights: Mapping[str, float],
    k: int = 60,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, ranked in ranked_lists.items():
        w = weights.get(name, 1.0)
        if w == 0.0:
            continue
        for rank, (pid, _score) in enumerate(ranked, start=1):
            out[pid] = out.get(pid, 0.0) + w / (k + rank)
    return out


def fuse(
    cfg: FusionConfig,
    bm25_ranked: Sequence[tuple[str, float]] | None,
    dense_ranked: Sequence[tuple[str, float]] | None,
    top_k: int,
) -> list[ScoredPassage]:
    """Fuse signal-specific ranked lists into one ranking.

    Returns ``ScoredPassage`` objects carrying the per-signal contributions, so
    any row of the results table can be traced back to the signal that produced
    it (and so the "which signal found the gold passage?" analysis in EVAL.md is
    possible at all).
    """
    bm25_ranked = list(bm25_ranked or [])
    dense_ranked = list(dense_ranked or [])

    if cfg.method == "bm25_only":
        combined = {pid: s for pid, s in bm25_ranked}
    elif cfg.method == "dense_only":
        combined = {pid: s for pid, s in dense_ranked}
    elif cfg.method == "linear":
        nb = minmax(dict(bm25_ranked))
        nd = minmax(dict(dense_ranked))
        combined = {}
        for pid in set(nb) | set(nd):
            combined[pid] = cfg.w_bm25 * nb.get(pid, 0.0) + cfg.w_dense * nd.get(
                pid, 0.0
            )
    elif cfg.method == "rrf":
        combined = rrf(
            {"bm25": bm25_ranked, "dense": dense_ranked},
            {"bm25": cfg.w_bm25, "dense": cfg.w_dense},
            k=cfg.rrf_k,
        )
    else:
        raise ValueError(f"unknown fusion method: {cfg.method!r}")

    bm25_rank = {pid: i + 1 for i, (pid, _) in enumerate(bm25_ranked)}
    dense_rank = {pid: i + 1 for i, (pid, _) in enumerate(dense_ranked)}
    bm25_score = dict(bm25_ranked)
    dense_score = dict(dense_ranked)

    # Stable tie-break on passage_id keeps rankings byte-identical across runs.
    order = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[ScoredPassage] = []
    for pid, score in order[:top_k]:
        out.append(
            ScoredPassage(
                passage_id=pid,
                score=float(score),
                components={
                    "bm25": float(bm25_score.get(pid, 0.0)),
                    "dense": float(dense_score.get(pid, 0.0)),
                    "bm25_rank": float(bm25_rank.get(pid, 0)),
                    "dense_rank": float(dense_rank.get(pid, 0)),
                    "fused": float(score),
                },
                reason=cfg.method,
            )
        )
    return out
