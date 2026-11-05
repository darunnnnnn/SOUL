"""Metrics.

Retrieval metrics
-----------------
``support_recall@k``        fraction of gold supporting *passages* in the top k.
                            The standard MuSiQue support measure.
``support_all@k``           1 if **every** gold passage is in the top k. This is
                            the metric that actually predicts whether a
                            multi-hop question is answerable, and it is much
                            harsher than mean recall: on a 4-hop question,
                            recall 0.75 means the chain is broken and the answer
                            is unreachable. Mean recall hides that; this does
                            not.
``doc_coverage@k``          fraction of gold *documents* represented in top k.
``all_docs@k``             1 if all gold documents are represented. **This is
                            the Hard-Requirement-#2 metric**: it is defined over
                            distinct documents, so stacking several passages of
                            one article cannot inflate it.
``support_precision@k``     signal efficiency -- how much of the context earns
                            its tokens.
``ndcg@k``                  rank-sensitive quality with binary gold relevance.
``mrr``                     rank of the first gold passage.

Answer metrics
--------------
MuSiQue's official answer metrics are Exact Match and token-level F1 over the
normalised string, with answer aliases accepted. We reimplement the official
SQuAD/MuSiQue normalisation (lowercase, strip articles, strip punctuation,
collapse whitespace) rather than importing it, and verify our implementation
against the published reference behaviour in ``tests/test_metrics.py``.

Citation metrics
----------------
These exist because "has citations" is not the same as "is grounded":

``citation_validity``   do cited anchors resolve to real locations whose text is
                        byte-identical to the live source? Mechanical, and the
                        direct check of Hard Requirement #1.
``citation_precision``  of the anchors cited, how many are in gold supporting
                        passages.
``citation_recall``     of the gold supporting documents, how many are cited.
``multi_doc_citation``  did the answer cite >= 2 distinct documents? A system can
                        score well on answers while citing one source; this
                        catches it.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Mapping, Sequence

__all__ = [
    "normalize_answer",
    "exact_match",
    "token_f1",
    "best_over_aliases",
    "retrieval_metrics",
    "citation_metrics",
    "ndcg_at_k",
]


# --------------------------------------------------------------------------
# answer metrics (official MuSiQue / SQuAD normalisation)
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """Official SQuAD-style normalisation, as used by MuSiQue's scorer."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        # Matches the reference implementation: if either side is empty, F1 is
        # 1.0 only when both are empty.
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def best_over_aliases(pred: str, golds: Sequence[str]) -> tuple[float, float]:
    """Max EM / F1 over the gold answer and its aliases (official protocol)."""
    if not golds:
        return 0.0, 0.0
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(token_f1(pred, g) for g in golds)
    return em, f1


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------


def ndcg_at_k(ranked_ids: Sequence[str], gold: set[str], k: int) -> float:
    """nDCG@k with binary relevance.

    The ideal DCG uses ``min(len(gold), k)`` gold items, so a question with more
    gold passages than k is not penalised for the impossible.
    """
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    ranked_passage_ids: Sequence[str],
    gold_passage_ids: Iterable[str],
    gold_doc_ids: Iterable[str],
    passage_to_doc: Mapping[str, str],
    k_values: Sequence[int] = (2, 5, 10, 20),
) -> dict[str, float]:
    gold_p = set(gold_passage_ids)
    gold_d = set(gold_doc_ids)
    out: dict[str, float] = {}

    for k in k_values:
        topk = list(ranked_passage_ids[:k])
        hit_p = gold_p & set(topk)
        hit_d = gold_d & {passage_to_doc[p] for p in topk if p in passage_to_doc}
        out[f"support_recall@{k}"] = len(hit_p) / len(gold_p) if gold_p else 0.0
        out[f"support_all@{k}"] = float(len(hit_p) == len(gold_p)) if gold_p else 0.0
        out[f"support_precision@{k}"] = len(hit_p) / k if k else 0.0
        out[f"doc_coverage@{k}"] = len(hit_d) / len(gold_d) if gold_d else 0.0
        out[f"all_docs@{k}"] = float(len(hit_d) == len(gold_d)) if gold_d else 0.0
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_passage_ids, gold_p, k)

    rr = 0.0
    for i, pid in enumerate(ranked_passage_ids, start=1):
        if pid in gold_p:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    return out


def context_metrics(
    context_passage_ids: Sequence[str],
    gold_passage_ids: Iterable[str],
    gold_doc_ids: Iterable[str],
    passage_to_doc: Mapping[str, str],
    n_words: int,
) -> dict[str, float]:
    """Metrics on the *assembled context* -- what the reader actually sees.

    Distinct from retrieval@k: assembly applies a token budget, a per-document
    cap and diversity, so the set handed to the reader is not the top-k prefix.
    These are the numbers that bound answer quality.
    """
    gold_p = set(gold_passage_ids)
    gold_d = set(gold_doc_ids)
    ctx = list(context_passage_ids)
    hit_p = gold_p & set(ctx)
    ctx_docs = {passage_to_doc[p] for p in ctx if p in passage_to_doc}
    hit_d = gold_d & ctx_docs
    return {
        "ctx_support_recall": len(hit_p) / len(gold_p) if gold_p else 0.0,
        "ctx_support_all": float(len(hit_p) == len(gold_p)) if gold_p else 0.0,
        "ctx_doc_coverage": len(hit_d) / len(gold_d) if gold_d else 0.0,
        "ctx_all_docs": float(len(hit_d) == len(gold_d)) if gold_d else 0.0,
        "ctx_support_precision": len(hit_p) / len(ctx) if ctx else 0.0,
        "ctx_n_passages": float(len(ctx)),
        "ctx_n_docs": float(len(ctx_docs)),
        "ctx_words": float(n_words),
    }


# --------------------------------------------------------------------------
# citation metrics
# --------------------------------------------------------------------------


def citation_metrics(
    cited_anchor_ids: Sequence[str],
    anchor_resolves: Mapping[str, bool],
    anchor_to_passage: Mapping[str, str],
    anchor_to_doc: Mapping[str, str],
    gold_passage_ids: Iterable[str],
    gold_doc_ids: Iterable[str],
) -> dict[str, float]:
    gold_p = set(gold_passage_ids)
    gold_d = set(gold_doc_ids)
    cited = list(dict.fromkeys(cited_anchor_ids))
    if not cited:
        return {
            "citation_validity": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "multi_doc_citation": 0.0,
            "n_citations": 0.0,
            "n_cited_docs": 0.0,
            "has_citation": 0.0,
        }
    valid = [a for a in cited if anchor_resolves.get(a, False)]
    in_gold = [a for a in valid if anchor_to_passage.get(a) in gold_p]
    cited_docs = {anchor_to_doc[a] for a in valid if a in anchor_to_doc}
    return {
        "citation_validity": len(valid) / len(cited),
        "citation_precision": len(in_gold) / len(valid) if valid else 0.0,
        "citation_recall": len(cited_docs & gold_d) / len(gold_d) if gold_d else 0.0,
        "multi_doc_citation": float(len(cited_docs) >= 2),
        "n_citations": float(len(cited)),
        "n_cited_docs": float(len(cited_docs)),
        "has_citation": 1.0,
    }
