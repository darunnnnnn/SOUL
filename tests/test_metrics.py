"""Metric correctness, including agreement with the official MuSiQue scorer.

These are the numbers the whole submission rests on, so they are tested against
hand-computed values and against the reference normalisation behaviour rather
than against our own implementation's output.
"""

from __future__ import annotations

import math

import pytest

from ctxlayer.eval.metrics import (
    best_over_aliases,
    citation_metrics,
    context_metrics,
    exact_match,
    ndcg_at_k,
    normalize_answer,
    retrieval_metrics,
    token_f1,
)


# --------------------------------------------------------------------------
# answer normalisation / EM / F1  (official SQuAD + MuSiQue protocol)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The Beatles", "beatles"),
        ("a dog's life", "dogs life"),
        ("  Multiple   spaces  ", "multiple spaces"),
        ("U.S.A.", "usa"),
        ("An Apple, a Pear; the End!", "apple pear end"),
        ("1987", "1987"),
        ("", ""),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_exact_match_ignores_articles_punctuation_case():
    assert exact_match("The University of Glasgow", "university of glasgow") == 1.0
    assert exact_match("Glasgow", "university of glasgow") == 0.0


def test_token_f1_hand_computed():
    # pred tokens: {university, of, glasgow} ; gold: {university, of, edinburgh}
    # overlap = 2 -> p = 2/3, r = 2/3, f1 = 2/3
    assert token_f1("University of Glasgow", "University of Edinburgh") == pytest.approx(
        2 / 3
    )


def test_token_f1_edge_cases():
    # Both empty -> 1.0; one empty -> 0.0. This mirrors the reference scorer.
    assert token_f1("", "") == 1.0
    assert token_f1("the a an", "") == 1.0  # normalises to empty on both sides
    assert token_f1("something", "") == 0.0
    assert token_f1("", "something") == 0.0
    # No overlap at all
    assert token_f1("cats", "dogs") == 0.0


def test_f1_is_symmetric_and_em_implies_f1():
    a, b = "John Smith Jr", "John Smith"
    assert token_f1(a, b) == pytest.approx(token_f1(b, a))
    assert token_f1("John Smith", "john smith") == 1.0
    assert exact_match("John Smith", "john smith") == 1.0


def test_best_over_aliases_takes_the_max():
    em, f1 = best_over_aliases("NYC", ["New York City", "NYC", "New York"])
    assert em == 1.0 and f1 == 1.0
    em, f1 = best_over_aliases("Paris", ["London", "Berlin"])
    assert em == 0.0 and f1 == 0.0


def test_best_over_aliases_empty_gold():
    assert best_over_aliases("anything", []) == (0.0, 0.0)


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------

P2D = {
    "p1": "dA", "p2": "dA",   # two passages of ONE document
    "p3": "dB", "p4": "dC", "p5": "dD",
}


def test_multiple_passages_of_one_document_do_not_satisfy_doc_coverage():
    """Hard Requirement #2, as a test.

    Gold evidence spans documents dA and dB. A ranking that returns both
    passages of dA has support_recall 0.5 but must NOT show full document
    coverage -- several chunks of one source are not multi-document evidence.
    """
    m = retrieval_metrics(
        ["p1", "p2"], gold_passage_ids=["p1", "p3"], gold_doc_ids=["dA", "dB"],
        passage_to_doc=P2D, k_values=(2,),
    )
    assert m["support_recall@2"] == pytest.approx(0.5)
    assert m["doc_coverage@2"] == pytest.approx(0.5)
    assert m["all_docs@2"] == 0.0
    assert m["support_all@2"] == 0.0


def test_all_docs_and_support_all_when_complete():
    m = retrieval_metrics(
        ["p1", "p3"], ["p1", "p3"], ["dA", "dB"], P2D, k_values=(2, 5)
    )
    assert m["support_all@2"] == 1.0
    assert m["all_docs@2"] == 1.0
    assert m["support_recall@2"] == 1.0
    assert m["support_precision@2"] == 1.0
    assert m["mrr"] == 1.0


def test_support_all_is_stricter_than_recall():
    """A broken 4-hop chain scores 0.75 recall but 0 on support_all."""
    p2d = {f"p{i}": f"d{i}" for i in range(1, 6)}
    gold = ["p1", "p2", "p3", "p4"]
    m = retrieval_metrics(["p1", "p2", "p3", "p5"], gold, list(p2d[g] for g in gold), p2d, (4,))
    assert m["support_recall@4"] == pytest.approx(0.75)
    assert m["support_all@4"] == 0.0


def test_mrr_uses_first_gold_rank():
    m = retrieval_metrics(["p5", "p4", "p1"], ["p1"], ["dA"], P2D, (3,))
    assert m["mrr"] == pytest.approx(1 / 3)


def test_mrr_zero_when_no_gold_retrieved():
    m = retrieval_metrics(["p4", "p5"], ["p1"], ["dA"], P2D, (2,))
    assert m["mrr"] == 0.0


def test_ndcg_perfect_and_ordering_sensitive():
    assert ndcg_at_k(["p1", "p3"], {"p1", "p3"}, 2) == pytest.approx(1.0)
    first = ndcg_at_k(["p1", "p4"], {"p1"}, 2)
    second = ndcg_at_k(["p4", "p1"], {"p1"}, 2)
    assert first > second  # rank 1 beats rank 2
    assert second == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_capped_by_k():
    """More gold than k must not be penalised as impossible."""
    assert ndcg_at_k(["p1"], {"p1", "p2", "p3"}, 1) == pytest.approx(1.0)


def test_context_metrics_count_distinct_documents():
    m = context_metrics(
        ["p1", "p2"], ["p1", "p3"], ["dA", "dB"], P2D, n_words=100
    )
    assert m["ctx_n_docs"] == 1.0       # p1 and p2 are the same document
    assert m["ctx_n_passages"] == 2.0
    assert m["ctx_all_docs"] == 0.0
    assert m["ctx_words"] == 100.0


# --------------------------------------------------------------------------
# citation metrics
# --------------------------------------------------------------------------

A2P = {"a1": "p1", "a2": "p3", "a3": "p4"}
A2D = {"a1": "dA", "a2": "dB", "a3": "dC"}


def test_citation_metrics_all_valid_and_multi_doc():
    m = citation_metrics(
        ["a1", "a2"], {"a1": True, "a2": True}, A2P, A2D, ["p1", "p3"], ["dA", "dB"]
    )
    assert m["citation_validity"] == 1.0
    assert m["citation_precision"] == 1.0
    assert m["citation_recall"] == 1.0
    assert m["multi_doc_citation"] == 1.0


def test_unresolvable_citation_lowers_validity_not_precision_denominator():
    """A fabricated anchor must be caught by validity.

    Precision is computed over *valid* citations, so a fabrication cannot be
    laundered into a precision score -- it shows up as a validity failure.
    """
    m = citation_metrics(
        ["a1", "fake"], {"a1": True, "fake": False}, A2P, A2D, ["p1"], ["dA"]
    )
    assert m["citation_validity"] == pytest.approx(0.5)
    assert m["citation_precision"] == 1.0  # the one valid citation was correct
    assert m["multi_doc_citation"] == 0.0


def test_citation_precision_penalises_citing_a_non_gold_passage():
    m = citation_metrics(
        ["a1", "a3"], {"a1": True, "a3": True}, A2P, A2D, ["p1"], ["dA"]
    )
    assert m["citation_precision"] == pytest.approx(0.5)


def test_no_citations_scores_zero_everywhere():
    m = citation_metrics([], {}, A2P, A2D, ["p1"], ["dA"])
    assert m["has_citation"] == 0.0
    assert m["citation_validity"] == 0.0
    assert m["n_citations"] == 0.0


def test_duplicate_citations_counted_once():
    m = citation_metrics(
        ["a1", "a1"], {"a1": True}, A2P, A2D, ["p1"], ["dA"]
    )
    assert m["n_citations"] == 1.0
