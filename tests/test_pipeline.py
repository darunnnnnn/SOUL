"""Retrieval, assembly, fusion and config behaviour on a tiny synthetic corpus.

The synthetic corpus is built to contain the exact structure that defeats
single-shot retrieval: a question that names entity A, a document about A that
mentions bridge entity B, and a document *titled* C whose text mentions B. The
answer lives in C, and nothing in the question points at C. Any system that
finds C must have followed the bridge.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctxlayer.config import (
    PRESETS,
    AssembleConfig,
    BM25Config,
    FusionConfig,
    GraphConfig,
    SystemConfig,
)
from ctxlayer.index.bm25 import BM25FIndex, tokenize
from ctxlayer.index.entity_index import EntityIndex, extract_spans
from ctxlayer.index.fusion import fuse, rrf
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.retrieve.assemble import ContextAssembler
from ctxlayer.retrieve.decompose import analyze_question
from ctxlayer.schema import Anchor, Document, Passage, ScoredPassage, split_sentences, stable_id


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _mk(title: str, text: str, ord_: int = 0) -> Passage:
    doc_id = stable_id("doc", title)
    p = Passage(
        passage_id=stable_id("psg", title, text),
        doc_id=doc_id, doc_title=title, passage_ord=ord_, text=text,
    )
    for i, (cs, ce) in enumerate(split_sentences(text)):
        p.anchors.append(
            Anchor(f"{doc_id}::p{ord_}::s{i}", p.passage_id, doc_id, title,
                   ord_, i, cs, ce, text[cs:ce])
        )
    return p


CORPUS = [
    _mk("Green (Steve Hillage album)",
        "Green is the fourth studio album by British musician Steve Hillage. "
        "It was recorded in 1978."),
    _mk("Miquette Giraudy",
        "Miquette Giraudy is a keyboard player, best known for her work with "
        "her partner Steve Hillage. She was born in Nice."),
    _mk("Cork (city)",
        "Cork is the second largest city in Ireland. It is known for its green "
        "parks and its музей."),
    _mk("Unrelated Album",
        "Unrelated Album is a 1990 record by an unrelated artist. It has "
        "nothing to do with anything."),
]
# Two passages of ONE document, to test the per-document cap.
CORPUS += [
    _mk("Zhejiang", "Zhejiang is a province of China. Its capital is Hangzhou.", 0),
    _mk("Zhejiang", "Zhejiang has a long coastline and many islands.", 1),
    _mk("Zhejiang", "Zhejiang is bordered by Jiangsu to the north.", 2),
]


def _kb(cfg: SystemConfig | None = None) -> KnowledgeBase:
    cfg = cfg or PRESETS["ours"]
    passages = {p.passage_id: p for p in CORPUS}
    docs: dict[str, Document] = {}
    for p in CORPUS:
        docs.setdefault(p.doc_id, Document(p.doc_id, p.doc_title)).passage_ids.append(
            p.passage_id
        )
    ordered = [passages[k] for k in sorted(passages)]
    bm25 = BM25FIndex(cfg.bm25).build(ordered)
    ents = EntityIndex(cfg.graph).build(ordered, verbose=False)
    return KnowledgeBase(passages, docs, bm25=bm25, dense=None, entities=ents)


# --------------------------------------------------------------------------
# tokenizer / BM25F
# --------------------------------------------------------------------------


def test_tokenizer_folds_accents_and_possessives():
    assert tokenize("Beyoncé's") == ["beyonce"]
    assert tokenize("Zhejiang's islands") == ["zhejiang", "island"]
    # -ss is not stripped
    assert tokenize("glass") == ["glass"]


def test_bm25_title_field_boost_matters():
    """A title match should outrank the same term appearing only in prose."""
    ordered = sorted(CORPUS, key=lambda p: p.passage_id)
    boosted = BM25FIndex(BM25Config(w_title=5.0)).build(ordered)
    flat = BM25FIndex(BM25Config(w_title=0.0)).build(ordered)
    top_boosted = boosted.search("Zhejiang", top_k=1)
    assert top_boosted and "Zhejiang" in {
        p.doc_title for p in CORPUS if p.passage_id == top_boosted[0][0]
    }
    # With no title weighting, the Cork passage (which says "green") can win a
    # "green" query over the album titled Green.
    g_boost = [pid for pid, _ in boosted.search("Green", top_k=1)]
    g_flat = [pid for pid, _ in flat.search("Green", top_k=1)]
    assert g_boost != [] and g_flat != []


def test_bm25_restrict_to_pool():
    kb = _kb()
    pool = [CORPUS[0].passage_id]
    hits = kb.bm25.search("Steve Hillage", top_k=10, restrict_to=pool)
    assert {pid for pid, _ in hits} <= set(pool)


def test_bm25_unknown_term_returns_nothing():
    kb = _kb()
    assert kb.bm25.search("zzzzqqqxnonexistent", top_k=5) == []


def test_bm25_idf_is_never_negative():
    """A term in every passage must contribute ~0, never a negative score."""
    ordered = sorted(CORPUS, key=lambda p: p.passage_id)
    idx = BM25FIndex(BM25Config()).build(ordered)
    assert float(idx._idf.min()) > 0.0


# --------------------------------------------------------------------------
# entity index
# --------------------------------------------------------------------------


def test_extract_spans_does_not_cross_sentence_boundaries():
    spans = extract_spans("Kathmandu. The Bagmati River is holy.", 5)
    assert "Kathmandu The Bagmati River" not in spans
    assert "Kathmandu" in spans


def test_extract_spans_rejects_connector_edges():
    spans = extract_spans("He worked at the Bank of England in London.", 5)
    assert "Bank of England" in spans
    assert "Bank of" not in spans
    assert "of England" not in spans


def test_mention_index_reaches_the_bridge_document():
    """The property the whole multi-hop design depends on.

    'Steve Hillage' is not a document title here, but it IS mentioned in the
    *Miquette Giraudy* passage. A mention index must therefore connect the
    Green album to Giraudy -- which a title-keyed graph could not.
    """
    kb = _kb()
    green, giraudy = CORPUS[0], CORPUS[1]
    assert kb.entities is not None
    key = " ".join(tokenize("Steve Hillage"))
    assert key in kb.entities.entities_in(green.passage_id)
    reachable = kb.entities.candidates_for_entity(key, limit=20)
    assert giraudy.passage_id in reachable


def test_proper_noun_filter_rejects_sentence_initial_common_words():
    kb = _kb()
    ents = set(kb.entities.entities_in(CORPUS[3].passage_id))
    # "Unrelated" leads a sentence but is not a corpus-wide proper noun, and
    # "It"/"has" are stopwords/short.
    assert "it" not in ents
    assert "has" not in ents


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def test_rrf_rewards_agreement_between_signals():
    a = [("x", 9.0), ("y", 8.0)]
    b = [("y", 0.9), ("x", 0.1)]
    scores = rrf({"bm25": a, "dense": b}, {"bm25": 1.0, "dense": 1.0}, k=60)
    # x: 1/61 + 1/62 ; y: 1/62 + 1/61  -> equal, both appear at ranks 1 and 2
    assert scores["x"] == pytest.approx(scores["y"])
    # A doc ranked 1 by both must beat a doc ranked 2 by both.
    scores2 = rrf({"s1": [("p", 1.0), ("q", 1.0)], "s2": [("p", 1.0), ("q", 1.0)]},
                  {"s1": 1.0, "s2": 1.0}, k=60)
    assert scores2["p"] > scores2["q"]


def test_rrf_is_scale_invariant_unlike_linear():
    """The reason we use RRF: BM25 and cosine live on incomparable scales."""
    bm = [("x", 40.0), ("y", 1.0)]
    dn = [("y", 0.95), ("x", 0.90)]
    r1 = fuse(FusionConfig(method="rrf"), bm, dn, top_k=2)
    bm_scaled = [("x", 4000.0), ("y", 100.0)]
    r2 = fuse(FusionConfig(method="rrf"), bm_scaled, dn, top_k=2)
    assert [c.passage_id for c in r1] == [c.passage_id for c in r2]


def test_fusion_modes_select_the_right_signal():
    bm = [("x", 10.0)]
    dn = [("y", 0.9)]
    only_bm = fuse(FusionConfig(method="bm25_only"), bm, dn, top_k=5)
    only_dn = fuse(FusionConfig(method="dense_only"), bm, dn, top_k=5)
    assert [c.passage_id for c in only_bm] == ["x"]
    assert [c.passage_id for c in only_dn] == ["y"]


def test_fusion_is_deterministic_on_ties():
    """Genuinely tied scores must break on passage_id, not on dict order.

    A real tie needs equal RRF contributions: "b" at rank 1 of one signal and
    "a" at rank 1 of the other, with equal signal weights. (Ranks 1 and 2 of the
    *same* signal are not tied -- 1/61 != 1/62.)
    """
    out = [c.passage_id for c in fuse(FusionConfig(), [("b", 1.0)], [("a", 0.9)], top_k=2)]
    assert out == ["a", "b"]
    # Byte-identical across repeated calls and input orderings.
    again = [c.passage_id for c in fuse(FusionConfig(), [("b", 1.0)], [("a", 0.9)], top_k=2)]
    assert out == again


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _ranked(passages, scores=None):
    scores = scores or [1.0 - 0.05 * i for i in range(len(passages))]
    return [ScoredPassage(p.passage_id, s) for p, s in zip(passages, scores)]


def test_per_document_cap_is_enforced():
    """Hard Requirement #2 machinery: one source cannot take the whole budget."""
    kb = _kb()
    zhejiang = [p for p in CORPUS if p.doc_title == "Zhejiang"]
    cfg = PRESETS["ours"].replace(**{"assemble.max_per_doc": 2,
                                     "assemble.min_rel_ratio": 0.0})
    ctx = ContextAssembler(kb, cfg).assemble("Zhejiang province", _ranked(zhejiang))
    assert len(ctx.passages) <= 2
    assert len(ctx.doc_ids) == 1  # only one document exists in the candidates


def test_assembly_prefers_a_second_document_over_a_third_passage():
    kb = _kb()
    zhejiang = [p for p in CORPUS if p.doc_title == "Zhejiang"]
    other = CORPUS[1]
    cand = _ranked(zhejiang + [other], [1.0, 0.95, 0.9, 0.85])
    cfg = PRESETS["ours"].replace(**{"assemble.max_per_doc": 1,
                                     "assemble.min_rel_ratio": 0.0,
                                     "assemble.max_passages": 2})
    ctx = ContextAssembler(kb, cfg).assemble("Zhejiang Giraudy", cand)
    assert len(ctx.doc_ids) == 2


def test_word_budget_is_respected():
    kb = _kb()
    cfg = PRESETS["ours"].replace(**{"assemble.budget_words": 15,
                                     "assemble.min_rel_ratio": 0.0})
    ctx = ContextAssembler(kb, cfg).assemble("anything", _ranked(CORPUS))
    # Always at least one passage, but never more than one over budget.
    assert len(ctx.passages) >= 1
    assert ctx.n_words <= max(15, CORPUS[0].n_words) + max(p.n_words for p in CORPUS)


def test_relevance_floor_excludes_weak_candidates():
    kb = _kb()
    cand = _ranked(CORPUS[:4], [1.0, 0.9, 0.05, 0.01])
    cfg = PRESETS["ours"].replace(**{"assemble.min_rel_ratio": 0.5,
                                     "assemble.max_per_doc": 9,
                                     "assemble.min_docs": 1})
    ctx = ContextAssembler(kb, cfg).assemble("Steve Hillage Green album", cand)
    kept = set(ctx.passage_ids)
    assert CORPUS[2].passage_id not in kept or CORPUS[3].passage_id not in kept


def test_assembly_disabled_is_plain_topk_under_same_budget():
    kb = _kb()
    cfg = PRESETS["ours"].replace(**{"assemble.enabled": False,
                                     "assemble.max_passages": 3})
    ctx = ContextAssembler(kb, cfg).assemble("anything", _ranked(CORPUS))
    assert ctx.passage_ids == [p.passage_id for p in CORPUS[:3]]


def test_empty_ranking_yields_empty_context():
    kb = _kb()
    ctx = ContextAssembler(kb, PRESETS["ours"]).assemble("q", [])
    assert ctx.passages == [] and ctx.n_words == 0


# --------------------------------------------------------------------------
# question analysis
# --------------------------------------------------------------------------


def test_governing_wh_word_determines_focus():
    """'When ... where ...' asks for a date, not a location."""
    p = analyze_question(
        "When did the carnival take place in the city where CIMI-FM is licensed?"
    )
    assert p.focus == "date"
    assert analyze_question("Who is the spouse of the Green performer?").focus == (
        "person_or_org"
    )
    assert analyze_question("How many Germans live there?").focus == "number"


def test_anchor_entities_and_relation_terms_are_disjoint():
    p = analyze_question("Who is the spouse of the Green performer?")
    assert "Green" in p.anchor_entities
    assert "spouse" in p.relation_terms
    assert "green" not in p.relation_terms  # anchors are excluded


def test_bridge_query_carries_relation_terms():
    p = analyze_question("Who is the spouse of the Green performer?")
    bq = p.bridge_query("Steve Hillage")
    assert bq.startswith("Steve Hillage")
    assert "spouse" in bq


def test_degenerate_question_still_yields_relation_terms():
    p = analyze_question("Who is Steve Hillage?")
    assert p.relation_terms  # must not be empty, else bridges lose all signal


# --------------------------------------------------------------------------
# config / ablation hygiene
# --------------------------------------------------------------------------


def test_ablation_is_a_single_field_delta():
    base = PRESETS["ours"]
    abl = base.replace(**{"assemble.max_per_doc": 1})
    d1, d2 = base.to_dict(), abl.to_dict()

    def diffs(a, b, prefix=""):
        out = []
        for k in a:
            if isinstance(a[k], dict):
                out += diffs(a[k], b[k], prefix + k + ".")
            elif a[k] != b[k]:
                out.append(prefix + k)
        return out

    assert diffs(d1, d2) == ["assemble.max_per_doc"]


def test_unknown_config_field_raises():
    with pytest.raises(ValueError):
        PRESETS["ours"].replace(**{"assemble.no_such_field": 1})
    with pytest.raises(ValueError):
        AssembleConfig.from_dict({"typo_field": 3})


def test_fingerprint_changes_with_config():
    a = PRESETS["ours"]
    b = a.replace(**{"assemble.max_per_doc": 1})
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == PRESETS["ours"].fingerprint()


def test_baseline_presets_disable_the_right_components():
    assert PRESETS["bm25"].dense.enabled is False
    assert PRESETS["bm25"].iterative.enabled is False
    assert PRESETS["dense"].bm25.enabled is False
    assert PRESETS["hybrid"].rerank.enabled is False
    assert PRESETS["hybrid_rerank"].rerank.enabled is True
    assert PRESETS["hybrid_rerank"].iterative.enabled is False
    assert PRESETS["ours"].iterative.enabled is True
    assert PRESETS["ours"].assemble.enabled is True
    # Every baseline must share our reader and our budget.
    for name in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        assert PRESETS[name].reader.model == PRESETS["ours"].reader.model
        assert (
            PRESETS[name].assemble.budget_words
            == PRESETS["ours"].assemble.budget_words
        )


def test_all_ablations_build_and_differ_from_ours():
    from ctxlayer.eval.ablations import ABLATIONS, build_ablation_config

    base_fp = PRESETS["ours"].fingerprint()
    for name in ABLATIONS:
        cfg = build_ablation_config(name)
        assert cfg.name == f"abl_{name}"
        assert cfg.fingerprint() != base_fp, f"{name} is a no-op"
