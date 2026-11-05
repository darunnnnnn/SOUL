"""Provenance invariants -- Hard Requirement #1, as executable checks.

The claim "every factual answer claim traces to a specific source and location"
is only as good as the machinery that enforces it. These tests cover the
properties that machinery must have:

* anchors re-resolve to byte-identical source text at their recorded offsets;
* anchor ids are stable across rebuilds (otherwise a committed citation rots);
* a fabricated or tampered citation is detected, not accepted;
* a claim whose numbers disagree with its source is rejected even when the
  wording matches.
"""

from __future__ import annotations

import pytest

from ctxlayer.index.store import KnowledgeBase
from ctxlayer.schema import (
    Anchor,
    AssembledContext,
    Answer,
    Claim,
    Document,
    Passage,
    split_sentences,
    stable_id,
)
from ctxlayer.synth.cite import verify_citations


# --------------------------------------------------------------------------
# sentence splitting / anchor minting
# --------------------------------------------------------------------------


def _mint(title: str, text: str, ord_: int = 0) -> Passage:
    doc_id = stable_id("doc", title)
    p = Passage(
        passage_id=stable_id("psg", title, text),
        doc_id=doc_id,
        doc_title=title,
        passage_ord=ord_,
        text=text,
    )
    for i, (cs, ce) in enumerate(split_sentences(text)):
        p.anchors.append(
            Anchor(
                anchor_id=f"{doc_id}::p{ord_}::s{i}",
                passage_id=p.passage_id,
                doc_id=doc_id,
                doc_title=title,
                passage_ord=ord_,
                sent_idx=i,
                char_start=cs,
                char_end=ce,
                text=text[cs:ce],
            )
        )
    return p


def _kb(*passages: Passage) -> KnowledgeBase:
    docs = {}
    for p in passages:
        docs.setdefault(p.doc_id, Document(p.doc_id, p.doc_title)).passage_ids.append(
            p.passage_id
        )
    return KnowledgeBase({p.passage_id: p for p in passages}, docs)


TEXT = (
    "Miquette Giraudy was born on 9 February 1953 in Nice, France. "
    "She is best known for her work with her partner Steve Hillage. "
    "They formed the ambient band System 7 in 1989."
)


def test_split_sentences_spans_reconstruct_the_source():
    spans = split_sentences(TEXT)
    assert len(spans) == 3
    for s, e in spans:
        assert 0 <= s < e <= len(TEXT)
    # Spans must be ordered and non-overlapping.
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 <= s2


def test_split_sentences_protects_abbreviations_and_initials():
    t = "Dr. Smith met Mr. Jones. They discussed J.R.R. Tolkien at 3 p.m."
    spans = split_sentences(t)
    assert len(spans) == 2
    assert t[spans[0][0] : spans[0][1]] == "Dr. Smith met Mr. Jones."


def test_split_sentences_handles_degenerate_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("No terminal punctuation") == [(0, 23)]


def test_every_anchor_reresolves_to_identical_text():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    assert p.anchors
    for a in p.anchors:
        assert kb.verify_anchor(a.anchor_id) is True
        assert p.text[a.char_start : a.char_end] == a.text


def test_anchor_ids_are_stable_across_rebuilds():
    """Content-addressed ids must not depend on build order or object identity.

    If ids moved between rebuilds, every previously-committed citation would
    silently point somewhere else.
    """
    a = _mint("Miquette Giraudy", TEXT)
    b = _mint("Miquette Giraudy", TEXT)
    assert a.passage_id == b.passage_id
    assert [x.anchor_id for x in a.anchors] == [x.anchor_id for x in b.anchors]
    # ... and differ for different content
    c = _mint("Miquette Giraudy", TEXT.replace("1953", "1954"))
    assert c.passage_id != a.passage_id


def test_unknown_anchor_is_not_verifiable():
    kb = _kb(_mint("Miquette Giraudy", TEXT))
    assert kb.verify_anchor("deadbeef::p0::s0") is False
    assert kb.resolve_anchor("deadbeef::p0::s0") is None


def test_tampered_offsets_are_detected():
    """A citation whose offsets no longer match its text must fail L1."""
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    good = p.anchors[0]
    bad = Anchor(
        anchor_id="tampered::p0::s0",
        passage_id=p.passage_id,
        doc_id=p.doc_id,
        doc_title=p.doc_title,
        passage_ord=0,
        sent_idx=0,
        char_start=good.char_start,
        char_end=good.char_end,
        text="Something that is not in the source at all.",
    )
    kb.anchors[bad.anchor_id] = bad
    assert kb.verify_anchor(bad.anchor_id) is False


# --------------------------------------------------------------------------
# citation verification
# --------------------------------------------------------------------------


def _answer_with(kb: KnowledgeBase, p: Passage, text: str, anchor_ids):
    ctx = AssembledContext(passages=[p], scores=[1.0])
    return Answer(answer=text, claims=[Claim(text=text, anchor_ids=list(anchor_ids))],
                  context=ctx)


def test_supported_claim_passes_verification():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(kb, p, "Steve Hillage", [p.anchors[1].anchor_id])
    rep = verify_citations(kb, ans)
    assert rep.n_resolvable == 1
    assert rep.n_supported == 1
    assert ans.claims[0].supported is True


def test_fabricated_anchor_is_rejected():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(kb, p, "Steve Hillage", ["0000000000::p9::s9"])
    rep = verify_citations(kb, ans, repair=False)
    assert rep.n_resolvable == 0
    assert rep.n_supported == 0
    assert ans.claims[0].supported is False
    assert any(f["why"] == "unresolvable_id" for f in rep.failures)


def test_uncited_claim_is_flagged():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(kb, p, "Steve Hillage", [])
    rep = verify_citations(kb, ans)
    assert rep.n_supported == 0
    assert any(f["why"] == "no_citation" for f in rep.failures)


def test_numeric_mismatch_is_rejected_despite_wording_overlap():
    """1953 vs 1954 is semantically near-identical and factually opposite."""
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(
        kb, p, "Miquette Giraudy was born on 9 February 1954", [p.anchors[0].anchor_id]
    )
    rep = verify_citations(kb, ans, repair=False)
    assert rep.n_supported == 0
    assert ans.claims[0].supported is False


def test_correct_number_is_accepted():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(
        kb, p, "Giraudy was born on 9 February 1953", [p.anchors[0].anchor_id]
    )
    rep = verify_citations(kb, ans, repair=False)
    assert rep.n_supported == 1


def test_wrong_pointer_is_repaired_from_elsewhere_in_context():
    """Right claim, wrong anchor: repaired and *recorded* as a repair."""
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = _answer_with(
        kb, p, "They formed the ambient band System 7 in 1989",
        [p.anchors[0].anchor_id],  # cites the birth sentence instead
    )
    rep = verify_citations(kb, ans, repair=True)
    assert rep.n_repaired == 1
    assert ans.claims[0].anchor_ids == [p.anchors[2].anchor_id]
    assert any(f["why"] == "repaired" for f in rep.failures)


def test_drop_unsupported_removes_the_claim():
    p = _mint("Miquette Giraudy", TEXT)
    kb = _kb(p)
    ans = Answer(
        answer="x",
        claims=[Claim(text="Entirely unrelated assertion about quantum chromodynamics",
                      anchor_ids=[p.anchors[0].anchor_id])],
        context=AssembledContext(passages=[p], scores=[1.0]),
    )
    rep = verify_citations(kb, ans, repair=False, drop_unsupported=True)
    assert rep.n_dropped == 1
    assert ans.claims == []


def test_cited_doc_ids_resolve_to_distinct_documents():
    p1 = _mint("Miquette Giraudy", TEXT)
    p2 = _mint("Steve Hillage", "Steve Hillage is a British progressive rock musician.")
    kb = _kb(p1, p2)
    ctx = AssembledContext(passages=[p1, p2], scores=[1.0, 0.9])
    ans = Answer(
        answer="x",
        claims=[
            Claim(text="a", anchor_ids=[p1.anchors[0].anchor_id]),
            Claim(text="b", anchor_ids=[p2.anchors[0].anchor_id]),
        ],
        context=ctx,
    )
    assert len(ans.cited_doc_ids(kb.anchors)) == 2
