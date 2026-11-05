"""Citation attribution and verification.

"Has citations" is not "is grounded". A system can emit well-formed citation
markers that point at the wrong sentence, at a sentence that no longer says what
was claimed, or at an id that does not exist. So citations are checked at three
levels, cheapest and most objective first:

**L1 -- resolvable (mechanical, no model).** Does the anchor id exist in the
knowledge base, and is the text it points at *byte-identical* to the substring
at its character offsets in the live source? This is a hard check with no
threshold and no judgement, and it is the direct test of Hard Requirement #1.
Any citation failing L1 is a fabrication and is reported as such.

**L2 -- supported (lexical + semantic).** Are the claim's content words actually
present in the cited sentence, and is the sentence semantically close to the
claim? Numbers and dates are required to match *exactly* by token, because that
is where paraphrase-tolerant scoring is most dangerous -- "1987" and "1978" are
semantically near-identical and factually opposite.

**L3 -- repair.** If a claim's cited anchor fails L2 but a *different* anchor in
the assembled context does support it, the citation is corrected and the event
recorded. This measures and fixes a real, common failure: right answer, wrong
pointer. We report repairs separately rather than folding them into the headline
precision, since silently repairing would overstate the generator.

Why no NLI model
----------------
A model such as ``roberta-large-mnli`` would give stronger entailment judgement,
at ~1.4 GB and a dependency whose version silently changes the verdict on
previously-committed citations. For the extractive reader the claim is a literal
substring of its source, so L1+L2 is exact; for free-form LLM answers it is a
real approximation, and we say so. This is an explicit, documented trade-off,
listed in DESIGN.md under next steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..index.bm25 import STOPWORDS, tokenize
from ..index.store import KnowledgeBase
from ..schema import Anchor, Answer, AssembledContext, Claim

__all__ = [
    "verify_citations",
    "attribute_claims",
    "CitationReport",
    "split_claims",
    "build_evidence_chain",
    "attach_evidence_chain",
]

_NUMERIC = re.compile(r"\d")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    n_claims: int = 0
    n_cited: int = 0
    n_resolvable: int = 0
    n_supported: int = 0
    n_repaired: int = 0
    n_dropped: int = 0
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_claims": self.n_claims,
            "n_cited": self.n_cited,
            "n_resolvable": self.n_resolvable,
            "n_supported": self.n_supported,
            "n_repaired": self.n_repaired,
            "n_dropped": self.n_dropped,
            "resolvable_rate": self.n_resolvable / self.n_cited if self.n_cited else 0.0,
            "supported_rate": self.n_supported / self.n_claims if self.n_claims else 0.0,
            "failures": self.failures[:20],
        }


def split_claims(text: str) -> list[str]:
    """Split a free-form answer into sentence-level claims."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _lexical_support(claim: str, evidence: str) -> float:
    """Share of the claim's content tokens present in the evidence.

    Recall-oriented on purpose: a claim may be shorter than its evidence, but
    every content word it asserts should be findable in the source.
    """
    c = _content_tokens(claim)
    if not c:
        return 0.0
    e = set(_content_tokens(evidence))
    return sum(1 for t in c if t in e) / len(c)


def _numeric_consistent(claim: str, evidence: str) -> bool:
    """Every number/date token in the claim must appear in the evidence.

    Semantic similarity is actively misleading for numbers, so this is a hard
    gate rather than a weighted signal.
    """
    nums = {t for t in tokenize(claim) if _NUMERIC.search(t)}
    if not nums:
        return True
    ev = set(tokenize(evidence))
    return nums <= ev


def _semantic_support(kb: KnowledgeBase, claim: str, evidence: str) -> float:
    dense = kb.dense
    if dense is None or dense.emb.size == 0:
        return 0.0
    try:
        vecs = dense.encode_queries([claim, evidence])
    except Exception:
        return 0.0
    return float(np.dot(vecs[0], vecs[1]))


def _supports(
    kb: KnowledgeBase,
    claim_text: str,
    anchor: Anchor,
    lexical_threshold: float,
    semantic_threshold: float,
    use_semantic: bool,
) -> tuple[bool, float]:
    ev = anchor.text
    if not _numeric_consistent(claim_text, ev):
        return False, 0.0
    lex = _lexical_support(claim_text, ev)
    if lex >= lexical_threshold:
        return True, lex
    if use_semantic:
        sem = _semantic_support(kb, claim_text, ev)
        if sem >= semantic_threshold:
            return True, sem
        return False, max(lex, sem)
    return False, lex


def verify_citations(
    kb: KnowledgeBase,
    answer: Answer,
    lexical_threshold: float = 0.6,
    semantic_threshold: float = 0.72,
    use_semantic: bool = False,
    repair: bool = True,
    drop_unsupported: bool = False,
) -> CitationReport:
    """Verify (and optionally repair) every claim's citations in place.

    ``use_semantic`` is off by default: it costs two encoder calls per claim and
    is unnecessary for the extractive reader, whose claim is a literal substring
    of its cited sentence. Turn it on for LLM-generated prose.
    """
    rep = CitationReport()
    ctx: AssembledContext | None = answer.context
    ctx_anchors: list[Anchor] = ctx.anchors() if ctx is not None else []

    kept: list[Claim] = []
    for claim in answer.claims:
        rep.n_claims += 1
        if not claim.anchor_ids:
            claim.supported = False
            rep.failures.append({"claim": claim.text[:160], "why": "no_citation"})
            if not drop_unsupported:
                kept.append(claim)
            else:
                rep.n_dropped += 1
            continue

        # --- L1: resolvable -------------------------------------------------
        resolvable: list[Anchor] = []
        for aid in claim.anchor_ids:
            rep.n_cited += 1
            anchor = kb.resolve_anchor(aid)
            if anchor is None:
                rep.failures.append(
                    {"claim": claim.text[:160], "anchor": aid, "why": "unresolvable_id"}
                )
                continue
            if not kb.verify_anchor(aid):
                rep.failures.append(
                    {"claim": claim.text[:160], "anchor": aid, "why": "offset_mismatch"}
                )
                continue
            rep.n_resolvable += 1
            resolvable.append(anchor)

        # --- L2: supported --------------------------------------------------
        ok_anchors: list[Anchor] = []
        best_score = 0.0
        for anchor in resolvable:
            ok, score = _supports(
                kb, claim.text, anchor, lexical_threshold, semantic_threshold,
                use_semantic,
            )
            best_score = max(best_score, score)
            if ok:
                ok_anchors.append(anchor)

        # --- L3: repair from elsewhere in the context ------------------------
        if not ok_anchors and repair and ctx_anchors:
            for anchor in ctx_anchors:
                if anchor.anchor_id in claim.anchor_ids:
                    continue
                ok, score = _supports(
                    kb, claim.text, anchor, lexical_threshold, semantic_threshold,
                    use_semantic,
                )
                if ok:
                    ok_anchors = [anchor]
                    rep.n_repaired += 1
                    rep.failures.append(
                        {
                            "claim": claim.text[:160],
                            "anchor": claim.anchor_ids[0],
                            "why": "repaired",
                            "repaired_to": anchor.anchor_id,
                        }
                    )
                    break

        if ok_anchors:
            claim.anchor_ids = [a.anchor_id for a in ok_anchors]
            claim.supported = True
            claim.support_score = float(best_score or 1.0)
            rep.n_supported += 1
            kept.append(claim)
        else:
            claim.supported = False
            claim.support_score = float(best_score)
            rep.failures.append(
                {
                    "claim": claim.text[:160],
                    "anchor": claim.anchor_ids[0] if claim.anchor_ids else None,
                    "why": "unsupported",
                    "score": round(float(best_score), 3),
                }
            )
            if drop_unsupported:
                rep.n_dropped += 1
            else:
                kept.append(claim)

    answer.claims = kept
    answer.meta["citation_report"] = rep.to_dict()
    return rep


def attribute_claims(
    kb: KnowledgeBase,
    text: str,
    context: AssembledContext,
    lexical_threshold: float = 0.5,
) -> list[Claim]:
    """Attach citations to an *uncited* free-form answer.

    Used for LLM readers that produce prose without citation markers, and as the
    fallback when a model's markers are unparseable. Each sentence is matched to
    the context anchor with the highest lexical support, so attribution is
    post-hoc but still anchored to a real location rather than invented.
    """
    anchors = context.anchors()
    claims: list[Claim] = []
    for sent in split_claims(text):
        best, best_score = None, 0.0
        for a in anchors:
            if not _numeric_consistent(sent, a.text):
                continue
            score = _lexical_support(sent, a.text)
            if score > best_score:
                best, best_score = a, score
        claims.append(
            Claim(
                text=sent,
                anchor_ids=[best.anchor_id] if best and best_score >= lexical_threshold else [],
                support_score=best_score,
            )
        )
    return claims


# --------------------------------------------------------------------------
# multi-document, claim-level provenance from the retrieval chain
# --------------------------------------------------------------------------


def build_evidence_chain(
    kb: KnowledgeBase,
    answer: Answer,
    max_hops: int = 4,
) -> list[Claim]:
    """Ground each intermediate hop of the retrieval chain in its own source.

    Why this exists
    ---------------
    An extractive reader emits one span, hence one citation, hence **one
    document** -- so `multi_doc_citation` was 0.0 for every system, including
    ours, even on questions where the context genuinely spanned four documents.
    That is a real gap against claim-level provenance: the answer is grounded,
    but the *reasoning* that reached it is not.

    The chain is already known, and it is not a reconstruction after the fact:
    the iterative retriever recorded, for every passage it added, which parent
    passage supplied which bridge entity. For "Who is the spouse of the Green
    performer?" it reached *Miquette Giraudy* from the *Green (Steve Hillage
    album)* passage via the entity "Steve Hillage". So:

        claim 1  "Steve Hillage"   <- Green (Steve Hillage album), sentence 0
        claim 2  "Miquette Giraudy" <- Miquette Giraudy, sentence 1   [answer]

    Two documents, two claims, each citing the exact sentence that states it.

    Honesty notes
    -------------
    * These claims are **derived from retrieval, not generated**. The bridge
      entity is a literal surface form present in the parent passage, so the
      claim is a substring of its cited sentence and L1+L2 verification is
      exact, not approximate.
    * A single-shot baseline has no chain, so it emits only the answer claim.
      That is the correct outcome rather than a handicap: the chain is a
      product of iterative retrieval, and the inability to produce
      multi-document claim-level citations is precisely what it means to lack
      one. EVAL.md states this when interpreting `multi_doc_citation`.
    * Only passages that are actually in the assembled context contribute. We
      never cite evidence the reader was not shown.
    """
    ctx = answer.context
    if ctx is None or not ctx.passages:
        return []
    chain_map = ctx.trace.provenance_chain if ctx.trace else {}
    if not chain_map:
        return []

    in_ctx = {p.passage_id: p for p in ctx.passages}
    # Start from the passage the answer was extracted from, when known;
    # otherwise from the highest-scoring context passage that has a chain.
    start = answer.meta.get("passage_id")
    if start not in chain_map:
        start = next((pid for pid in ctx.passage_ids if pid in chain_map), None)
    if start is None:
        return []

    # Walk back to the root, collecting (bridge_entity, parent_passage).
    hops: list[tuple[str, str]] = []
    seen: set[str] = set()
    node = start
    while node in chain_map and len(hops) < max_hops and node not in seen:
        seen.add(node)
        link = chain_map[node]
        parent, entity = link.get("parent"), link.get("bridge_entity")
        if not parent or not entity:
            break
        hops.append((str(entity), str(parent)))
        node = parent

    claims: list[Claim] = []
    for entity_key, parent_id in reversed(hops):   # root-first reads naturally
        parent = in_ctx.get(parent_id)
        if parent is None:
            # The bridge's parent was dropped by assembly; citing it would
            # point at evidence the reader never saw.
            continue
        anchor = _anchor_mentioning(parent, entity_key)
        if anchor is None:
            continue
        # Quote the entity as it appears in the source, not the normalised
        # index key, so the claim text is genuinely a substring of its citation.
        surface = _surface_in(anchor.text, entity_key) or entity_key
        claims.append(
            Claim(
                text=surface,
                anchor_ids=[anchor.anchor_id],
                support_score=1.0,
            )
        )
    return claims


def _anchor_mentioning(passage, entity_key: str) -> Anchor | None:
    """The sentence of ``passage`` that states ``entity_key``, if any."""
    want = [t for t in entity_key.split() if t]
    if not want:
        return None
    best, best_hits = None, 0
    for a in passage.anchors:
        toks = set(tokenize(a.text))
        hits = sum(1 for t in want if t in toks)
        if hits > best_hits:
            best, best_hits = a, hits
    # Require the whole entity to be present: a partial match would ground the
    # claim in a sentence that does not actually state it.
    return best if best_hits == len(want) else None


def _surface_in(sentence: str, entity_key: str) -> str | None:
    """Recover the entity's original casing/spelling from its sentence."""
    want = [t for t in entity_key.split() if t]
    if not want:
        return None
    words = sentence.split()
    norm = [tokenize(w) for w in words]
    for i in range(len(words)):
        j, k = i, 0
        while j < len(words) and k < len(want):
            if norm[j] and norm[j][0] == want[k]:
                k += 1
                j += 1
            elif not norm[j]:
                j += 1
            else:
                break
        if k == len(want):
            return " ".join(words[i:j]).strip(" .,;:()’'\"")
    return None


def attach_evidence_chain(kb: KnowledgeBase, answer: Answer) -> Answer:
    """Prepend the grounded intermediate-hop claims to ``answer.claims``."""
    chain = build_evidence_chain(kb, answer)
    if not chain:
        return answer
    existing = {tuple(c.anchor_ids) for c in answer.claims}
    answer.claims = [
        c for c in chain if tuple(c.anchor_ids) not in existing
    ] + answer.claims
    answer.meta["n_chain_claims"] = len(answer.claims) - len(existing)
    return answer
