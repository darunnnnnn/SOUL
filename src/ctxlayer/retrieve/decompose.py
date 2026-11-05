"""Question analysis: what is known, what is being asked, what is missing.

This module is deliberately **LLM-free and deterministic**. That is a design
choice with two justifications:

1. *Attribution of credit.* If query planning used an LLM, a measured gain over
   a BM25 baseline would be partly an LLM-capability gain, and the experiment
   would no longer isolate the context layer. Everything here is rule-based, so
   the improvement reported in EVAL.md is attributable to retrieval structure.
2. *Cost and reproducibility.* The iterative retriever issues
   1 + rounds x beam queries per question. Putting a model call in that loop
   would multiply evaluation cost by ~10x and make results non-deterministic.

What it produces
----------------
``QueryPlan`` separates three things that single-shot retrieval conflates:

``anchor_entities``  entities the question *gives* you ("Green"). These locate
                     the first hop but must NOT be re-used as bridges -- they
                     are where we started.
``relation_terms``   the relational/attribute words that survive after the
                     anchor entities are removed ("spouse", "performer"). These
                     are what must be carried into every later hop: the bridge
                     entity alone ("Steve Hillage") is a topic, not a question.
``focus``            the expected answer type inferred from the wh-word, used
                     by the extractive reader to filter implausible spans.

Composing ``bridge_query(entity) = entity + relation_terms`` is the single most
important trick here. Searching for the bare bridge entity returns the article
*about* it; searching entity + relation returns the paragraph that states the
*relation* -- which is the one holding the next hop's answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..index.bm25 import STOPWORDS, tokenize
from ..index.entity_index import extract_spans

__all__ = ["QueryPlan", "analyze_question", "ANSWER_TYPES"]


# Expected-answer-type cues. Used only as a soft signal by the reader; a wrong
# guess never removes a candidate, it only re-ranks (see synth/reader.py).
ANSWER_TYPES: dict[str, str] = {
    "who": "person_or_org",
    "whom": "person_or_org",
    "whose": "person_or_org",
    "when": "date",
    "where": "location",
    "which country": "location",
    "which city": "location",
    "which state": "location",
    "how many": "number",
    "how much": "number",
    "how long": "number",
    "what year": "date",
    "what date": "date",
}

# Interrogative and framing words that carry no retrieval signal. Kept separate
# from STOPWORDS because BM25 stopping and relation extraction want different
# lists: "of" is a useless BM25 term but "spouse of" vs "spouse" matters little,
# whereas "what" must never become a relation term.
_WH = frozenset(
    """who whom whose what which when where why how was were is are did does do
    the a an in on at of for to from by with that this these those there it its
    his her their they he she you we i been being be am had has have having
    name named call called also known as well""".split()
)


@dataclass
class QueryPlan:
    question: str
    content_terms: list[str] = field(default_factory=list)
    anchor_entities: list[str] = field(default_factory=list)
    anchor_entity_tokens: set[str] = field(default_factory=set)
    relation_terms: list[str] = field(default_factory=list)
    focus: str = "unknown"
    n_expected_hops: int | None = None

    def bridge_query(self, entity_surface: str, max_relation_terms: int = 8) -> str:
        """Query for the next hop: the bridge entity plus the relation asked.

        The entity alone retrieves the article *about* it; the entity plus the
        relation retrieves the paragraph that *states* the relation, which is
        where the next hop's answer lives.
        """
        rel = " ".join(self.relation_terms[:max_relation_terms])
        return f"{entity_surface} {rel}".strip()

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "anchor_entities": self.anchor_entities,
            "relation_terms": self.relation_terms,
            "focus": self.focus,
        }


def _detect_focus(question: str) -> str:
    """Infer the expected answer type from the *governing* wh-word.

    Multi-hop questions routinely contain several wh-words, only the first of
    which governs the answer: "**When** did the carnival take place in the city
    **where** CIMI-FM is licensed?" asks for a date, not a location. Matching
    the longest cue anywhere in the string gets this wrong, so cues are ranked
    by position first and length only as a tie-break.
    """
    q = question.lower().strip()
    best: tuple[int, int, str] | None = None
    for cue in ANSWER_TYPES:
        pos = q.find(cue)
        if pos < 0:
            continue
        cand = (pos, -len(cue), cue)
        if best is None or cand < best:
            best = cand
    if best is not None:
        return ANSWER_TYPES[best[2]]
    if q.startswith("what"):
        return "entity"
    return "unknown"


def analyze_question(question: str, max_entity_words: int = 5) -> QueryPlan:
    plan = QueryPlan(question=question)
    plan.focus = _detect_focus(question)

    # --- anchor entities: proper-noun spans the question hands us ----------
    # Keep only maximal spans for the anchor list (sub-spans are useful for
    # *matching* but noisy when deciding "what did the question give me").
    spans = extract_spans(question, max_entity_words)
    maximal: list[str] = []
    for s in spans:
        if not any(s != o and s in o for o in spans):
            maximal.append(s)
    # A leading capital is just sentence case, so a question-initial single word
    # is only an anchor if it is not an interrogative/framing word.
    cleaned: list[str] = []
    for s in maximal:
        toks = tokenize(s)
        if not toks or all(t in _WH or t in STOPWORDS for t in toks):
            continue
        cleaned.append(s)
    plan.anchor_entities = list(dict.fromkeys(cleaned))
    plan.anchor_entity_tokens = {t for s in plan.anchor_entities for t in tokenize(s)}

    # --- content and relation terms ---------------------------------------
    all_toks = tokenize(question)
    plan.content_terms = [t for t in dict.fromkeys(all_toks) if t not in STOPWORDS]
    plan.relation_terms = [
        t
        for t in plan.content_terms
        if t not in _WH and t not in plan.anchor_entity_tokens and len(t) > 2
    ]
    # Degenerate case: the whole question is one entity ("Who is X?"). Fall back
    # to the content terms so a later hop still carries *some* relational
    # signal instead of searching for a bare topic.
    if not plan.relation_terms:
        plan.relation_terms = [t for t in plan.content_terms if t not in _WH]
    return plan
