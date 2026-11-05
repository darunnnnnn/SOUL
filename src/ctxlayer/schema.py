"""Core data model for the context layer.

Provenance design
-----------------
The unit of *retrieval* is a ``Passage`` (a Wikipedia paragraph). The unit of
*citation* is an ``Anchor``: a single sentence inside a passage, addressed by a
stable, human-auditable string::

    <doc_id>::p<passage_ord>::s<sent_idx>

Every anchor additionally carries explicit character offsets into the passage
text, so a citation can be mechanically re-resolved to the exact substring of
the original source. Nothing in the pipeline invents an anchor: anchors are
minted once at ingestion time and only ever referenced afterwards. This is what
makes Hard Requirement #1 (provenance) checkable rather than aspirational --
see ``ctxlayer.synth.cite.verify_citations``.

A ``Document`` is a distinct source (one Wikipedia article / title). The
distinction between *document* and *passage* is load-bearing: Hard Requirement
#2 is about evidence spanning >= 2 distinct **documents**, so multi-document
coverage is always measured over ``doc_id``, never over passage count.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "Anchor",
    "Passage",
    "Document",
    "Question",
    "ScoredPassage",
    "RetrievalTrace",
    "AssembledContext",
    "Claim",
    "Answer",
    "stable_id",
    "split_sentences",
]


def stable_id(*parts: str, length: int = 12) -> str:
    """Deterministic content-addressed id.

    Used instead of positional ids so that an index built from a different data
    ordering (or from a subset) yields identical ids for identical content.
    That property is what lets cached embeddings and committed raw eval outputs
    stay valid across rebuilds.
    """
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

# Deliberately dependency-free and deterministic. A model-based splitter would
# add a heavy dependency and, worse, make anchor ids depend on a model version
# -- which would silently invalidate every previously committed citation.
# Abbreviations that matter for encyclopedic prose are protected explicitly.
_ABBREV = (
    r"Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|vs|etc|No|Nos|Inc|Ltd|Co|Corp|Dept|Univ"
    r"|Gen|Col|Capt|Lt|Sgt|Rev|Hon|Gov|Sen|Rep|Fig|Vol|Ch|pp|ca|cf|al|approx"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_PROTECT = re.compile(r"\b(?:" + _ABBREV + r")\.")
_INITIAL = re.compile(r"\b([A-Z])\.")
_ELLIPSIS = re.compile(r"\.\.\.")
_SENT_END = re.compile("(?<=[.!?])[\"')\\]]*\\s+")
_SENTINEL = ""  # private-use char; cannot occur in Wikipedia text

_MIN_SENT_CHARS = 20


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentences returned as ``(start, end)`` char spans.

    Returning *spans* rather than strings is intentional: spans are what make a
    citation verifiable against the untouched source string.
    """
    if not text or not text.strip():
        return []

    # Mask characters that would cause false sentence breaks, preserving length
    # exactly so that offsets computed on the mask are valid for the original.
    masked = _PROTECT.sub(lambda m: m.group(0)[:-1] + _SENTINEL, text)
    masked = _INITIAL.sub(lambda m: m.group(1) + _SENTINEL, masked)
    masked = _ELLIPSIS.sub(_SENTINEL * 3, masked)
    if len(masked) != len(text):  # pragma: no cover - invariant guard
        raise AssertionError("sentence masking must preserve length")

    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(masked):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))

    # Merge pathologically short fragments into the previous sentence so that
    # no anchor ever points at something uselessly small to cite.
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and (e - s) < _MIN_SENT_CHARS:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged or [(0, len(text))]


# --------------------------------------------------------------------------
# Ingested units
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anchor:
    """A citable sentence-level location inside a passage."""

    anchor_id: str
    passage_id: str
    doc_id: str
    doc_title: str
    passage_ord: int
    sent_idx: int
    char_start: int
    char_end: int
    text: str

    def locator(self) -> str:
        """Human-readable source location, e.g. ``Green (album) p1 s2``."""
        return f"{self.doc_title} ¶{self.passage_ord} s{self.sent_idx}"


@dataclass(slots=True)
class Passage:
    """A paragraph: the atomic unit of retrieval and of context packing."""

    passage_id: str
    doc_id: str
    doc_title: str
    passage_ord: int
    text: str
    anchors: list[Anchor] = field(default_factory=list)
    # Provenance back to the benchmark record, kept for auditability only --
    # never consulted by retrieval.
    source_ref: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    def with_title(self) -> str:
        """Title-prefixed text, as fed to the lexical and dense encoders.

        Wikipedia paragraphs are frequently anaphoric ("The album was ...") and
        omit their own subject, so the article title carries a large share of
        the retrievable signal. Prefixing it is what makes paragraph-level
        retrieval viable at all.
        """
        return f"{self.doc_title}. {self.text}"


@dataclass(slots=True)
class Document:
    """A distinct source document (one Wikipedia article)."""

    doc_id: str
    title: str
    passage_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Question:
    """A benchmark question plus its gold provenance labels."""

    qid: str
    question: str
    answer: str
    answer_aliases: list[str] = field(default_factory=list)
    n_hops: int = 2
    hop_type: str = "2hop"
    # Gold supporting passages, and the distinct documents they come from.
    gold_passage_ids: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_doc_titles: list[str] = field(default_factory=list)
    # Per-question candidate pool (the standard MuSiQue "distractor" setting).
    pool_passage_ids: list[str] = field(default_factory=list)
    # Gold decomposition. Used ONLY for oracle/analysis conditions, never by
    # any evaluated system -- see eval.runner for the enforcement check.
    decomposition: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_gold_docs(self) -> int:
        return len(set(self.gold_doc_ids))

    def all_answers(self) -> list[str]:
        return [self.answer, *self.answer_aliases]


# --------------------------------------------------------------------------
# Retrieval / assembly outputs
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ScoredPassage:
    passage_id: str
    score: float
    # Per-signal diagnostics (bm25 / dense / rrf / rerank / hop) so that every
    # number in the eval report can be traced to the signal that produced it.
    components: dict[str, float] = field(default_factory=dict)
    hop: int = 0
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScoredPassage({self.passage_id}, {self.score:.4f}, hop={self.hop})"


@dataclass(slots=True)
class RetrievalTrace:
    """Full audit trail of one retrieval, for debugging and for the report."""

    rounds: list[dict[str, Any]] = field(default_factory=list)
    queries_issued: list[str] = field(default_factory=list)
    bridge_entities: list[str] = field(default_factory=list)
    # passage_id -> {parent, bridge_entity, hop}: how each passage was reached.
    # This is the evidence *chain*, and it is what makes multi-document,
    # claim-level citation possible: the answer span grounds the final hop,
    # while the chain grounds each intermediate hop in the passage that
    # actually supplied the bridge entity. See synth.cite.build_evidence_chain.
    provenance_chain: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_llm_calls: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    def add_round(self, **kw: Any) -> None:
        self.rounds.append(kw)


@dataclass(slots=True)
class AssembledContext:
    """The context layer's product: a token-budgeted, multi-document bundle."""

    passages: list[Passage] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)

    @property
    def passage_ids(self) -> list[str]:
        return [p.passage_id for p in self.passages]

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(p.doc_id for p in self.passages))

    @property
    def n_words(self) -> int:
        return sum(p.n_words for p in self.passages)

    def anchors(self) -> list[Anchor]:
        return [a for p in self.passages for a in p.anchors]

    def render(self, numbered: bool = True) -> str:
        """Render the context for a reader, with anchor ids inline.

        Anchor ids are emitted *in the prompt* so a generator can cite by id and
        the citation can then be mechanically verified. A reader that cannot see
        location ids cannot produce claim-level provenance.
        """
        out: list[str] = []
        for i, p in enumerate(self.passages, 1):
            head = f"[{i}] {p.doc_title}" if numbered else p.doc_title
            body = " ".join(f"({a.anchor_id}) {a.text}" for a in p.anchors)
            out.append(f"{head}\n{body}")
        return "\n\n".join(out)


# --------------------------------------------------------------------------
# Synthesis outputs
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Claim:
    """One factual assertion in an answer, with its supporting anchors."""

    text: str
    anchor_ids: list[str] = field(default_factory=list)
    supported: bool | None = None
    support_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    answer: str
    claims: list[Claim] = field(default_factory=list)
    abstained: bool = False
    context: AssembledContext | None = None
    raw_generation: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def cited_anchor_ids(self) -> list[str]:
        return list(dict.fromkeys(a for c in self.claims for a in c.anchor_ids))

    def cited_doc_ids(self, anchor_lookup: dict[str, Anchor]) -> list[str]:
        return list(
            dict.fromkeys(
                anchor_lookup[a].doc_id
                for a in self.cited_anchor_ids()
                if a in anchor_lookup
            )
        )
