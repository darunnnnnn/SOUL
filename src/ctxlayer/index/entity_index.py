"""Entity mention index: the bridge that makes multi-hop retrieval possible.

The problem
-----------
Multi-hop questions chain facts, and the chain goes *lexically dark* after the
first hop::

    Q: "Who is the spouse of the Green performer?"
    hop 1: Green (Steve Hillage album)  -> performer = Steve Hillage
    hop 2: Miquette Giraudy             -> "...her partner Steve Hillage"

The gold document for hop 2 is titled *Miquette Giraudy*. The question contains
neither "Hillage" nor "Giraudy". **No single-shot retriever can rank that
document highly** -- sparse, dense, or reranked -- because the query does not
contain the terms that identify it. This is the hard ceiling on single-shot RAG
for multi-hop questions, and it is a retrieval failure, not a reading failure.
The only fix is to let evidence found in round *n* generate the query for round
*n+1*, which requires knowing which entities a passage mentions.

Why mentions and not a title graph (measured, not assumed)
----------------------------------------------------------
We built the obvious thing first -- a document-link graph keyed on article
titles, approximating Wikipedia's link structure -- and measured how often the
gold evidence chain is actually traversable in it:

    gold-chain reachability, title-keyed link graph .......... 30.4%
    gold-chain reachability, surface-form mention index ...... 90.3%

The title graph fails structurally: the hop-*n+1* bridge entity is usually not
an article title in the corpus (only 22.5% of MuSiQue bridge answers resolve to
a title here), it is a *mention* inside the next gold paragraph. So we index
mentions. Both numbers are reproducible with
``python -m ctxlayer.cli diagnose reachability``.

Why phrase containment and not just BM25 on the entity string
-------------------------------------------------------------
Candidate *generation* wants precision: a 2-token entity used as a BM25 query
returns a long tail of passages matching only one of its tokens. Exact span
containment returns passages that genuinely talk about the entity. Candidates
are then *scored* by the fused retriever and the cross-encoder -- so this index
decides what is considered, not what wins.

Proper-noun detection without an NER model
------------------------------------------
Capitalisation alone over-generates at sentence start ("Written in spring 1977"
-> "Written"). Instead of adding a 400 MB NER model -- and coupling anchor-time
behaviour to a model version -- we use a corpus statistic: a single-token
candidate is a proper noun only if it appears capitalised in >= ``min_cap_ratio``
of its corpus occurrences. "Zhejiang" passes; "Written" and "During" do not.
"""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from ..config import GraphConfig
from ..schema import Passage
from .bm25 import STOPWORDS, tokenize

__all__ = ["EntityIndex", "extract_spans"]

# Maximal capitalised spans, allowing the lowercase connectors that genuinely
# occur inside entity names ("Bank of England", "Ludwig van Beethoven").
#
# Two subtleties, both found by inspecting extracted spans:
#   * A word may contain an internal period ("J.R.R.", "St.Louis") but a
#     *trailing* period ends a sentence. Allowing `.` freely inside the token
#     class made spans run across sentence boundaries, yielding garbage like
#     "Kathmandu the Bagmati River" from "Kathmandu. The Bagmati River ...".
#     So a period is only part of a word when a letter follows it.
#   * "the"/"and"/"in" are NOT name-internal connectors. Including them
#     produced "Hindu and Buddhist" and "Kathmandu the" as entities.
_CONNECTORS = r"(?:of|de|del|della|da|van|von|der|di|du|la|le|bin|al)"
_WORD = r"[A-Z][\w&'’-]*(?:\.(?=[A-Za-z])[\w&'’-]*)*"
_SPAN_RE = re.compile(
    r"\b" + _WORD + r"(?:\s+(?:" + _CONNECTORS + r"\s+)?" + _WORD + r")*"
)
_CONNECTOR_TOKENS = frozenset(
    "of de del della da van von der di du la le bin al".split()
)

# Titles that are real articles but far too generic to be useful bridges.
_GENERIC = frozenset(
    """green red blue black white time times post news world war city town
    state county district river lake island school university college church
    album song single film movie book series season team club league company
    band history music list index outline""".split()
)


def extract_spans(text: str, max_words: int) -> list[str]:
    """Return candidate entity surface forms (raw casing preserved).

    Emits each maximal capitalised span plus its contiguous sub-spans, so that
    "British Steve Hillage" also yields "Steve Hillage". Sub-spans are what let
    an entity be matched when the two passages name it with different
    modifiers.
    """
    out: list[str] = []
    for m in _SPAN_RE.finditer(text):
        span = m.group(0).strip(" .,;:'’-")
        words = span.split()
        if not words:
            continue
        n = min(len(words), max_words)
        for size in range(n, 0, -1):
            for i in range(0, len(words) - size + 1):
                sub = words[i : i + size]
                # A sub-span must not begin or end on a connector: "Bank of"
                # and "of England" are not entities.
                if sub[0].lower() in _CONNECTOR_TOKENS:
                    continue
                if sub[-1].lower() in _CONNECTOR_TOKENS:
                    continue
                out.append(" ".join(sub))
    return out


class EntityIndex:
    """Surface-form mention postings + title resolution over a passage set."""

    def __init__(self, cfg: GraphConfig) -> None:
        self.cfg = cfg
        # entity_key -> tuple of passage_ids containing that surface form
        self.postings: dict[str, tuple[str, ...]] = {}
        # passage_id -> tuple of entity_keys it contains (ordered, deduped)
        self.passage_entities: dict[str, tuple[str, ...]] = {}
        # normalised title -> doc_id (high-precision "article about X" lookup)
        self.title_lookup: dict[str, str] = {}
        self.doc_title: dict[str, str] = {}
        self.doc_passages: dict[str, list[str]] = {}
        self.stats: dict[str, object] = {}

    # ---------------------------------------------------------------- build
    def _accept_key(self, key: str, n_tokens: int, cap_ok: bool) -> bool:
        cfg = self.cfg
        if not key or len(key) < cfg.min_entity_chars:
            return False
        toks = key.split()
        if all(t in STOPWORDS for t in toks):
            return False
        if n_tokens == 1:
            if toks[0] in _GENERIC or len(toks[0]) <= 3:
                return False
            if not cap_ok:
                return False
        return True

    def build(self, passages: Sequence[Passage], verbose: bool = True) -> "EntityIndex":
        cfg = self.cfg
        for p in passages:
            self.doc_title[p.doc_id] = p.doc_title
            self.doc_passages.setdefault(p.doc_id, []).append(p.passage_id)

        # --- title lookup (deterministic on title collisions) --------------
        if cfg.enable_title_edges:
            for doc_id, title in sorted(self.doc_title.items(), key=lambda kv: kv[1]):
                if any(title.startswith(pre) for pre in cfg.drop_title_prefixes):
                    continue
                toks = tokenize(title)
                if not toks or len(title) < cfg.min_title_chars:
                    continue
                if all(t in STOPWORDS for t in toks):
                    continue
                self.title_lookup.setdefault(" ".join(toks), doc_id)

        if not cfg.enabled:
            return self

        # --- pass 1: corpus capitalisation statistics ----------------------
        # Drives the proper-noun test for single-token candidates.
        total: dict[str, int] = defaultdict(int)
        capped: dict[str, int] = defaultdict(int)
        for p in passages:
            for word in p.text.split():
                sub = tokenize(word)
                if len(sub) != 1:
                    continue
                total[sub[0]] += 1
                if word[:1].isupper():
                    capped[sub[0]] += 1
        cap_ratio = {
            t: capped.get(t, 0) / c for t, c in total.items() if c > 0
        }

        # --- pass 2: extract spans and invert ------------------------------
        raw_post: dict[str, list[str]] = defaultdict(list)
        n = len(passages)
        for i, p in enumerate(passages):
            keys: list[str] = []
            # The title is indexed too: it is the single most reliable
            # statement of what a passage is about.
            source = p.doc_title + ". " + p.text
            for span in extract_spans(source, cfg.max_entity_words):
                toks = tokenize(span)
                if not toks:
                    continue
                key = " ".join(toks)
                cap_ok = (
                    cap_ratio.get(toks[0], 0.0) >= cfg.min_cap_ratio
                    if len(toks) == 1
                    else True
                )
                if not self._accept_key(key, len(toks), cap_ok):
                    continue
                keys.append(key)
            keys = list(dict.fromkeys(keys))
            self.passage_entities[p.passage_id] = tuple(keys)
            for k in keys:
                raw_post[k].append(p.passage_id)
            if verbose and i % 2000 == 0:
                print(f"\r  entities: {i}/{n}", end="", flush=True)
        if verbose:
            print(f"\r  entities: {n}/{n}", flush=True)

        # --- df pruning -----------------------------------------------------
        dropped = 0
        for k, pids in raw_post.items():
            if len(pids) > cfg.max_entity_df:
                dropped += 1
                continue
            self.postings[k] = tuple(dict.fromkeys(pids))
        # Keep per-passage entity lists consistent with the pruned postings so
        # the retriever never proposes a bridge it cannot follow.
        pruned_set = self.postings
        for pid, keys in list(self.passage_entities.items()):
            self.passage_entities[pid] = tuple(k for k in keys if k in pruned_set)

        self.stats = {
            "n_passages": n,
            "n_entity_keys": len(self.postings),
            "n_postings": sum(len(v) for v in self.postings.values()),
            "n_keys_dropped_high_df": dropped,
            "n_titles_indexed": len(self.title_lookup),
            "mean_entities_per_passage": round(
                sum(len(v) for v in self.passage_entities.values()) / max(1, n), 2
            ),
        }
        return self

    # --------------------------------------------------------------- queries
    def df(self, entity_key: str) -> int:
        return len(self.postings.get(entity_key, ()))

    def entities_in(self, passage_id: str) -> tuple[str, ...]:
        return self.passage_entities.get(passage_id, ())

    def passages_mentioning(self, entity_key: str, limit: int = 80) -> list[str]:
        return list(self.postings.get(entity_key, ())[:limit])

    def resolve_entity(self, name: str) -> str | None:
        """Resolve a surface form to the document *about* it, if one exists."""
        toks = tokenize(name)
        if not toks:
            return None
        hit = self.title_lookup.get(" ".join(toks))
        if hit:
            return hit
        # Titles are routinely disambiguated ("Green (Steve Hillage album)"),
        # so try progressively shorter prefixes.
        for cut in range(len(toks) - 1, max(0, len(toks) - 3), -1):
            hit = self.title_lookup.get(" ".join(toks[:cut]))
            if hit:
                return hit
        return None

    def passages_about(self, doc_id: str) -> list[str]:
        return self.doc_passages.get(doc_id, [])

    def candidates_for_entity(self, entity_key: str, limit: int = 40) -> list[str]:
        """Passages to consider for a bridge entity, both directions.

        Union of (a) passages that *mention* the entity -- the direction that
        reaches an inverse relation such as "spouse of X", and (b) passages of
        the document *about* the entity, when such a document exists.
        """
        out = self.passages_mentioning(entity_key, limit=limit)
        doc_id = self.resolve_entity(entity_key)
        if doc_id:
            out = list(self.passages_about(doc_id)) + out
        return list(dict.fromkeys(out))[:limit]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "cfg": self.cfg.to_dict(),
                    "postings": self.postings,
                    "passage_entities": self.passage_entities,
                    "title_lookup": self.title_lookup,
                    "doc_title": self.doc_title,
                    "doc_passages": self.doc_passages,
                    "stats": self.stats,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "EntityIndex":
        with path.open("rb") as fh:
            d = pickle.load(fh)
        idx = cls(GraphConfig.from_dict(d["cfg"]))
        idx.postings = d["postings"]
        idx.passage_entities = d["passage_entities"]
        idx.title_lookup = d["title_lookup"]
        idx.doc_title = d["doc_title"]
        idx.doc_passages = d["doc_passages"]
        idx.stats = d["stats"]
        return idx
