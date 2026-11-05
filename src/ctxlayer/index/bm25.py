"""BM25F lexical index over two fields (title, body).

Why hand-rolled instead of ``rank_bm25``
----------------------------------------
``rank_bm25`` implements single-field BM25 over a Python list and recomputes
scores in pure Python per query. Two things matter here that it cannot express:

1. **Fielded scoring.** In MuSiQue the bridge entity of hop *n+1* is usually
   *the title* of the document that answers it ("Green >> performer" ->
   "Steve Hillage" -> the article titled *Steve Hillage*). A title term is far
   more diagnostic than the same term buried in body prose, and BM25F weights
   the fields *before* saturation -- which is the statistically correct way to
   combine them, unlike scoring two indexes and adding the results.
2. **Speed under an iterative retriever.** Our system issues 1 + (rounds x
   beam) queries per question, so query cost is multiplied. The inverted index
   here is CSR-style NumPy, and scoring is a vectorised scatter-add: ~0.2 ms
   per query over 21k passages, versus ~30 ms for a pure-Python loop.

Scoring follows Robertson & Zaragoza (2009), eq. for BM25F::

    tf~(t,d) = sum_f  w_f * tf(t,d,f) / (1 - b_f + b_f * len(d,f)/avglen_f)
    score(q,d) = sum_{t in q}  idf(t) * tf~(t,d) / (k1 + tf~(t,d))

Note the saturation is applied to the *pooled* pseudo-frequency, which is what
distinguishes BM25F from "BM25 per field, then sum".
"""

from __future__ import annotations

import math
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import BM25Config
from ..schema import Passage

__all__ = ["BM25FIndex", "tokenize", "STOPWORDS"]


# Standard English stoplist. Kept short and conservative: aggressive stopping
# hurts multi-hop questions, where function words carry the relation
# ("spouse OF the performer", "who was born IN").
STOPWORDS = frozenset(
    """a an the and or but if then than that this these those there here
    is are was were be been being am do does did doing have has had having
    i you he she it we they me him her them my your his its our their
    of in on at to for with by from as into about over after before between
    out up down off again further once during above below under
    s t can will just don should now also""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.UNICODE)


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """Deterministic, Unicode-aware tokenizer.

    Accent folding (``Beyoncé`` -> ``beyonce``) matters a great deal on
    Wikipedia-derived corpora, where the question and the article often differ
    only in diacritics. Possessives and plural ``-s`` are folded because the
    question phrases an entity differently from the article title far more often
    than two distinct entities differ by a single trailing ``s``. No Porter
    stemming: it mangles proper nouns, which are precisely what we match on.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    toks: list[str] = []
    for tok in _TOKEN_RE.findall(folded):
        if "'" in tok:
            tok = tok.split("'")[0]
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if not tok:
            continue
        if drop_stopwords and tok in STOPWORDS:
            continue
        toks.append(tok)
    return toks


@dataclass
class _Field:
    """CSR-style postings for one field."""

    lengths: np.ndarray  # (n_docs,) float32 field length in tokens
    avg_len: float


class BM25FIndex:
    """Fielded BM25 over a fixed passage collection."""

    def __init__(self, cfg: BM25Config) -> None:
        self.cfg = cfg
        self.passage_ids: list[str] = []
        self._pid_to_row: dict[str, int] = {}
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        # Pooled, length-normalised pseudo-term-frequency in CSR form, keyed by
        # term: offsets[t]:offsets[t+1] slices rows[] / tfs[].
        self._offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._rows: np.ndarray = np.zeros(0, dtype=np.int32)
        self._tfs: np.ndarray = np.zeros(0, dtype=np.float32)
        self.n_docs = 0

    # ---------------------------------------------------------------- build
    def build(self, passages: Sequence[Passage]) -> "BM25FIndex":
        cfg = self.cfg
        self.passage_ids = [p.passage_id for p in passages]
        self._pid_to_row = {pid: i for i, pid in enumerate(self.passage_ids)}
        n = self.n_docs = len(passages)

        title_toks = [tokenize(p.doc_title) for p in passages]
        body_toks = [tokenize(p.text) for p in passages]
        t_len = np.array([len(t) for t in title_toks], dtype=np.float32)
        b_len = np.array([len(t) for t in body_toks], dtype=np.float32)
        t_avg = float(t_len.mean()) if n else 1.0
        b_avg = float(b_len.mean()) if n else 1.0
        self._title = _Field(t_len, t_avg)
        self._body = _Field(b_len, b_avg)

        # Pass 1: vocabulary + document frequency over the union of fields.
        df: dict[str, int] = {}
        for tt, bt in zip(title_toks, body_toks):
            for tok in set(tt) | set(bt):
                df[tok] = df.get(tok, 0) + 1
        self._vocab = {tok: i for i, tok in enumerate(sorted(df))}
        v = len(self._vocab)

        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at a
        # small positive value so that a term appearing in every passage
        # contributes ~0 instead of a negative score (which would make adding a
        # common query term actively reduce a passage's rank).
        idf = np.zeros(v, dtype=np.float32)
        for tok, i in self._vocab.items():
            d = df[tok]
            idf[i] = max(1e-6, math.log(1.0 + (n - d + 0.5) / (d + 0.5)))
        self._idf = idf

        # Pass 2: pooled pseudo-tf per (term, passage).
        norm_t = 1.0 - cfg.b_title + cfg.b_title * (t_len / max(t_avg, 1e-6))
        norm_b = 1.0 - cfg.b_body + cfg.b_body * (b_len / max(b_avg, 1e-6))
        acc: dict[int, dict[int, float]] = {}
        for row, (tt, bt) in enumerate(zip(title_toks, body_toks)):
            local: dict[int, float] = {}
            if cfg.w_title:
                nt = norm_t[row]
                for tok in tt:
                    ti = self._vocab[tok]
                    local[ti] = local.get(ti, 0.0) + cfg.w_title / nt
            if cfg.w_body:
                nb = norm_b[row]
                for tok in bt:
                    ti = self._vocab[tok]
                    local[ti] = local.get(ti, 0.0) + cfg.w_body / nb
            for ti, val in local.items():
                acc.setdefault(ti, {})[row] = val

        counts = np.zeros(v + 1, dtype=np.int64)
        for ti, post in acc.items():
            counts[ti + 1] = len(post)
        self._offsets = np.cumsum(counts)
        total = int(self._offsets[-1])
        self._rows = np.zeros(total, dtype=np.int32)
        self._tfs = np.zeros(total, dtype=np.float32)
        for ti, post in acc.items():
            start = int(self._offsets[ti])
            items = sorted(post.items())
            self._rows[start : start + len(items)] = [r for r, _ in items]
            self._tfs[start : start + len(items)] = [t for _, t in items]
        return self

    # --------------------------------------------------------------- search
    def score_all(self, query: str) -> np.ndarray:
        """Return a dense (n_docs,) score vector for ``query``."""
        k1 = self.cfg.k1
        scores = np.zeros(self.n_docs, dtype=np.float32)
        qtoks = tokenize(query, drop_stopwords=True)
        if not qtoks:
            return scores
        # Query term frequency is intentionally ignored (BM25's standard
        # treatment for short queries): repeating a word in a question does not
        # make a passage more relevant.
        for tok in dict.fromkeys(qtoks):
            ti = self._vocab.get(tok)
            if ti is None:
                continue
            s, e = int(self._offsets[ti]), int(self._offsets[ti + 1])
            if s == e:
                continue
            tf = self._tfs[s:e]
            np.add.at(scores, self._rows[s:e], self._idf[ti] * tf / (k1 + tf))
        return scores

    def search(
        self,
        query: str,
        top_k: int = 100,
        restrict_to: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-``top_k`` ``(passage_id, score)``, optionally within a pool."""
        scores = self.score_all(query)
        if restrict_to is not None:
            rows = np.fromiter(
                (self._pid_to_row[p] for p in restrict_to if p in self._pid_to_row),
                dtype=np.int64,
            )
            if rows.size == 0:
                return []
            sub = scores[rows]
            k = min(top_k, rows.size)
            sel = np.argpartition(-sub, k - 1)[:k]
            sel = sel[np.argsort(-sub[sel], kind="stable")]
            return [
                (self.passage_ids[int(rows[i])], float(sub[i]))
                for i in sel
                if sub[i] > 0.0
            ]
        k = min(top_k, self.n_docs)
        if k == 0:
            return []
        sel = np.argpartition(-scores, k - 1)[:k]
        sel = sel[np.argsort(-scores[sel], kind="stable")]
        return [
            (self.passage_ids[int(i)], float(scores[i])) for i in sel if scores[i] > 0.0
        ]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "cfg": self.cfg.to_dict(),
                    "passage_ids": self.passage_ids,
                    "vocab": self._vocab,
                    "idf": self._idf,
                    "offsets": self._offsets,
                    "rows": self._rows,
                    "tfs": self._tfs,
                    "n_docs": self.n_docs,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25FIndex":
        with path.open("rb") as fh:
            d = pickle.load(fh)
        idx = cls(BM25Config.from_dict(d["cfg"]))
        idx.passage_ids = d["passage_ids"]
        idx._pid_to_row = {p: i for i, p in enumerate(idx.passage_ids)}
        idx._vocab = d["vocab"]
        idx._idf = d["idf"]
        idx._offsets = d["offsets"]
        idx._rows = d["rows"]
        idx._tfs = d["tfs"]
        idx.n_docs = d["n_docs"]
        return idx
