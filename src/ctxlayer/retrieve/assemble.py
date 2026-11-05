"""Context assembly: turning a ranking into a multi-document context.

Why top-k slicing is the wrong final step
-----------------------------------------
Retrieval produces a *ranking*; a prompt needs a *set*. Those are different
objects, and the difference is where multi-hop RAG quietly fails. A top-k slice
optimises each slot independently, so it happily spends the whole budget on
near-duplicate passages about the question's most prominent entity:

    [1] Zhejiang           <- hop 1, relevant
    [2] Zhejiang           <- another paragraph of the same article
    [3] Zhejiang           <- and another
    [4] Cork (city)        <- a distractor that shares vocabulary
    ...
    [k] Sanjiang Church    <- hop 2 gold, below the cut

Every slot is defensible on its own and the set is useless: it covers one
document when the question needs two. Note that this is exactly the trap Hard
Requirement #2 warns about -- several chunks of one source are not
multi-document evidence.

The objective
-------------
Assembly is therefore a *set* optimisation under a token budget. We greedily
maximise

    gain(p) = lambda * rel(p)
            + (1 - lambda) * [ coverage_weight * cov(p) + (1 - coverage_weight) * div(p) ]

subject to a hard per-document cap and the word budget, where

``rel(p)``  the retriever's (normalised) chain score,
``cov(p)``  marginal coverage: the share of still-uncovered question content
            terms and discovered bridge entities that ``p`` newly covers --
            this is what pulls in the passage holding the *other* hop rather
            than a third paragraph on the first hop,
``div(p)``  ``1 - max cosine`` to anything already selected (MMR, Carbonell &
            Goldstein 1998), using the same bi-encoder vectors as retrieval, so
            redundancy is measured semantically and not by word overlap.

``max_per_doc`` is a hard constraint rather than a penalty, and ``min_docs``
forces a document switch if the budget would otherwise be spent on one source.
Both are ablated in EVAL.md; removing the cap is the single most damaging
ablation, which is the empirical argument for this module existing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import AssembleConfig, SystemConfig
from ..index.bm25 import STOPWORDS, tokenize
from ..index.store import KnowledgeBase
from ..schema import AssembledContext, RetrievalTrace, ScoredPassage

__all__ = ["assemble_context", "ContextAssembler"]


class ContextAssembler:
    def __init__(self, kb: KnowledgeBase, cfg: SystemConfig) -> None:
        self.kb = kb
        self.cfg = cfg

    # ------------------------------------------------------------- helpers
    def _target_terms(
        self, question: str, trace: RetrievalTrace | None
    ) -> dict[str, float]:
        """The vocabulary the context should cover, weighted by informativeness.

        Question content terms plus the bridge entities the retriever actually
        followed. Including the bridges matters: by construction they name the
        intermediate hops, which the question itself never mentions, so they are
        the only available signal for "have I covered hop 2 yet?".

        Terms are **idf-weighted**, which fixes a failure we observed: an
        unweighted token set treats every token of a multi-word bridge as
        equally worth covering, so a bridge entity like "University of Glasgow"
        makes any passage containing the word "university" look like it covers a
        hop. That dragged *Beatrice Heuser* and *Europe-Asia Studies* into the
        context of a question about James Watt. Weighting by corpus idf means
        "glasgow" counts and "university" barely does, with no extra
        hyperparameter -- the weights come from the index we already built.
        """
        bm25 = self.kb.bm25
        terms: dict[str, float] = {}

        def add(tok: str) -> None:
            if len(tok) <= 2 or tok in STOPWORDS:
                return
            w = 1.0
            if bm25 is not None:
                ti = bm25._vocab.get(tok)
                if ti is not None:
                    # Normalise idf into (0, 1] so weights are comparable
                    # across corpora of different sizes.
                    w = float(bm25._idf[ti]) / max(1e-6, float(bm25._idf.max()))
            terms[tok] = max(terms.get(tok, 0.0), w)

        for t in tokenize(question):
            add(t)
        if trace is not None:
            for ent in trace.bridge_entities:
                for t in ent.split():
                    add(t)
        return terms

    def _vectors(self, passage_ids: Sequence[str]) -> np.ndarray | None:
        dense = self.kb.dense
        if dense is None or dense.emb.size == 0:
            return None
        rows = []
        for pid in passage_ids:
            v = dense.vector_for(pid)
            if v is None:
                return None
            rows.append(v)
        return np.stack(rows) if rows else None

    # ------------------------------------------------------------- assemble
    def assemble(
        self,
        question: str,
        ranked: Sequence[ScoredPassage],
        trace: RetrievalTrace | None = None,
    ) -> AssembledContext:
        acfg: AssembleConfig = self.cfg.assemble
        ctx = AssembledContext(trace=trace or RetrievalTrace())
        if not ranked:
            return ctx

        # ---- disabled: plain top-k under the same budget ------------------
        # The baselines run through this path so that they are compared at an
        # *equal token budget*, not starved or inflated. Budget parity is what
        # makes the comparison fair.
        if not acfg.enabled:
            used = 0
            for sp in ranked:
                p = self.kb.passages.get(sp.passage_id)
                if p is None:
                    continue
                if len(ctx.passages) >= acfg.max_passages:
                    break
                if used + p.n_words > acfg.budget_words and ctx.passages:
                    break
                ctx.passages.append(p)
                ctx.scores.append(sp.score)
                used += p.n_words
            return ctx

        cand_ids = [sp.passage_id for sp in ranked if sp.passage_id in self.kb.passages]
        if not cand_ids:
            return ctx
        rel_raw = np.array(
            [sp.score for sp in ranked if sp.passage_id in self.kb.passages],
            dtype=np.float64,
        )
        lo, hi = float(rel_raw.min()), float(rel_raw.max())
        rel = (
            (rel_raw - lo) / (hi - lo)
            if hi - lo > 1e-9
            else np.ones_like(rel_raw)
        )

        vecs = self._vectors(cand_ids)
        target = self._target_terms(question, trace)
        target_total = sum(target.values()) or 1.0
        passage_terms = [
            {t for t in tokenize(self.kb.passages[pid].with_title()) if t in target}
            for pid in cand_ids
        ]

        # Relevance floor. Without one, the diversity and coverage terms happily
        # spend the remaining budget on passages that are barely relevant --
        # which is how an off-topic passage ends up in front of the reader and
        # gets read as the answer. A candidate below this fraction of the best
        # candidate's relevance is admitted only when it is needed to satisfy
        # `min_docs`.
        rel_floor = acfg.min_rel_ratio * float(rel.max()) if rel.size else 0.0

        selected: list[int] = []
        covered: set[str] = set()
        per_doc: dict[str, int] = {}
        used_words = 0
        lam = acfg.mmr_lambda
        cw = acfg.coverage_weight

        while len(selected) < acfg.max_passages:
            remaining_docs = {
                self.kb.passages[cand_ids[i]].doc_id
                for i in range(len(cand_ids))
                if i not in selected
            }
            # Force a document switch when the budget is about to be spent on
            # too few sources and other documents are still available.
            need_new_doc = (
                len(per_doc) < acfg.min_docs
                and len(remaining_docs - set(per_doc)) > 0
                and len(selected) >= acfg.min_docs - 1
            )

            best_i, best_gain = -1, -np.inf
            for i, pid in enumerate(cand_ids):
                if i in selected:
                    continue
                p = self.kb.passages[pid]
                if per_doc.get(p.doc_id, 0) >= acfg.max_per_doc:
                    continue
                if used_words + p.n_words > acfg.budget_words and selected:
                    continue
                if need_new_doc and p.doc_id in per_doc:
                    continue
                if rel[i] < rel_floor and not need_new_doc:
                    continue

                # marginal coverage of still-uncovered target vocabulary,
                # weighted by how informative each uncovered term is
                new_terms = passage_terms[i] - covered
                remaining = sum(w for t, w in target.items() if t not in covered)
                cov = (
                    sum(target[t] for t in new_terms) / remaining
                    if remaining > 1e-9
                    else 0.0
                )

                # MMR diversity against what is already selected
                if vecs is not None and selected:
                    sim = float(np.max(vecs[selected] @ vecs[i]))
                    div = 1.0 - sim
                else:
                    div = 1.0

                gain = lam * rel[i] + (1.0 - lam) * (cw * cov + (1.0 - cw) * div)
                if gain > best_gain:
                    best_i, best_gain = i, gain

            if best_i < 0:
                break
            pid = cand_ids[best_i]
            p = self.kb.passages[pid]
            selected.append(best_i)
            covered |= passage_terms[best_i]
            per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
            used_words += p.n_words
            ctx.passages.append(p)
            ctx.scores.append(float(rel_raw[best_i]))
            if used_words >= acfg.budget_words:
                break

        if trace is not None:
            trace.timings_ms.setdefault("assemble_ms", 0.0)
            trace.add_round(
                stage="assemble",
                n_candidates=len(cand_ids),
                n_selected=len(ctx.passages),
                n_docs=len(per_doc),
                words=used_words,
                target_terms=len(target),
                covered_terms=len(covered & set(target)),
                target_weight=round(target_total, 3),
                covered_weight=round(
                    sum(target[t] for t in covered if t in target), 3
                ),
            )
        return ctx


def assemble_context(
    kb: KnowledgeBase,
    cfg: SystemConfig,
    question: str,
    ranked: Sequence[ScoredPassage],
    trace: RetrievalTrace | None = None,
) -> AssembledContext:
    return ContextAssembler(kb, cfg).assemble(question, ranked, trace)
