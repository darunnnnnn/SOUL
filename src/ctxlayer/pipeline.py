"""The context layer as a single callable, plus the end-to-end QA system.

``ContextLayer.build_context`` is the object the whole assignment is about:
question in, token-budgeted multi-document cited context out. Every evaluated
system -- baselines, ablations and ours -- goes through this one class, with
behaviour determined entirely by ``SystemConfig``. There is no separate "our
system" code path, which is what guarantees the comparisons are fair.
"""

from __future__ import annotations

import time
from typing import Sequence

from .config import SystemConfig
from .index.store import KnowledgeBase
from .retrieve.assemble import ContextAssembler
from .retrieve.hybrid import SingleShotRetriever
from .retrieve.iterative import IterativeRetriever
from .schema import AssembledContext, Question, RetrievalTrace, ScoredPassage

__all__ = ["ContextLayer"]


class ContextLayer:
    def __init__(self, kb: KnowledgeBase, cfg: SystemConfig) -> None:
        self.kb = kb
        self.cfg = cfg
        self.single = SingleShotRetriever(kb, cfg)
        self.iterative = IterativeRetriever(kb, cfg)
        self.assembler = ContextAssembler(kb, cfg)

    # ------------------------------------------------------------- retrieval
    def retrieve(
        self,
        question: str,
        restrict_to: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[ScoredPassage], RetrievalTrace]:
        cfg = self.cfg
        k = top_k or max(cfg.assemble.max_passages * 4, 40)
        if cfg.iterative.enabled:
            return self.iterative.retrieve(
                question, top_k=k, restrict_to=list(restrict_to) if restrict_to else None
            )
        trace = RetrievalTrace()
        ranked = self.single.retrieve(
            question, top_k=k, restrict_to=restrict_to, trace=trace
        )
        return ranked, trace

    # -------------------------------------------------------------- context
    def build_context(
        self,
        question: str,
        restrict_to: Sequence[str] | None = None,
        gold_passage_ids: Sequence[str] | None = None,
    ) -> AssembledContext:
        """Assemble the context for one question.

        ``gold_passage_ids`` is accepted **only** to serve ``mode='oracle'``.
        The runner asserts that no non-oracle system receives it, so gold labels
        cannot leak into an evaluated system.
        """
        cfg = self.cfg
        t0 = time.time()

        if cfg.mode == "closed_book":
            ctx = AssembledContext(trace=RetrievalTrace())
            ctx.trace.timings_ms["total_ms"] = (time.time() - t0) * 1000
            return ctx

        if cfg.mode == "oracle":
            if gold_passage_ids is None:
                raise ValueError("oracle mode requires gold_passage_ids")
            trace = RetrievalTrace()
            ctx = AssembledContext(trace=trace)
            for pid in gold_passage_ids:
                p = self.kb.passages.get(pid)
                if p is not None:
                    ctx.passages.append(p)
                    ctx.scores.append(1.0)
            trace.timings_ms["total_ms"] = (time.time() - t0) * 1000
            return ctx

        ranked, trace = self.retrieve(question, restrict_to=restrict_to)
        ctx = self.assembler.assemble(question, ranked, trace)
        trace.timings_ms["total_ms"] = (time.time() - t0) * 1000
        # Keep the full ranking on the trace: retrieval metrics are computed at
        # several k, which requires the ranking and not just the final context.
        trace.timings_ms.setdefault("assemble_ms", 0.0)
        ctx.trace = trace
        self._last_ranked = ranked
        return ctx

    def build_context_for(
        self, q: Question, setting: str = "unified"
    ) -> tuple[AssembledContext, list[ScoredPassage]]:
        """Context for a benchmark question under the chosen retrieval setting."""
        restrict = q.pool_passage_ids if setting == "pool" else None
        gold = q.gold_passage_ids if self.cfg.mode == "oracle" else None
        ctx = self.build_context(
            q.question, restrict_to=restrict, gold_passage_ids=gold
        )
        ranked = getattr(self, "_last_ranked", [])
        if self.cfg.mode in {"oracle", "closed_book"}:
            ranked = [
                ScoredPassage(passage_id=p.passage_id, score=1.0)
                for p in ctx.passages
            ]
        return ctx, ranked
