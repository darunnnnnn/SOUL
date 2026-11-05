"""Evaluation runner: systems -> per-question raw outputs on disk.

Separation of concerns (this is what Hard Requirement #4 asks for)
------------------------------------------------------------------
``runner.py`` *only* produces raw per-question records. ``report.py`` *only*
consumes them and computes every aggregate, interval and p-value. Nothing is
aggregated during the run. Consequences:

* ``ctxlayer report`` rebuilds every number in EVAL.md from the committed raw
  outputs without re-running retrieval or any model.
* A reviewer can recompute a metric we did not think of, or re-derive ours, from
  the artefacts in the repository.
* A bug in aggregation can never corrupt the evidence, only the view of it.

Fairness guarantees, enforced in code
-------------------------------------
1. **One knowledge base.** Every system is handed the same ``KnowledgeBase``
   instance, so no system can see differently-built indexes.
2. **One reader.** The same reader object instance serves every system, so
   answer deltas cannot come from a different generator.
3. **Equal token budget.** Baselines run through the same assembly budget
   (``assemble.enabled=False`` still applies ``budget_words``), so a baseline is
   never starved of context nor our system quietly given more. Context length is
   reported per system so budget parity is auditable, not just asserted.
4. **No gold leakage.** ``_assert_no_leakage`` raises if a non-oracle system is
   given gold labels. Gold is used only for scoring and for ``mode='oracle'``.
5. **Fixed seeds.** Re-seeded before every system.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .._env import seed_everything
from ..config import EvalConfig, PathsConfig, SystemConfig
from ..index.store import KnowledgeBase
from ..ingest.musique import IngestResult
from ..pipeline import ContextLayer
from ..retrieve.decompose import analyze_question
from ..schema import Question
from ..synth.cite import attach_evidence_chain, verify_citations
from ..synth.reader import ExtractiveReader
from .metrics import (
    best_over_aliases,
    citation_metrics,
    context_metrics,
    retrieval_metrics,
)

__all__ = ["run_system", "raw_output_path", "environment_fingerprint"]


_ENV_FINGERPRINT: dict[str, Any] | None = None


def environment_fingerprint() -> dict[str, Any]:
    """Everything needed to explain a numerical difference between machines.

    Computed **once per process** and cached. The code version that produced a
    run is the one loaded at import time, not whatever HEAD happens to be when
    a particular system starts -- so re-reading git per system would stamp
    different commits onto systems that ran identical code, and the
    mixed-commit guard in report.py would then reject a perfectly valid run.
    (Which it duly did, when a documentation commit landed mid-evaluation.)
    """
    global _ENV_FINGERPRINT
    if _ENV_FINGERPRINT is not None:
        return dict(_ENV_FINGERPRINT)
    _ENV_FINGERPRINT = _compute_environment_fingerprint()
    return dict(_ENV_FINGERPRINT)


def _compute_environment_fingerprint() -> dict[str, Any]:
    try:
        import numpy
        import torch
        import transformers

        libs = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
        }
    except Exception:  # pragma: no cover
        libs = {}
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "git_commit": commit,
        **libs,
    }


def raw_output_path(paths: PathsConfig, setting: str, system: str) -> Path:
    out = paths.resolve("raw_outputs_dir") / setting
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{system}.jsonl"


def _assert_no_leakage(cfg: SystemConfig, q: Question) -> None:
    """Gold labels must not reach an evaluated system.

    ``mode='oracle'`` is the single exception, and it is reported as a reference
    row rather than as one of our systems.
    """
    if cfg.mode != "oracle" and getattr(cfg, "_gold_access", False):
        raise AssertionError(
            f"system {cfg.name!r} attempted to access gold labels for {q.qid}"
        )


def run_system(
    kb: KnowledgeBase,
    cfg: SystemConfig,
    questions: Sequence[Question],
    ecfg: EvalConfig,
    paths: PathsConfig,
    reader: ExtractiveReader | None = None,
    setting: str | None = None,
    verbose: bool = True,
    progress_every: int = 25,
) -> Path:
    """Run one system over the question set, writing one JSON record per line."""
    setting = setting or ecfg.setting
    seed_everything(ecfg.seed)
    path = raw_output_path(paths, ecfg.label, cfg.name)

    p2d = {p.passage_id: p.doc_id for p in kb.passages.values()}
    a2p = {a.anchor_id: a.passage_id for a in kb.anchors.values()}
    a2d = {a.anchor_id: a.doc_id for a in kb.anchors.values()}

    layer = ContextLayer(kb, cfg)
    t_start = time.time()
    n = len(questions)

    with path.open("w", encoding="utf-8") as fh:
        # A self-describing header: the raw outputs carry the exact config,
        # environment and corpus fingerprint that produced them, so a stale
        # artefact can always be detected.
        fh.write(
            json.dumps(
                {
                    "_record": "header",
                    "system": cfg.name,
                    "description": cfg.description,
                    "config": cfg.to_dict(),
                    "config_fingerprint": cfg.fingerprint(),
                    "setting": setting,
                    "eval_config": ecfg.to_dict(),
                    "n_questions": n,
                    "corpus": {
                        "n_passages": kb.n_passages,
                        "n_documents": kb.n_documents,
                        "source_sha256": kb.stats.get("source_sha256"),
                        "questions_sha256": kb.stats.get("questions_sha256"),
                        "selection": kb.stats.get("selection"),
                        "selection_params": kb.stats.get("selection_params"),
                    },
                    "reader": (
                        {"kind": cfg.reader.kind, "model": cfg.reader.model}
                        if reader is not None
                        else {"kind": "none"}
                    ),
                    "environment": environment_fingerprint(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        for i, q in enumerate(questions, start=1):
            _assert_no_leakage(cfg, q)
            t0 = time.time()
            ctx, ranked = layer.build_context_for(q, setting=setting)
            t_ctx = (time.time() - t0) * 1000

            ranked_ids = [s.passage_id for s in ranked]
            rec: dict[str, Any] = {
                "qid": q.qid,
                "hop_type": q.hop_type,
                "n_hops": q.n_hops,
                "question": q.question,
                "gold_answer": q.answer,
                "gold_answer_aliases": q.answer_aliases,
                "gold_passage_ids": q.gold_passage_ids,
                "gold_doc_ids": q.gold_doc_ids,
                "gold_doc_titles": q.gold_doc_titles,
                # Truncated to 50: enough for every reported k, bounded size.
                "ranked_passage_ids": ranked_ids[:50],
                "ranked_scores": [round(float(s.score), 6) for s in ranked[:50]],
                "ranked_hops": [int(s.hop) for s in ranked[:50]],
                "context_passage_ids": ctx.passage_ids,
                "context_doc_ids": ctx.doc_ids,
                "context_doc_titles": [p.doc_title for p in ctx.passages],
                "context_words": ctx.n_words,
                "context_ms": round(t_ctx, 1),
            }

            rec["metrics"] = retrieval_metrics(
                ranked_ids, q.gold_passage_ids, q.gold_doc_ids, p2d, ecfg.k_values
            )
            rec["metrics"].update(
                context_metrics(
                    ctx.passage_ids,
                    q.gold_passage_ids,
                    q.gold_doc_ids,
                    p2d,
                    ctx.n_words,
                )
            )

            # Retrieval trace: bridge entities, rounds, query count. This is
            # what the failure analysis in EVAL.md is built from.
            rec["trace"] = {
                "n_queries": len(ctx.trace.queries_issued),
                "queries": ctx.trace.queries_issued[:12],
                "bridge_entities": ctx.trace.bridge_entities[:12],
                "rounds": ctx.trace.rounds[:12],
                "timings_ms": {k: round(v, 1) for k, v in ctx.trace.timings_ms.items()},
            }

            # ---- synthesis -------------------------------------------------
            if reader is not None:
                plan = analyze_question(q.question)
                t1 = time.time()
                ans = reader.read(q.question, ctx, focus=plan.focus)
                t_read = (time.time() - t1) * 1000
                # Ground the intermediate hops the retriever actually followed,
                # then verify every claim -- chain claims included -- with the
                # same machinery. A single-shot system has no chain and so is
                # unaffected, which is the honest outcome rather than a
                # handicap (see synth.cite.build_evidence_chain).
                attach_evidence_chain(kb, ans)
                report = verify_citations(kb, ans)
                em, f1 = best_over_aliases(ans.answer, q.all_answers())
                cited = ans.cited_anchor_ids()
                rec["answer"] = {
                    "text": ans.answer,
                    "abstained": ans.abstained,
                    "claims": [c.to_dict() for c in ans.claims],
                    "meta": {
                        k: v
                        for k, v in ans.meta.items()
                        if k not in {"citation_report"}
                    },
                    "read_ms": round(t_read, 1),
                }
                rec["metrics"]["answer_em"] = em
                rec["metrics"]["answer_f1"] = f1
                rec["metrics"]["abstained"] = float(ans.abstained)
                rec["metrics"].update(
                    citation_metrics(
                        cited,
                        {a: kb.verify_anchor(a) for a in cited},
                        a2p,
                        a2d,
                        q.gold_passage_ids,
                        q.gold_doc_ids,
                    )
                )
                rec["citation_report"] = report.to_dict()

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if verbose and (i % progress_every == 0 or i == n):
                el = time.time() - t_start
                eta = el / i * (n - i)
                print(
                    f"\r  {cfg.name:16s} {i}/{n}  {el:6.1f}s elapsed  "
                    f"{eta:6.1f}s eta",
                    end="",
                    flush=True,
                )

    if verbose:
        stats = kb.cache_stats()
        extra = (
            f"  [cross-encoder: {int(stats.get('n_cross_encoder_calls', 0))} scored, "
            f"cache hit {stats.get('cache_hit_rate', 0):.0%}]"
            if stats
            else ""
        )
        print(f"\r  {cfg.name:16s} {n}/{n}  {time.time()-t_start:6.1f}s{extra}")
    return path
