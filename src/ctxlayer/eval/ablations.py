"""Ablation suite.

Each ablation is expressed as a dict of **config overrides applied to the
``ours`` preset**, so by construction it differs from the full system in exactly
the listed fields and nothing else -- there is no separate code path, no second
prompt, no re-tuned hyperparameter. ``SystemConfig.replace`` raises on an
unknown field, so a typo produces an error rather than silently evaluating the
unmodified system.

The suite is designed to answer specific questions rather than to be long:

*Which component carries the gain?*
    ``no_iterative``, ``no_assembly``, ``no_rerank`` remove one pillar each.
    ``no_iterative`` is the important one -- it reduces our system to
    "hybrid + rerank + coverage assembly", isolating the multi-hop loop.

*Is the mention index actually doing the work, or would any second query do?*
    ``no_graph`` keeps the iteration but draws bridge candidates from open
    retrieval instead of exact mention containment. ``no_query_rewrite`` keeps
    the mention index but searches the bare entity without the question's
    relation terms.

*Is the multi-document constraint load-bearing?*
    ``no_doc_cap`` lifts the per-document cap; ``no_coverage`` zeroes the
    coverage term. These test the Hard-Requirement-#2 machinery directly.

*Are the hyperparameters chosen or lucky?*
    ``rounds_*``, ``beam_*``, ``fusion_linear``, ``bm25_no_title_boost``,
    ``rrf_*`` sweep the main knobs. A system whose gain survives its knobs
    being moved is a better claim than a single tuned point.

Note that ``bm25_*`` ablations change the *index*, not just the query path, so
``run_ablations`` rebuilds the BM25F index for those runs. Running them against
an index built with different field weights would silently measure nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..config import PRESETS, EvalConfig, PathsConfig, SystemConfig
from ..index.bm25 import BM25FIndex
from ..index.store import KnowledgeBase
from ..ingest.musique import load_processed

__all__ = ["ABLATIONS", "run_ablations", "build_ablation_config"]


# name -> (description, overrides on the `ours` preset)
ABLATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    # ---- component removal -------------------------------------------------
    "no_iterative": (
        "Remove multi-round bridged retrieval (single shot + rerank + assembly). "
        "Isolates the contribution of the multi-hop loop itself.",
        {"iterative.enabled": False},
    ),
    "no_assembly": (
        "Remove coverage/diversity-aware assembly; plain top-k under the same "
        "word budget. Isolates set selection from ranking.",
        {"assemble.enabled": False},
    ),
    "no_rerank": (
        "Remove the cross-encoder everywhere (fused scores only).",
        {"rerank.enabled": False},
    ),
    # ---- mechanism of the bridge -------------------------------------------
    "no_graph": (
        "Keep iteration but generate bridge candidates by open retrieval "
        "instead of exact mention containment. Tests whether the mention index "
        "matters or any second query would do.",
        {"iterative.use_graph": False},
    ),
    "no_query_rewrite": (
        "Keep the mention index but query the bare bridge entity, dropping the "
        "question's relation terms. Tests the 'entity + relation' construction.",
        {"iterative.use_query_rewrite": False},
    ),
    # ---- the multi-document constraint -------------------------------------
    "no_doc_cap": (
        "Lift the per-document cap (max_per_doc=99). Directly tests the "
        "constraint that stops one source consuming the whole budget.",
        {"assemble.max_per_doc": 99},
    ),
    "no_coverage": (
        "Zero the coverage term, leaving relevance + MMR diversity only.",
        {"assemble.coverage_weight": 0.0},
    ),
    "floor_all": (
        "Apply the relevance floor to every candidate instead of the tail "
        "only. Trades recall for a markedly smaller context: the other end of "
        "the quality/token frontier.",
        {"assemble.floor_exempt_top_k": False},
    ),
    "no_floor": (
        "Remove the relevance floor entirely.",
        {"assemble.min_rel_ratio": 0.0},
    ),
    "no_mmr": (
        "Pure relevance in assembly (mmr_lambda=1.0), keeping the doc cap.",
        {"assemble.mmr_lambda": 1.0},
    ),
    # ---- hop budget --------------------------------------------------------
    "rounds_1": ("Hop budget 1 (2 hops of evidence).", {"iterative.max_rounds": 1}),
    "rounds_2": ("Hop budget 2 (3 hops).", {"iterative.max_rounds": 2}),
    "rounds_4": ("Hop budget 4 (5 hops); tests over-expansion.",
                 {"iterative.max_rounds": 4}),
    "no_adaptive_stop": (
        "Disable the adaptive stop (always spend the full hop budget). Tests "
        "whether early stopping costs recall or only saves compute.",
        {"iterative.gain_threshold": 0.0},
    ),
    # ---- beam --------------------------------------------------------------
    "beam_2": ("Narrower bridge beam (2 entities/round).", {"iterative.beam_entities": 2}),
    "beam_8": ("Wider bridge beam (8 entities/round).", {"iterative.beam_entities": 8}),
    # ---- retrieval primitives ---------------------------------------------
    "fusion_linear": (
        "Min-max linear score interpolation instead of RRF.",
        {"fusion.method": "linear"},
    ),
    "rrf_k_10": ("RRF k=10 (sharper top-rank emphasis).", {"fusion.rrf_k": 10}),
    "rrf_k_200": ("RRF k=200 (flatter).", {"fusion.rrf_k": 200}),
    "bm25_no_title_boost": (
        "BM25F title field weight 1.0 instead of 2.5 (rebuilds the index). "
        "Tests the fielded-scoring design decision.",
        {"bm25.w_title": 1.0},
    ),
    # ---- budget ------------------------------------------------------------
    "budget_500w": ("Halve the context budget to 500 words.",
                    {"assemble.budget_words": 500}),
    "budget_1800w": ("Double the context budget to 1800 words.",
                     {"assemble.budget_words": 1800}),
}


def build_ablation_config(name: str) -> SystemConfig:
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
    desc, overrides = ABLATIONS[name]
    cfg = PRESETS["ours"].replace(**overrides)
    cfg.name = f"abl_{name}"
    cfg.description = desc
    return cfg


def _touches_index(overrides: dict[str, Any]) -> bool:
    """Does this ablation change how the BM25F index itself is built?"""
    return any(k.startswith("bm25.") for k in overrides)


def run_ablations(
    paths: PathsConfig,
    ecfg: EvalConfig,
    limit: int | None = None,
    only: Sequence[str] | None = None,
    kb: KnowledgeBase | None = None,
    reader: Any = None,
    no_reader: bool = False,
    reference: bool = True,
) -> int:
    from .runner import run_system

    names = list(only) if only else list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        print(f"unknown ablations: {unknown}")
        return 1

    res = load_processed(paths)
    base_cfg = PRESETS["ours"]
    if kb is None:
        try:
            kb = KnowledgeBase.load(res, base_cfg, paths)
        except FileNotFoundError:
            kb = KnowledgeBase.build(res, base_cfg, paths, verbose=True)
            kb.save(paths, base_cfg)
        kb.enable_score_cache(paths, base_cfg)

    if reader is None and not no_reader:
        from ..index.score_cache import BlobCache
        from ..synth.reader import ExtractiveReader

        reader = ExtractiveReader(
            base_cfg.reader,
            cache=BlobCache(
                paths.resolve("cache_dir") / "reader_answers.sqlite", "reader-v1"
            ),
        )

    questions = res.questions[:limit] if limit else res.questions
    pristine_bm25 = kb.bm25

    print(f"\nrunning {len(names)} ablations on {len(questions)} questions "
          f"({ecfg.setting} setting, output label {ecfg.label!r})\n")

    # Reference runs *inside the ablation directory*, on exactly these
    # questions. Without them an ablation table has nothing valid to be
    # compared against: report.py requires every run in a directory to cover an
    # identical question set, so the main directory's 500-question `ours` can
    # not serve as the reference for a 200-question ablation sweep.
    if reference:
        for ref in ("ours", "hybrid_rerank"):
            kb.bm25 = pristine_bm25
            run_system(kb, PRESETS[ref], questions, ecfg, paths, reader=reader,
                       setting=ecfg.setting)

    for name in names:
        cfg = build_ablation_config(name)
        _, overrides = ABLATIONS[name]
        if _touches_index(overrides):
            # Rebuild BM25F with the ablated field weights, else the ablation
            # would be a no-op measured against the original index.
            print(f"  (rebuilding BM25F for {name}: {overrides})")
            ordered = [res.passages[p] for p in sorted(res.passages)]
            kb.bm25 = BM25FIndex(cfg.bm25).build(ordered)
        else:
            kb.bm25 = pristine_bm25
        run_system(kb, cfg, questions, ecfg, paths, reader=reader,
                   setting=ecfg.setting)
    kb.bm25 = pristine_bm25
    print("\nablation raw outputs ->", paths.resolve("raw_outputs_dir") / ecfg.label)
    return 0
