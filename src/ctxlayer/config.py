"""Typed, serialisable configuration for every stage of the pipeline.

Design rule: **a system is fully determined by its config**. The evaluation
runner instantiates systems from these dataclasses alone, which is what makes
ablations honest -- an ablation is literally one field changed, so there is no
chance of an unintended second difference creeping into a comparison.

Configs round-trip to/from YAML-ish plain dicts (``to_dict`` / ``from_dict``)
and every raw eval output embeds the exact config that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PathsConfig",
    "IngestConfig",
    "BM25Config",
    "DenseConfig",
    "FusionConfig",
    "RerankConfig",
    "GraphConfig",
    "IterativeConfig",
    "AssembleConfig",
    "ReaderConfig",
    "SystemConfig",
    "EvalConfig",
    "load_config",
    "PRESETS",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Construct a (possibly nested) dataclass from a plain dict.

    Unknown keys raise rather than being silently dropped: a typo in an
    ablation config must not quietly evaluate the default system.
    """
    if not is_dataclass(cls):
        return data
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, val in data.items():
        ftype = known[name].type
        if isinstance(val, dict) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, val)  # type: ignore[arg-type]
        else:
            kwargs[name] = val
    return cls(**kwargs)


@dataclass
class _Base:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        return _from_dict(cls, data)

    def replace(self, **kw: Any) -> Any:
        """Return a copy with fields overridden (dotted paths supported).

        ``cfg.replace(**{"assemble.max_per_doc": 1})`` is how ablations are
        expressed, so an ablation is provably a single-field delta.
        """
        import copy

        out = copy.deepcopy(self)
        for key, val in kw.items():
            target: Any = out
            *path, leaf = key.split(".")
            for part in path:
                target = getattr(target, part)
            if not hasattr(target, leaf):
                raise ValueError(f"no such config field: {key}")
            setattr(target, leaf, val)
        return out


# --------------------------------------------------------------------------
# stage configs
# --------------------------------------------------------------------------


@dataclass
class PathsConfig(_Base):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"
    cache_dir: str = "data/cache"
    results_dir: str = "eval/results"
    raw_outputs_dir: str = "eval/raw_outputs"

    def resolve(self, which: str) -> Path:
        p = Path(getattr(self, which))
        return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class IngestConfig(_Base):
    dataset: str = "musique"
    split: str = "validation"
    source_file: str = "musique_ans_v1.0_dev.jsonl"
    # Subset selection (see EVAL.md). All selection is deterministic.
    subset_size: int = 500
    seed: int = 20251105
    # Hard Requirement #2: drop questions whose gold support does not span at
    # least this many DISTINCT documents. MuSiQue-Ans dev contains 54 such
    # questions; including them would let a single-document answer count as a
    # multi-document success.
    min_gold_docs: int = 2
    # Stratify the subset over hop types so 3/4-hop questions are not drowned
    # out by the 2-hop majority.
    stratify_by_hops: bool = True


@dataclass
class BM25Config(_Base):
    """BM25F over two fields (title, body).

    Multi-hop questions hinge on rare named entities, where exact lexical match
    is far stronger than any dense embedding. Titles get their own field weight
    because a bridge entity is usually *the title* of the document that answers
    the next hop.
    """

    k1: float = 0.9
    b_title: float = 0.4
    b_body: float = 0.75
    w_title: float = 2.5
    w_body: float = 1.0
    enabled: bool = True


@dataclass
class DenseConfig(_Base):
    model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect an asymmetric instruction prefix on the QUERY side only.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    max_length: int = 384
    batch_size: int = 64
    normalize: bool = True
    device: str = "auto"
    enabled: bool = True


@dataclass
class FusionConfig(_Base):
    """Weighted Reciprocal Rank Fusion.

    RRF is used rather than score interpolation because BM25 and cosine scores
    live on incomparable, corpus-dependent scales; rank-space fusion needs no
    per-corpus calibration and is markedly more robust.
    """

    method: str = "rrf"  # rrf | bm25_only | dense_only | linear
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_dense: float = 1.0
    candidates_per_signal: int = 100


@dataclass
class RerankConfig(_Base):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_in: int = 60
    max_length: int = 320
    batch_size: int = 64
    device: str = "auto"


@dataclass
class GraphConfig(_Base):
    """Entity bridge graph built at ingestion time.

    Edges are *document-title mentions*: if passage p (in document A) contains a
    surface form that matches the title of document B, we add A -> B. This
    reconstructs, from text alone, the Wikipedia link structure that multi-hop
    questions were built on top of -- giving the retriever a way to reach a
    second-hop document whose text shares almost no vocabulary with the
    question.
    """

    enabled: bool = True
    min_title_chars: int = 4
    min_title_words: int = 1
    # Titles that are common words make catastrophic edges ("Green", "List of
    # ...", "The Times"). Require the surface form to be capitalised in context
    # and not be a stopword-only string.
    require_capitalised: bool = True
    max_edges_per_passage: int = 24
    drop_title_prefixes: tuple[str, ...] = ("List of", "Index of", "Outline of")


@dataclass
class IterativeConfig(_Base):
    """Multi-round, entity-bridged retrieval -- the core of the context layer.

    ``max_rounds`` is the hop budget. ``gain_threshold`` implements an adaptive
    stop: if a round fails to add meaningfully-new evidence, we stop early,
    which keeps 2-hop questions cheap while letting 4-hop questions spend more.
    """

    enabled: bool = True
    max_rounds: int = 3
    per_round_k: int = 20
    # How many frontier entities to expand per round.
    beam_entities: int = 4
    # Expand from the top-N passages of the previous round only.
    frontier_passages: int = 5
    # Stop early when a round's best new passage scores below this fraction of
    # the running best.
    gain_threshold: float = 0.55
    # Use the graph for expansion in addition to text queries.
    use_graph: bool = True
    # Re-issue the residual (unanswered) part of the question with the bridge
    # entity substituted in.
    use_query_rewrite: bool = True


@dataclass
class AssembleConfig(_Base):
    """Coverage-aware context assembly.

    A plain top-k slice routinely spends its whole budget on near-duplicate
    passages from one document -- fatal for multi-hop. Assembly therefore
    maximises a joint objective over (relevance, query-term coverage,
    document diversity) under a token budget, with an explicit per-document cap.
    """

    budget_words: int = 900
    max_passages: int = 12
    # Hard cap on passages taken from any single document.
    max_per_doc: int = 2
    # Maximal-marginal-relevance tradeoff; 1.0 = pure relevance.
    mmr_lambda: float = 0.65
    # Weight on marginal coverage of question content terms.
    coverage_weight: float = 0.35
    # Guarantee at least this many distinct documents if candidates allow.
    min_docs: int = 2
    enabled: bool = True


@dataclass
class ReaderConfig(_Base):
    """The reader is held FIXED across every system and baseline.

    Holding the generator constant and varying only the context layer is what
    makes answer-level deltas attributable to retrieval and assembly rather
    than to prompt or model luck.
    """

    kind: str = "extractive"  # extractive | llm | none
    model: str = "deepset/roberta-base-squad2"
    max_length: int = 384
    stride: int = 128
    max_answer_words: int = 12
    batch_size: int = 16
    device: str = "auto"
    # LLM reader (used only when an API key is configured).
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    # Abstain when the reader's calibrated confidence is below this.
    abstain_threshold: float = 0.0


@dataclass
class SystemConfig(_Base):
    """A complete, named, reproducible QA system."""

    name: str = "ours"
    description: str = ""
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    iterative: IterativeConfig = field(default_factory=IterativeConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    # "oracle" and "closed_book" are special-cased in the runner.
    mode: str = "retrieval"  # retrieval | oracle | closed_book

    def fingerprint(self) -> str:
        from .schema import stable_id

        return stable_id(json.dumps(self.to_dict(), sort_keys=True), length=10)


@dataclass
class EvalConfig(_Base):
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    # Retrieval setting: "pool" = standard per-question 20-paragraph distractor
    # pool; "unified" = one shared knowledge base over every passage in the
    # subset (much harder, and the realistic deployment).
    setting: str = "unified"
    # k values at which retrieval metrics are reported.
    k_values: tuple[int, ...] = (2, 5, 10, 20)
    bootstrap_samples: int = 2000
    permutation_samples: int = 10000
    seed: int = 20251105


# --------------------------------------------------------------------------
# System presets: the baselines and our system
# --------------------------------------------------------------------------


def _base() -> SystemConfig:
    return SystemConfig()


def _off(cfg: SystemConfig, *paths: str) -> SystemConfig:
    out = cfg
    for p in paths:
        out = out.replace(**{f"{p}.enabled": False})
    return out


def _presets() -> dict[str, SystemConfig]:
    """Credible standard baselines, strongest-first reasoning in EVAL.md."""
    p: dict[str, SystemConfig] = {}

    # --- B0: no retrieval. Measures parametric leakage / contamination. ---
    p["closed_book"] = _base().replace(
        name="closed_book",
        description="No retrieval. Isolates how much the benchmark can be "
        "answered from the reader's parameters alone.",
        mode="closed_book",
    )

    # --- B1: BM25 top-k. The classic sparse baseline. ---
    p["bm25"] = _off(
        _base().replace(
            name="bm25",
            description="BM25F top-k, no fusion/rerank/iteration. Classic "
            "sparse retrieval baseline.",
            **{"fusion.method": "bm25_only"},
        ),
        "dense",
        "rerank",
        "graph",
        "iterative",
        "assemble",
    )

    # --- B2: dense top-k. The standard single-vector embedding baseline. ---
    p["dense"] = _off(
        _base().replace(
            name="dense",
            description="Dense bi-encoder (bge-small-en-v1.5) top-k. Standard "
            "single-vector embedding baseline.",
            **{"fusion.method": "dense_only"},
        ),
        "bm25",
        "rerank",
        "graph",
        "iterative",
        "assemble",
    )

    # --- B3: hybrid RRF. Strong, very widely deployed. ---
    p["hybrid"] = _off(
        _base().replace(
            name="hybrid",
            description="Hybrid BM25F + dense with weighted RRF. Strong, "
            "widely deployed production baseline.",
        ),
        "rerank",
        "graph",
        "iterative",
        "assemble",
    )

    # --- B4: hybrid + cross-encoder rerank. "Standard RAG at its best". ---
    p["hybrid_rerank"] = _off(
        _base().replace(
            name="hybrid_rerank",
            description="Hybrid RRF + cross-encoder reranking. The strongest "
            "single-shot pipeline in common use; our primary comparison point.",
        ),
        "graph",
        "iterative",
        "assemble",
    )

    # --- OURS: iterative, graph-bridged, coverage-assembled. ---
    p["ours"] = _base().replace(
        name="ours",
        description="Iterative entity-bridged multi-hop retrieval with "
        "graph expansion, cross-encoder reranking and coverage-aware "
        "multi-document context assembly.",
    )

    # --- Oracle: gold supporting paragraphs. Reader-ceiling reference. ---
    p["oracle"] = _base().replace(
        name="oracle",
        description="Gold supporting paragraphs only. Upper bound given the "
        "fixed reader; separates retrieval error from reading error.",
        mode="oracle",
    )
    return p


PRESETS: dict[str, SystemConfig] = _presets()


def load_config(path: str | Path) -> SystemConfig:
    """Load a SystemConfig from a JSON or simple-YAML file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".json"}:
        data = json.loads(text)
    else:
        data = _tiny_yaml(text)
    base_name = data.pop("extends", None)
    if base_name:
        base = PRESETS[base_name]
        overrides = _flatten(data)
        return base.replace(**overrides)
    return SystemConfig.from_dict(data)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _tiny_yaml(text: str) -> dict[str, Any]:
    """Minimal 2-space-indent YAML subset: scalars, nesting, no lists.

    Avoids a PyYAML dependency for the handful of config files we ship. Raises
    loudly on anything it does not understand rather than guessing.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            raise ValueError(f"config line {lineno}: expected 'key: value'")
        key, _, val = line.strip().partition(":")
        key, val = key.strip(), val.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _scalar(val)
    return root


def _scalar(v: str) -> Any:
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip("\"'")
