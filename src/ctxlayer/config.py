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
    """Entity mention index built at ingestion time.

    This component exists because of a measured failure mode, not a hunch. We
    first built the obvious thing -- a document-link graph keyed on article
    titles -- and measured how often the gold evidence chain is traversable in
    it: **30.4%**. The reason is structural: in MuSiQue the bridge entity of
    hop *n+1* is usually not an article *title* in the corpus, it is a *mention*
    inside the next gold paragraph. A title-keyed graph cannot reach it.

    Replacing title edges with a **surface-form mention index** (every proper-
    noun span in every passage, inverted) raised gold-chain reachability to
    **90.3%** on the same subset. That measurement is reproducible via
    ``python -m ctxlayer.cli diagnose reachability`` and is reported in EVAL.md.

    Phrase containment is used for candidate *generation* rather than BM25 on
    the entity string, because a 2-token entity query returns many partial
    lexical matches, whereas exact span containment is high precision. Scoring
    of those candidates is still done by the fused retriever + cross-encoder.
    """

    enabled: bool = True
    # --- proper-noun span extraction ---
    max_entity_words: int = 5
    min_entity_chars: int = 4
    # A single-token candidate is accepted as a proper noun only if it appears
    # capitalised in at least this fraction of its corpus occurrences. This is
    # a data-driven proper-noun detector: it keeps "Zhejiang" and rejects
    # "Written"/"During" picked up at sentence start, with no NER dependency
    # and no model-version coupling.
    min_cap_ratio: float = 0.6
    # Entities occurring in more than this many passages are non-discriminative
    # as bridges and are dropped from the postings (they would expand the
    # frontier by hundreds of passages for no gain).
    max_entity_df: int = 60
    # --- title edges (kept as an additional high-precision signal) ---
    enable_title_edges: bool = True
    min_title_chars: int = 4
    min_title_words: int = 1
    require_capitalised: bool = True
    max_edges_per_passage: int = 24
    drop_title_prefixes: tuple[str, ...] = ("List of", "Index of", "Outline of")


@dataclass
class IterativeConfig(_Base):
    """Multi-round, entity-bridged retrieval -- the core of the context layer.

    One round = (pick bridge entities from the current evidence frontier) ->
    (generate a bridge query per entity) -> (retrieve + rerank against that
    bridge query) -> (merge into the evidence pool with a chain score).

    ``max_rounds=3`` means up to 4 hops of evidence (round 0 plus 3 expansions),
    matching the deepest MuSiQue questions. ``gain_threshold`` implements an
    adaptive stop so 2-hop questions do not pay for hops they do not need.
    """

    enabled: bool = True
    max_rounds: int = 3
    per_round_k: int = 20
    # How many frontier entities to expand per round (the beam width).
    beam_entities: int = 4
    # Expand from the top-N highest-scoring passages not yet expanded.
    frontier_passages: int = 5
    # Candidate passages pulled from the mention index per bridge entity.
    candidates_per_entity: int = 30
    # Rerank depth for bridge queries. Shallower than round 0 on purpose: the
    # mention index has already constrained candidates to passages that
    # literally contain the bridge entity, so precision@depth is far higher
    # than for an open query and a deep rerank buys little. This is the main
    # lever on total cross-encoder cost (~350 vs ~540 pairs/question).
    bridge_rerank_depth: int = 24
    # Adaptive stop: halt when the best new passage of a round scores below
    # this fraction of the best score seen so far.
    gain_threshold: float = 0.55
    # Use the mention index for candidate generation (vs. plain re-query).
    use_graph: bool = True
    # Carry the question's relation terms into each bridge query.
    use_query_rewrite: bool = True

    # --- chain scoring -------------------------------------------------
    # A hop-2 passage is reranked against its *bridge* query, so its raw score
    # is not comparable to a hop-0 passage's score against the question. Scores
    # are therefore min-max normalised within the round that produced them, and
    # combined into a path score:
    #     chain = w_rel * rel_norm + w_parent * parent_chain + w_entity * idf
    # This keeps a passage's score dependent on how good the evidence that
    # *reached* it was, which is exactly the multi-hop notion of relevance.
    w_rel: float = 0.60
    w_parent: float = 0.30
    w_entity: float = 0.10
    # Decay applied per hop, so deep speculative chains cannot outrank direct
    # evidence on score alone.
    hop_decay: float = 0.92
    # Minimum inverse-document-frequency (normalised 0-1) for an entity to be
    # usable as a bridge: common entities expand the frontier without
    # discriminating.
    min_entity_idf: float = 0.35


@dataclass
class AssembleConfig(_Base):
    """Coverage-aware context assembly.

    A plain top-k slice routinely spends its whole budget on near-duplicate
    passages from one document -- fatal for multi-hop. Assembly therefore
    maximises a joint objective over (relevance, query-term coverage,
    document diversity) under a token budget, with an explicit per-document cap.
    """

    budget_words: int = 900
    max_passages: int = 10
    # Relevance floor, as a fraction of the best candidate's score. Diversity
    # and coverage must not be able to spend the budget on near-irrelevant
    # passages: an off-topic passage in the context is not merely wasted tokens,
    # it is something the reader can read an answer out of. Observed directly --
    # before this floor existed, a question about James Watt admitted a passage
    # about a Thai architect and the reader answered from it.
    min_rel_ratio: float = 0.35
    # Apply the relevance floor to the TAIL only, exempting anything already
    # within the top `max_passages` by relevance.
    #
    # The floor exists to stop the diversity and coverage terms dragging up
    # deep-tail passages; it was never meant to reject head candidates. With
    # the floor applied to everything, the 20-paragraph `pool` setting suffers:
    # there the budget is not the binding constraint (10 passages of ~80 words
    # fit inside 900), so rejecting a mid-ranked candidate forfeits recall for
    # no saving. Exempting the head keeps the tail protection and removes the
    # cost. Measured by scripts/ab_assembly.py in both settings.
    #
    # Enabled after scripts/ab_assembly.py measured it in both settings
    # (n=150, retrieval-only). AllDocs vs the hybrid+rerank baseline:
    #
    #            floor applied to all   floor applied to tail only
    #   unified        58.0 (+8.7)            60.7 (+11.4)
    #   pool           63.3 (-2.0)            72.0 ( +6.7)
    #
    # In the 20-paragraph `pool` setting the full floor actually LOST to the
    # baseline: with 20 clean candidates the budget is never contested, so
    # rejecting a head candidate forfeits recall for no saving. Tail-only wins
    # in both settings and lands at near-exact budget parity (739 vs 718 words
    # unified, 716 vs 721 pool), which also makes the comparison cleaner.
    # `abl_floor_all` keeps the other end of the frontier on the record.
    floor_exempt_top_k: bool = True
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
    # "concat"       read the whole assembled context as one windowed sequence,
    #                so the reader can COMPOSE hops. Required for multi-hop:
    #                the answer-bearing paragraph often does not mention the
    #                question's own entities, so a reader shown that paragraph
    #                alone cannot connect them. Measured on the oracle
    #                condition (scripts/reader_mode_probe.py, n=60):
    #                per-passage 28.3 EM vs concat 36.7 EM.
    # "per_passage"  score each passage independently. Simpler provenance and
    #                no position bias, but cannot compose. Kept as an ablation.
    mode: str = "concat"
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
    # Weight on the retriever's passage score when choosing *which* passage to
    # answer from. A span-margin comparison across passages is unreliable on
    # its own -- the margin is calibrated within a passage, not between them --
    # so an irrelevant passage with a confident-looking span can beat the right
    # one. Adding a passage-selection term is the standard fix (cf. DPR's
    # reader, which combines a span score with a passage-selection score).
    passage_selection_weight: float = 4.0


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
    # Directory label for raw outputs. Normally equal to `setting`, but the
    # ablation suite uses its own label so it can run on a smaller subset
    # without breaking the paired-comparison validation in report.py (which
    # requires every run in a directory to cover an identical question set).
    # An ablation directory carries its own `ours` and `hybrid_rerank`
    # reference runs on exactly those questions.
    out_label: str = ""
    # k values at which retrieval metrics are reported.
    k_values: tuple[int, ...] = (2, 5, 10, 20)
    # Questions used for hyperparameter selection (scripts/ab_assembly.py ran
    # on questions[:150]). The report can drop them to produce a split that was
    # never used for any tuning decision -- see RunSet.holdout_skip.
    tuning_split: int = 150
    bootstrap_samples: int = 2000
    permutation_samples: int = 10000
    seed: int = 20251105

    @property
    def label(self) -> str:
        return self.out_label or self.setting


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
