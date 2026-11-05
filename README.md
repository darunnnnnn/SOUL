# ctxlayer — A Knowledge Base as a Context Layer

Multi-document retrieval, coverage-aware context assembly, and grounded
synthesis with **claim-level, mechanically verifiable citations** — evaluated on
an open multi-hop benchmark against credible baselines, with a one-command
reproduction.

- **[DESIGN.md](DESIGN.md)** — architecture, the reasoning and measurements
  behind each decision, provenance, tradeoffs, failure modes, next steps.
- **[EVAL.md](EVAL.md)** — benchmark, subset, metrics, baselines, results,
  ablations, failure analysis. *Generated from raw outputs; never hand-edited.*
- **[WALKTHROUGH.md](WALKTHROUGH.md)** — a scripted 5–10 minute tour, including
  a successful multi-document answer and a real failure.

---

## The one-paragraph version

A single-shot retriever can only find documents the *question* describes. For a
multi-hop question the later evidence is **lexically dark** — it shares no
vocabulary with the question — so no reranker can recover it, because reranking
only reorders what recall proposed. `ctxlayer` fixes candidate generation
instead: it indexes **entity mentions**, then retrieves in rounds, letting
evidence found in round *n* generate the query for round *n+1*. It then treats
context construction as a **set optimisation** (relevance × coverage ×
diversity, under a hard per-document cap and token budget) rather than a top-k
slice, and cites **sentences** by stable id with character offsets, verified by
re-reading the source.

---

## Requirements

| | |
|---|---|
| Python | ≥ 3.10 (tested on 3.11) |
| Dependencies | `torch`, `transformers`, `numpy` — that's all |
| Disk | ~1.5 GB (≈700 MB models, ~30 MB benchmark, ~50 MB indexes) |
| Hardware | **CPU is enough.** No GPU, no API key required. |
| Network | once, to fetch the benchmark and three HuggingFace models |

BM25F, vector search, the CLI and config parsing are all implemented here
rather than pulled in (no `rank_bm25`, `faiss`, `sentence-transformers`,
`typer` or `PyYAML`) — fewer pinned dependencies means fewer ways for a
reproduction to fail. See DESIGN.md §3.6 for why each was a deliberate choice
rather than an omission.

Models downloaded on first run (all small, all public):

| Model | Size | Role |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 133 MB | dense retrieval |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90 MB | reranking |
| `deepset/roberta-base-squad2` | 496 MB | the fixed reader |

---

## Setup

```bash
git clone <this-repo> && cd ctxlayer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
pytest -q                                              # 69 tests, ~5 s, no network
```

## Reproduce everything with one command

```bash
python -m ctxlayer.cli reproduce
```

This downloads and **sha256-verifies** the benchmark, selects the subset
deterministically, builds all three indexes, runs all seven systems, runs the
ablation suite, and rebuilds `EVAL.md` from the raw outputs. Expected wall
clock on a 16-core CPU laptop: **~4–6 hours**, dominated by the cross-encoder.
Monetary cost: **$0** — nothing leaves the machine.

To get the headline table faster:

```bash
python -m ctxlayer.cli reproduce --subset 150 --skip-ablations   # ~35 min
```

---

## Step by step

```bash
# 1. Fetch the benchmark (MuSiQue-Ans v1.0 dev) and verify its checksum   ~30 s
python -m ctxlayer.cli download

# 2. Build the knowledge base: deterministic subset selection + normalisation
#    One title = one document; sentence-level citation anchors are minted here.
python -m ctxlayer.cli ingest --subset 500                                # ~10 s

# 3. Build indexes: BM25F, dense (cached to disk), entity mention index
python -m ctxlayer.cli index                        # ~8 min first time, then ~25 s

# 4. Ask a question — prints the assembled context and verified citations
python -m ctxlayer.cli ask "Who is the spouse of the Green performer?"

# 5. Run the systems (writes per-question raw outputs, aggregates nothing)
python -m ctxlayer.cli eval   --setting unified                     # ~2.5–3 h
python -m ctxlayer.cli eval   --setting unified --limit 100         # ~25 min

# 6. Ablations (own directory + own reference runs, so comparisons stay paired)
python -m ctxlayer.cli ablate --setting unified --limit 200
python -m ctxlayer.cli ablate --list                     # show the suite

# 7. Rebuild every published number from the raw outputs — offline, seconds
python -m ctxlayer.cli report --setting unified --write-md
```

### Runtime and cost

| Step | Time (16-core CPU) | Cost |
|---|---|---|
| download + ingest | ~40 s | $0 |
| index (first build) | ~8 min (dense encoding; cached afterwards) | $0 |
| `eval`, 500 q × 7 systems | ~2.5–3 h | $0 |
| `ablate`, 200 q × 20 ablations | ~2–3 h (cross-encoder cache makes this far cheaper than 20 full runs) | $0 |
| `report` | ~10 s, offline | $0 |

The persistent cross-encoder score cache (`data/cache/rerank_scores.sqlite`)
means re-runs and ablations reuse scored `(query, passage)` pairs. It is keyed on
model + truncation length + query + passage text, so changing the reranker
misses the cache rather than silently reusing another model's scores.

---

## CLI reference

| Command | What it does |
|---|---|
| `download` | fetch the benchmark, verify sha256 |
| `ingest` | deterministic subset selection → `data/processed/` + manifest |
| `index` | build BM25F / dense / entity indexes → `data/index/` |
| `ask` | answer one question, printing context, bridges and verified citations |
| `eval` | run systems → per-question raw outputs |
| `ablate` | run the ablation suite |
| `report` | rebuild all tables from raw outputs; `--write-md` regenerates EVAL.md |
| `diagnose` | reproduce the design measurements (below) |
| `reproduce` | all of the above, one command |

### Diagnostics — check our claims rather than trusting them

```bash
# The measurement that chose the bridge mechanism: a title-keyed document-link
# graph reaches 30.4% of gold evidence chains; a surface-form mention index
# reaches 90.3%. Prints both, plus the root cause.
python -m ctxlayer.cli diagnose reachability

# Hard Requirement #2: assert every evaluated question needs >= 2 distinct docs
python -m ctxlayer.cli diagnose corpus

# Hard Requirement #1: exhaustively re-resolve all ~24k citation anchors
# against the live source text
python -m ctxlayer.cli diagnose anchors

# Honest failure analysis over the raw outputs, with examples
python -m ctxlayer.cli diagnose failures
```

---

## Using it as a library

```python
from ctxlayer.config import PRESETS, PathsConfig
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.ingest import load_processed
from ctxlayer.pipeline import ContextLayer
from ctxlayer.synth.cite import verify_citations
from ctxlayer.synth.reader import ExtractiveReader

paths = PathsConfig()
cfg = PRESETS["ours"]
kb = KnowledgeBase.load(load_processed(paths), cfg, paths)
kb.enable_score_cache(paths, cfg)

layer = ContextLayer(kb, cfg)
ctx = layer.build_context("Who is the spouse of the Green performer?")

print(f"{len(ctx.passages)} passages from {len(ctx.doc_ids)} documents, "
      f"{ctx.n_words} words")
print("bridge entities followed:", ctx.trace.bridge_entities)

answer = ExtractiveReader(cfg.reader).read(ctx.trace.queries_issued[0], ctx)
verify_citations(kb, answer)

for claim in answer.claims:
    for aid in claim.anchor_ids:
        a = kb.resolve_anchor(aid)
        assert kb.verify_anchor(aid)              # byte-exact against source
        print(f'"{claim.text}"  <-  {a.locator()}\n    "{a.text}"')
```

Swapping the generator for an LLM is one field (plus an API key in the
environment — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GOOGLE_API_KEY`):

```python
cfg = PRESETS["ours"].replace(**{"reader.kind": "llm",
                                 "reader.llm_model": "claude-haiku-5-5"})
```

The LLM path emits claim-level citations and is verified by the **same**
`verify_citations`, so an LLM citation is held to the identical mechanical
standard. It is deliberately *not* used for the headline numbers — see
DESIGN.md §3.5.

---

## Systems compared

| Name | What it is |
|---|---|
| `closed_book` | no retrieval — measures what the reader answers from parameters alone |
| `bm25` | BM25F top-k (title + body fields) |
| `dense` | dense bi-encoder top-k |
| `hybrid` | BM25F + dense, weighted RRF |
| `hybrid_rerank` | hybrid + cross-encoder — **the primary comparison point**: the strongest single-shot pipeline in common use |
| `ours` | iterative entity-bridged retrieval + coverage-aware assembly |
| `oracle` | gold paragraphs only — a *ceiling*, separating retrieval error from reading error |

Every one of these is the **same code** with config fields switched off; there
is no separate "our system" path. They share one knowledge base, one reader
instance and one token budget, all enforced in the runner. `EVAL.md` reports
context size per system so budget parity is auditable rather than asserted.

---

## Repository layout

```
src/ctxlayer/
  schema.py              data model; citation anchors; sentence splitting
  config.py              every system is fully determined by a dataclass
  pipeline.py            ContextLayer: question -> assembled cited context
  cli.py                 argparse CLI
  ingest/musique.py      ingestion + deterministic subset selection
  index/
    bm25.py              BM25F over (title, body), NumPy CSR
    dense.py             bge-small, CLS pooling, exact search, disk cache
    entity_index.py      *** surface-form mention index: the bridge ***
    fusion.py            weighted Reciprocal Rank Fusion
    rerank.py            cross-encoder, cache-aware
    score_cache.py       persistent (query, passage) -> score store
    store.py             KnowledgeBase facade
  retrieve/
    decompose.py         rule-based query planning (anchors / relation / focus)
    hybrid.py            single-shot retrieval — all baselines live here
    iterative.py         *** iterative entity-bridged retrieval ***
    assemble.py          *** coverage-aware multi-document assembly ***
  synth/
    reader.py            fixed local extractive reader (windowed concat)
    llm.py               pluggable LLM reader (optional)
    cite.py              L1/L2/L3 citation verification
  eval/
    metrics.py  stats.py  runner.py  report.py  ablations.py  diagnose.py  writeup.py
tests/                   69 tests: metrics, provenance, retrieval, assembly, config
eval/raw_outputs/        per-question records (the evidence)
eval/results/            rebuilt aggregates (derived)
```

---

## How the hard requirements are met

| Requirement | Where |
|---|---|
| **1. Provenance** — every claim traces to a specific source location | Citations are sentence `Anchor`s (`<doc_id>::p<ord>::s<idx>`) with character offsets, minted at ingestion and verified by re-reading the source. `diagnose anchors` checks all ~24k exhaustively. `cite.py` enforces L1 resolvable → L2 supported → L3 repair. |
| **2. Genuinely multi-document** | The 54 dev questions whose gold evidence sits in a single document are **excluded** (`diagnose corpus` asserts it). Coverage is measured over **distinct documents** (`ctx_all_docs`), so several paragraphs of one article can never satisfy it — and `tests/test_metrics.py` has a test for exactly that. Assembly enforces a hard per-document cap. |
| **3. Open benchmark, measured improvement** | MuSiQue-Ans v1.0 dev (sha256-pinned), chosen and justified in EVAL.md §1. Baselines up to `hybrid + cross-encoder`. Paired permutation tests, bootstrap CIs, Holm correction, reported at equal token budget. |
| **4. Reproducible evaluation** | `python -m ctxlayer.cli reproduce`. Subset selection is a seeded pure function; the runner only writes raw records and `report` recomputes every number from them offline, refusing to aggregate runs with mismatched question sets, corpus fingerprints or readers. |

## Known limitations

Stated plainly, with detail in DESIGN.md §6 and EVAL.md §9:

- **The reader extracts; it does not reason across passages.** `EM`/`F1` here
  measure *"did the context layer put the answer-bearing paragraph and its
  supporting chain in front of the reader?"* — the `oracle` row bounds reading
  error separately so the two are not conflated.
- **A 500-question subset**, seeded and stratified; confidence intervals
  quantify the resulting uncertainty.
- **L2 citation support is lexical + a hard numeric gate, not entailment** —
  exact for extractive claims, approximate for free-form paraphrase.
- **English Wikipedia prose**: the proper-noun detector relies on
  capitalisation.
