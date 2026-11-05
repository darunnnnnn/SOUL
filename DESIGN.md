# DESIGN.md — A Knowledge Base as a Context Layer

## 1. Interpretation of the assignment

The brief leaves the scope, the source material and the architecture to me, and
says to explain the interpretation. Mine rests on one claim:

> **The hard problem in a multi-document context layer is not ranking. It is
> candidate generation.**

A single-shot retriever — sparse, dense, hybrid or reranked — can only surface
documents that the *question* describes. For a question whose evidence spans
several sources, the later evidence is usually *lexically dark*: it shares no
vocabulary with the question at all. Concretely, from the benchmark:

```
Q: "Who is the spouse of the Green performer?"
   hop 1  Green (Steve Hillage album)   -> performer = Steve Hillage
   hop 2  Miquette Giraudy              -> "...her partner Steve Hillage"
```

The second gold document is titled *Miquette Giraudy*. The question contains
neither "Giraudy" nor "Hillage". **No reranker can fix this**, because reranking
only reorders what recall already proposed, and recall never proposed it. The
ceiling is in candidate generation, so that is where the engineering went.

Three consequences shaped everything else:

1. **Retrieval must be iterative**: evidence found in round *n* has to generate
   the query for round *n+1*. That requires an index of what entities a passage
   *mentions*, not just what it is *about*.
2. **A ranking is not a context.** A prompt needs a *set*, and a top-k slice
   optimises each slot independently — so it cheerfully spends the whole budget
   on near-duplicate passages from the question's most prominent entity, and
   covers one document when the question needs two. Assembly is a separate
   problem from ranking and gets its own component.
3. **Provenance has to be mechanical, not promised.** A citation is only
   evidence if a machine can re-resolve it to the exact source bytes. So
   citations address *sentences* by stable id with character offsets, and are
   verified by re-reading the source.

### Scope and source material

The knowledge base is **21k paragraph-level passages over ~17.6k distinct
English Wikipedia articles**, ingested from MuSiQue-Answerable v1.0's own source
paragraphs (the evaluated subset uses 6,793 passages / 5,848 documents).

I chose the benchmark's own corpus deliberately. Ingesting some unrelated
document collection would have made for a nicer demo and a worthless
evaluation: there would be no gold supporting labels, so every claim about
multi-document retrieval would have been an assertion. Using the benchmark's
paragraphs means **support recall, document coverage and citation precision are
all measured against gold labels** rather than eyeballed.

On file format — the brief does not prescribe one, and I treat that as a
question about the *unit of provenance* rather than about parsing. I ingest
JSONL and normalise to a three-level model (document → passage → sentence
anchor). The paragraph is the retrieval unit because it is the natural
semantic unit of the source and because fixed-size token windows destroy
citation precision: a 512-token chunk spanning three topics can be "relevant"
while no sentence in it supports the claim. Ingestion of other formats is a
matter of writing another normaliser to the same model; the context layer above
it is format-agnostic.

### Two retrieval settings, from one index

| Setting | Candidate space | Purpose |
|---|---|---|
| `unified` **(headline)** | every passage in the subset (~6.8k) | A real knowledge base. Distractors from every other question are live, so retrieval is genuinely open-domain. |
| `pool` | each question's own 20-paragraph pool | The standard MuSiQue distractor setting, so our numbers stay legible against published work. |

The setting is a per-query argument, not a separate build, so both numbers
provably come from the same index. Published MuSiQue work almost always reports
the 20-paragraph setting; I lead with the harder one because a per-question pool
of 20 is not a knowledge base.

---

## 2. Architecture and data flow

```
                        ┌─────────────────────────────────────────┐
  musique_ans_v1.0      │ INGEST  (sha256-pinned, deterministic)  │
  _dev.jsonl       ───► │  normalise → document / passage /       │
  (sha256 verified)     │  sentence-anchor; content-addressed ids │
                        └───────────────────┬─────────────────────┘
                                            ▼
      ┌──────────────────────────── KNOWLEDGE BASE ────────────────────────────┐
      │  BM25F (title+body)      dense bge-small (CLS, exact)   entity         │
      │  own impl, NumPy CSR     21k x 384, cached .npy         mention index  │
      │  0.15 ms/query                                          76k keys       │
      └───────────────────────────────────┬───────────────────────────────────┘
                                          ▼
   question ──► QUERY PLAN ──────► ITERATIVE ENTITY-BRIDGED RETRIEVAL
               anchors / relation      round 0: hybrid RRF + cross-encoder
               terms / answer focus    round r: frontier → bridge entities
               (rule-based, no LLM)             (idf x chain score, top-b)
                                              → "entity + relation" query
                                              → mention-index candidates
                                              → rerank vs BRIDGE query
                                              → chain score, adaptive stop
                                          │
                                          ▼
                              CONTEXT ASSEMBLY  (set optimisation)
                              relevance x coverage x MMR diversity
                              hard per-document cap + word budget
                                          │
                                          ▼
                              SYNTHESIS (fixed local reader)
                              windowed extractive QA over the context
                              span → passage → sentence anchor
                                          │
                                          ▼
                              CITATION VERIFIER
                              L1 resolvable (byte-exact) → L2 supported
                              (lexical + hard numeric gate) → L3 repair
                                          │
                                          ▼
                              Answer{ answer, claims[ text, anchor_ids ] }
```

Every stage is determined entirely by a `SystemConfig` dataclass. There is no
separate "our system" code path: the baselines are the same code with fields
switched off, which is what makes the comparison fair rather than merely
claimed.

---

## 3. The decisions that mattered, and the measurements behind them

### 3.1 The bridge mechanism: measured, then rebuilt

My first design was the intuitive one — a **document-link graph keyed on article
titles**, approximating the Wikipedia hyperlink structure that multi-hop
benchmarks are built from. Before building retrieval on top of it, I measured
the only thing that matters for a candidate generator: *starting from the first
gold passage, is the rest of the gold evidence chain reachable at all?* A
retriever cannot rank what its candidate generator can never propose, so this
upper-bounds any iterative method.

| Bridge mechanism | Gold-chain reachability |
|---|---|
| Title-keyed document link graph | **34.3 %** |
| Surface-form **mention** index | **89.5 %** |

The title graph fails *structurally*, and the root cause is measurable: only
**21.9 %** of MuSiQue bridge answers are themselves article titles in the
corpus. The hop-*n+1* bridge entity is normally a *mention inside* the next gold
paragraph — *Steve Hillage* appears in Giraudy's article, but there is no
*Steve Hillage* node to traverse to. So I threw the graph away and indexed
mentions instead.

Reproduce both numbers: `python -m ctxlayer.cli diagnose reachability`.

Two follow-on choices came from the same reasoning:

- **Phrase containment for candidate generation, not BM25 on the entity.** A
  two-token entity used as a BM25 query returns a long tail of passages matching
  one of its tokens. Exact span containment returns passages that genuinely
  discuss the entity. Those candidates are then *scored* by the full fused
  retriever and cross-encoder — the index decides what is considered, not what
  wins.
- **Proper nouns from a corpus statistic, not an NER model.** Capitalisation
  alone over-generates at sentence start ("Written in spring 1977" → "Written").
  A single-token candidate is accepted only if it appears capitalised in ≥60 % of
  its corpus occurrences. "Zhejiang" passes, "Written" and "During" do not. This
  avoids a ~400 MB dependency and, more importantly, avoids coupling
  anchor-time behaviour to a model version that could silently change
  previously-committed citations.

### 3.2 Bridge queries carry the relation, not just the entity

Searching the bare bridge entity retrieves the article *about* it. Searching
`entity + relation_terms` retrieves the paragraph that *states the relation* —
which is where the next hop's answer lives. The query plan therefore separates
what the question *gives* you (anchor entities) from what it *asks* (relation
terms), and only the latter is carried into later hops:

```
Q "Who is the spouse of the Green performer?"
  anchors  : [Green]           # where we started; never re-used as a bridge
  relation : [spouse, performer]
  bridge   : "Steve Hillage spouse performer"
```

The `no_query_rewrite` ablation measures what dropping this costs.

### 3.3 Cross-query score comparability

A hop-2 passage is reranked against its *bridge* query, not the question, so its
cross-encoder logit is not comparable to a hop-0 passage's. Treating them as
comparable is a subtle, invisible bug. Scores are therefore min-max normalised
**within the round that produced them** and combined into a path score:

```
chain(p) = hop_decay^r · [ w_rel·rel_norm(p) + w_parent·chain(parent) + w_ent·idf(entity) ]
```

A passage is good if it is relevant *and* was reached through good evidence —
which is the multi-hop notion of relevance. The hop decay stops deep
speculative chains from outranking direct evidence on score alone.

### 3.4 Assembly is a set problem

Observed directly, before the per-document cap existed:

```
[1] Zhejiang          <- hop 1, relevant
[2] Zhejiang          <- another paragraph of the same article
[3] Zhejiang          <- and another
[4] Cork (city)       <- distractor sharing vocabulary
...
[k] Sanjiang Church   <- hop 2 gold, below the cut
```

Every slot is individually defensible; the set is useless. This is exactly the
trap Hard Requirement #2 names. Assembly therefore maximises

```
gain(p) = λ·rel(p) + (1-λ)·[ w_cov·cov(p) + (1-w_cov)·div(p) ]
```

under a word budget, with a **hard** per-document cap and a `min_docs` floor
that forces a document switch. `cov` is marginal coverage of still-uncovered
question terms *and discovered bridge entities* — the bridges matter because by
construction they name the intermediate hops that the question never mentions,
so they are the only available signal for "have I covered hop 2 yet?". `div` is
MMR over the same bi-encoder vectors used for retrieval.

Two defects here were found by reading one question's actual output rather than
by reasoning, and both are worth recording because the fixes are not cosmetic:

- **Coverage credited individual tokens of multi-word bridges.** The bridge
  *University of Glasgow* made any passage containing "university" look like it
  covered a hop, dragging *Beatrice Heuser* and *Heuser Nunatak* into a context
  about James Watt. Fix: weight coverage terms by **idf taken from the BM25
  index we already built** — "glasgow" counts, "university" barely does, and no
  new hyperparameter is introduced.
- **No relevance floor.** Diversity and coverage happily spent leftover budget
  on near-irrelevant passages. An off-topic passage is not merely wasted tokens
  — it is something the reader can read an answer *out of*, and it did (a Thai
  architect, for a question about James Watt). Fix: `min_rel_ratio`, admitting
  weak candidates only when needed to satisfy `min_docs`.

### 3.5 Synthesis: a fixed local reader, deliberately not an LLM

The headline evaluation holds the generator **fixed** — same local model, same
prompt construction, same decoding — and varies only the context layer. So
every answer-level delta is attributable to retrieval and assembly. With an LLM
generator, three things break:

- **Credit attribution.** A strong model answers many 2-hop Wikipedia questions
  from parametric memory, confounding "better context" with "better model". The
  `closed_book` baseline exists precisely to quantify that risk.
- **Determinism.** Seven systems × 500 questions × 20 ablations against a
  sampling API gives numbers that move between runs.
- **Reproducibility.** A reviewer would need the same model version and a funded
  key.

An LLM reader *is* implemented (`src/ctxlayer/synth/llm.py`, Anthropic / OpenAI
/ Gemini) and shares the same `Answer`/`Claim` schema and the same citation
verifier, so an LLM citation is held to the identical mechanical standard. It is
enabled with one env var. It is not used for the headline numbers.

**Reading protocol, chosen by measurement.** I first scored each passage
independently and took the globally best span — attractive because provenance is
exact and there is no "lost in the middle" position bias. On the oracle
condition (gold paragraphs only, so any failure is a *reading* failure) that
protocol capped EM at **22.5**. The reason is structural: per-passage reading
cannot *compose* hops — the paragraph holding the answer often never mentions
the question's own entities. So the default reads the assembled context as one
windowed sequence, keeping an offset→passage map so a span still resolves to a
specific passage and then to a specific sentence. The per-passage mode is
retained as an ablation.

This is the honest limitation of the submission: with a fixed extractive reader,
`EM`/`F1` measure *"did the context layer put the answer-bearing paragraph and
its supporting chain in front of the reader?"* — not *"can the reader chain
facts?"*. That is the quantity the assignment is about, and the `oracle` row
bounds reading error separately so the two are never conflated.

### 3.6 Smaller choices, briefly

- **BM25F, hand-written.** `rank_bm25` is single-field and loops in Python.
  Fielded scoring matters because the bridge entity is usually the *title* of
  the document answering the next hop, and a title term is far more diagnostic
  than the same term in prose — BM25F weights fields *before* saturation, which
  is the statistically correct combination, unlike scoring two indexes and
  adding. NumPy CSR scatter-add gives 0.15 ms/query, which matters when the
  iterative retriever issues ~13 queries per question.
- **RRF, not score interpolation.** BM25 scores are unbounded and
  corpus-dependent; cosines sit in a narrow band. Any fixed linear combination
  needs per-corpus calibration. The `fusion_linear` ablation shows the cost.
- **Exact vector search, not FAISS/HNSW.** At 10⁴ passages a brute-force
  normalised matmul is sub-millisecond and *exact*, so no ANN recall ceiling is
  folded invisibly into the retrieval numbers. ANN is right at 10⁷; here it
  would add hyperparameters and lose recall for speed we do not need.
- **bge-small-en-v1.5 with CLS pooling.** bge-v1.5 is trained with CLS pooling
  and an asymmetric query instruction; mean pooling (the common copy-paste
  error) costs several nDCG points. The encoder is shared identically by our
  system and every retrieval baseline, so its absolute strength biases no
  comparison.
- **Persistent cross-encoder score cache.** The reranker dominates evaluation
  cost, and pairs repeat heavily across systems, ablations and re-runs. A
  content-addressed SQLite cache (keyed on model + max_length + query + passage
  text, so a changed model misses rather than silently reusing) turns the
  ablation suite from *N × one run* into roughly *one run + deltas*. It is a
  pure speed optimisation — scores are deterministic under `inference_mode`.

---

## 4. Provenance

The unit of citation is an **anchor**: one sentence, addressed as
`<doc_id>::p<passage_ord>::s<sent_idx>`, carrying explicit character offsets.

- **Anchors are minted once at ingestion and only ever referenced.** Nothing
  downstream can invent one.
- **Ids are content-addressed**, so a rebuild from a different data ordering or
  subset yields identical ids for identical content — a previously-committed
  citation does not rot.
- **Sentence splitting is dependency-free and deterministic.** A model-based
  splitter would make anchor ids depend on a model version, silently
  invalidating every stored citation.
- **Verification is mechanical.** `KnowledgeBase.verify_anchor` re-reads the
  live passage at the recorded offsets and requires byte-identical text.
  `python -m ctxlayer.cli diagnose anchors` checks all ~24k anchors
  exhaustively, not by sampling.

Citations are then checked at three levels (`src/ctxlayer/synth/cite.py`):

| Level | Check | Nature |
|---|---|---|
| **L1 resolvable** | the anchor exists and its text is byte-identical to the source at its offsets | mechanical, no threshold — a failure here is a fabrication |
| **L2 supported** | the claim's content words are present in the cited sentence; **numbers and dates must match exactly** | lexical + hard numeric gate |
| **L3 repair** | a claim supported by a *different* anchor in the context has its pointer corrected, and the repair is **recorded separately** | fixes "right answer, wrong pointer" without laundering it into precision |

The numeric gate exists because paraphrase-tolerant scoring is most dangerous
exactly where it is least safe: "1987" and "1978" are semantically
near-identical and factually opposite.

---

## 5. Tradeoffs

| Decision | Gained | Paid |
|---|---|---|
| Iterative retrieval | reaches lexically dark hops; the main gain | ~13 queries/question instead of 1; more retrieval compute |
| Mention index for candidates | 89.5 % vs 34.3 % chain reachability | 76k-key index; recall envelope ~18 % of corpus must be reranked down |
| Exact vector search | no hidden ANN recall ceiling | O(N) per query — would need ANN past ~10⁶ passages |
| Fixed local extractive reader | clean credit attribution, determinism, zero cost | low absolute EM; `EM`/`F1` discriminate weakly (see §3.5) |
| Rule-based query planning | gain is attributable to retrieval, not to an LLM | brittle on questions whose relation is implicit |
| Paragraph-level retrieval unit | precise citations, natural semantics | a fact spanning two paragraphs needs both retrieved |
| Hard per-document cap | guarantees multi-document context | can exclude a genuinely 3-passage-deep single source |
| L2 lexical + numeric support | no 1.4 GB NLI dependency; exact for extractive claims | approximate for free-form paraphrase |

---

## 6. Failure modes

**Bridging onto the answer entity.** Once a hop discovers the answer entity, it
becomes a candidate bridge, and expanding it spends budget retrieving passages
*about the answer* rather than evidence for it. Visible in the walkthrough:
*University of Glasgow* became a bridge and pulled in *Beatrice Heuser*. The
relevance floor and idf-weighted coverage contain the damage but do not remove
the cause.

**Relation terms that are not lexical.** "Who is the spouse of X" works because
"spouse" appears in the target text. Comparative or implicit relations ("which
came first", "the larger of the two") give the bridge query no usable signal.

**Entity-poor first hops.** If round 0 retrieves a passage with few proper
nouns, or the gold chain passes through a common entity pruned by
`max_entity_df`, the bridge never forms. This bounds our recall at the ~89.5 %
reachability envelope, not at 100 %.

**Error cascade.** A wrong round-0 passage sends the beam down a wrong chain,
and later rounds confidently retrieve coherent, irrelevant evidence. The
adaptive stop limits the cost but does not detect the error.

**Reading failure with complete evidence.** The largest single bucket in the
failure analysis is questions where the context contains *all* gold documents
and the extractive reader still answers wrongly — a reader limitation, not a
context-layer one, and the reason the `oracle` row is reported.

**Caseless or unsegmented scripts.** The proper-noun detector relies on
capitalisation and does not transfer to them.

---

## 7. Next steps

1. **Break the reader ceiling.** The strongest immediate gain is answer-level,
   not retrieval-level: run the implemented LLM reader as a *secondary* protocol
   alongside the fixed reader, so answer quality rises while credit attribution
   is preserved by reporting both plus `closed_book`.
2. **Hop-aware stopping with an answerability check.** Today stopping is based
   on score gain. A cheap "is the question now answerable from the pool?" test
   would stop 2-hop questions after one round and let genuine 4-hop questions
   spend more, instead of applying one threshold to both.
3. **Suppress answer-entity bridging.** Score candidate bridges by whether they
   plausibly *lie on the path* (e.g. demote entities matching the question's
   answer-type focus) rather than by rarity alone.
4. **NLI-based L2 support**, behind a feature flag with a pinned model revision,
   reporting lexical and NLI verdicts side by side so the stricter check never
   silently changes historical citation numbers.
5. **Scale the index.** Swap exact search for HNSW past ~10⁶ passages and
   measure the recall cost explicitly rather than inheriting it silently; shard
   the mention postings.
6. **Full-split evaluation.** Lift the 500-question subset to all 2,363 eligible
   dev questions to narrow the confidence intervals reported in EVAL.md.
7. **Sentence-level retrieval as a second granularity**, fusing paragraph and
   sentence rankings, to attack the precision loss from 84-word passages whose
   relevant content is one clause.

---

## 8. Where to look in the code

| Concern | File |
|---|---|
| Data model, anchors, sentence splitting | `src/ctxlayer/schema.py` |
| All configuration, baseline presets | `src/ctxlayer/config.py` |
| Ingestion, deterministic subset selection | `src/ctxlayer/ingest/musique.py` |
| BM25F | `src/ctxlayer/index/bm25.py` |
| Dense index | `src/ctxlayer/index/dense.py` |
| **Entity mention index (the bridge)** | `src/ctxlayer/index/entity_index.py` |
| Rank fusion | `src/ctxlayer/index/fusion.py` |
| **Iterative bridged retrieval** | `src/ctxlayer/retrieve/iterative.py` |
| **Coverage-aware assembly** | `src/ctxlayer/retrieve/assemble.py` |
| Query planning | `src/ctxlayer/retrieve/decompose.py` |
| Reader | `src/ctxlayer/synth/reader.py` |
| **Citation verification** | `src/ctxlayer/synth/cite.py` |
| Metrics / significance tests | `src/ctxlayer/eval/metrics.py`, `stats.py` |
| Runner / report / ablations | `src/ctxlayer/eval/` |
