Build me a single-page interactive artifact: an architecture diagram of a
multi-document retrieval system, with every stage inspectable.

This is a reference diagram, not a story. It is the slide I will leave on
screen while talking through the design, and the page a reviewer will open to
see what the system is made of. The companion artifact animates one question
travelling through the system; this one shows the whole machine at rest.

Every number and model name below is from the shipped configuration. Do not
invent components, do not rename them, and do not add stages.

## WHAT THE PAGE IS

A vertical pipeline of seven stages, each a block. Clicking or hovering a
block expands it to show what is inside: the components, their parameters, and
one line on why the stage exists.

Collapsed, the page reads as seven stages in under ten seconds. Expanded, each
stage is specific enough that an engineer could re-implement it. Both states
matter -- do not make the collapsed view useless, and do not hide the detail
behind so many clicks that nobody finds it.

Default: all collapsed, with a single "expand all" control.

## THE STAGES

### 1. INGEST
Source: MuSiQue-Answerable v1.0 dev, sha256-pinned.

    2417 questions in the split
  -   54 dropped: all gold evidence inside ONE document
    2363 eligible
     500 selected, seed 20251105, stratified by hop count

Output: 5,848 documents -> 6,793 passages -> 24,064 anchors.

Key detail to surface: normalisation is whitespace-only. Punctuation, casing
and quoting are left exactly as they are, because citations are verified by
re-resolving character offsets into this exact string. Any cleverer cleaning
would be a transformation a citation has to survive.

Three levels, and the diagram should show them nested:
    document -- one Wikipedia article; the unit multi-document evidence is counted over
    passage  -- one paragraph; the unit retrieval scores and ranks
    anchor   -- one sentence, with byte offsets; the unit a citation points at

### 2. INDEX  (built once, three indexes side by side)
Show these three as parallel, not sequential.

    BM25F (fielded lexical)
      k1 = 0.9, b_title = 0.4, b_body = 0.75
      title weight 2.5 vs body 1.0
      Why: Wikipedia paragraphs are anaphoric ("The album was released...")
      and often omit their own subject, so the title carries a large share of
      the retrievable signal.

    Dense
      BAAI/bge-small-en-v1.5, 384 dimensions, L2-normalised
      max_length 384
      query prefix: "Represent this sentence for searching relevant passages:"

    Mention index (entity -> passages)
      Surface forms: every capitalised span and noun phrase, not just exact
      document titles.
      Why: see the measurement panel below. This is the component the whole
      method depends on.

### 3. QUESTION ANALYSIS
Input: the raw question. Output: anchor entities, relation terms, focus type.
Small stage; do not give it more space than it deserves.

### 4. SINGLE-SHOT RETRIEVAL
This stage is also the complete strongest baseline, which is worth labelling:
the baseline is not a weakened version of the system, it is this stage run
alone.

    BM25F  ->|
             |-> RRF fusion (k = 60, 100 candidates per signal)
    Dense  ->|
                   -> cross-encoder rerank
                      cross-encoder/ms-marco-MiniLM-L-6-v2
                      top 60 candidates in, max_length 320

### 5. ITERATIVE RETRIEVAL   <-- the contribution
Draw this as a loop, not a line. It is the only stage that cycles, and it
should be visually obvious that it is the only one.

    READ     take the top 5 frontier passages
    EXTRACT  pull entity candidates from their text
    GATE     reachability: require min_entity_df >= 2, or that the entity
             resolves to a document title. An entity appearing in exactly one
             passage can only lead back to the passage it came from.
    RANK     score bridge candidates:
               w_rel 0.6 (relevance) + w_parent 0.3 (parent chain) + w_entity 0.1
               hop_decay 0.92 per hop, min_entity_idf 0.35
    REWRITE  build a new query from the entity plus the question's relation terms
    SEARCH   re-enter stage 4 with that query (per_round_k = 20)
    STOP?    three independent conditions

    budget: max_rounds 3, beam_entities 4, candidates_per_entity 30

Stopping conditions, shown as three separate gates:
    gain_threshold 0.55     -- halt when another round would add little.
                               Fires on 87.8% of questions; mean rounds 1.73
                               rather than the full budget of 3.
    min_frontier_rel 0.25   -- a round whose results are too weak spawns no
                               bridges.
    max_barren_rounds 2     -- halt after two consecutive rounds that add
                               nothing.

### 6. CONTEXT ASSEMBLY
Not top-k truncation. A joint objective under a budget.

    budget_words 900, max_passages 10
    maximise: relevance + query-term coverage (weight 0.35) + diversity (MMR lambda 0.65)
    constraints: max_per_doc 2, min_docs 2
    relevance floor: min_rel_ratio 0.35, with the top-k exempt

Why the per-document cap: without it a budget fills with ten passages from one
document, which is precisely the failure mode in a multi-hop setting. Why the
floor exempts the head: the floor exists to stop diversity and coverage
dragging up deep-tail passages, not to reject strong candidates.

### 7. READ, CITE, VERIFY
Two interchangeable readers over the same context -- show them as a swap, as
this is what isolates retrieval quality from reader quality:

    extractive  deepset/roberta-base-squad2 (local, no API)
    LLM         gpt-4.1-mini-2025-04-14, temperature 0, exact pinned version

Then, for each claim in the answer:
    attach an anchor:  doc_id::p0::s1
    re-resolve its byte offsets against the source corpus
    compare byte-for-byte; reject on any mismatch

Result to show: 24,064 anchors re-resolve exactly, 0 mismatches. A fabricated
anchor is rejected.

## ONE MEASUREMENT PANEL

Beside stage 2, or directly below it, two bars:

    title link graph   30.4%   [REJECTED]
    mention index      90.3%

Label: "reachable gold evidence, 831 links, measured both ways".

One line, no more: the title link graph was built first and rejected on this
measurement. The mention index is in the system because of it.

## ONE CONFIGURATION NOTE, SOMEWHERE SMALL

Every evaluated system -- all five baselines, every ablation, and the full
system -- runs through this same pipeline. Behaviour is determined entirely by
a config object; there is no separate code path for "our system". The
baselines are this diagram with stages switched off:

    bm25           stage 4, lexical only
    dense          stage 4, dense only
    hybrid         stage 4, fusion, no rerank
    hybrid_rerank  stages 4 complete  (the strongest baseline)
    ours           stages 4 + 5 + 6
    oracle         gold documents injected directly (a ceiling, not a system)

Show this as a small matrix of which stages each system uses. It is the reason
the comparison is fair, and it is the kind of thing a reviewer looks for.

## DESIGN DIRECTION

- Restrained and technical. A reference diagram, not a product launch. No
  gradient hero, no emoji, no marketing language.
- The loop at stage 5 must be the visual focus. Everything else is a straight
  pipeline; one stage cycles, and that asymmetry carries the argument.
- Parameters are monospace. Prose is not.
- Colour sparingly and consistently: one accent for the contributed stage,
  one for rejected/failed, grey for everything structural. Never decorative.
- Works on a phone and correct in both light and dark mode.
- Plain descriptive title. No author name.
