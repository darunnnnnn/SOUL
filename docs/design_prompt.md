Build me a three-tab interactive artifact. Tabs: Mechanism, Results,
Architecture. Mechanism opens by default.

THIS PAGE WILL BE SCREEN-RECORDED AND NARRATED LIVE. That governs everything:
it is a visual aid someone talks over at 1080p in a video player, not a page
read at the reader's own pace.

- The Mechanism canvas must FILL the viewport at 1920x1080. If the content
  occupies less than ~80% of the width and ~60% of the height, it is wrong.
- Nothing smaller than 14px. The caption under the canvas is a subtitle:
  24-28px.
- Video compression destroys thin lines and low-contrast grey. Strokes 2px
  minimum, high contrast.
- Step control reachable without scrolling. Autoplay exists but is OFF by
  default: a presenter advances each frame by hand.

## TAB 1 — MECHANISM

### THE ONE IDEA THIS TAB MUST TEACH

A normal RAG system searches ONCE, using only the words in the question. If
the document holding the answer does not contain those words, that system can
never reach it — no reranker and no larger top-k fixes this, because it is
searching for a string it does not have.

My system runs a LOOP. Search, read what came back, take a new word out of
it, search again with that word, add what comes back to the evidence, read
that, take another new word, search again. Each round discovers the term that
makes the next round possible. It stops when a round stops adding anything.

THE REPETITION IS THE POINT. If a viewer finishes the animation thinking "it
does one extra search", the page has failed. They must see the same operation
happen TWICE, recognisably the same both times, with the evidence pile growing
each time.

### THE EXAMPLE — a three-document chain, deliberately

    Question: "In which city is the university attended by the founder of
               Microsoft located?"
    Answer:   "Cambridge"

Three documents are needed, and they must be found in order, because each one
contains the search term for the next:

    Document 1 - "Microsoft"
      "Microsoft Corporation is an American multinational technology company
       headquartered in Redmond, Washington. It was founded by Bill Gates and
       Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for
       the Altair 8800."
      -> gives the name BILL GATES

    Document 2 - "Bill Gates"
      "William Henry Gates III is an American businessman and philanthropist.
       He enrolled at Harvard University in 1973 and left two years later,
       before completing his degree, to work full time on the software company
       he had founded."
      -> gives HARVARD UNIVERSITY

    Document 3 - "Harvard University"
      "Harvard University is a private Ivy League research university in
       Cambridge, Massachusetts. Established in 1636, it is the oldest
       institution of higher learning in the United States."
      -> gives the answer, CAMBRIDGE

Now check the question against those documents. The question contains: city,
university, attended, founder, Microsoft.

It does NOT contain "Bill Gates". It does NOT contain "Harvard". It does NOT
contain "Cambridge".

So every single word you need in order to find documents 2 and 3 is a word
that only exists INSIDE an earlier document. That is why one search cannot
work, and why the loop has to run twice.

Two absences to make visible on screen, because they are the proof:
  - Document 2 never says "Microsoft" — it says "the software company he had
    founded". So no query containing "Microsoft" reaches it.
  - Document 3 never says "Microsoft" or "Bill Gates" either.

CRITICAL: do not reword these documents to read more smoothly. If "Microsoft"
ever appears in Document 2, the page stops making its point.

### LAYOUT

Two columns, filling the canvas, sharing one question at the top.

The question, large, centred. Highlight "Microsoft" in colour A. Colour A
means, for the whole animation: WORDS THE QUESTION CONTAINS.

LEFT COLUMN — "SINGLE-SHOT RETRIEVAL"
  Short by design. It will be finished early and then sit frozen while the
  right column keeps working. That visual asymmetry is a large part of the
  argument.
    - one query box
    - a document store: a compact grid of ~45 tiles, ~15 of them titled
      (Microsoft, Bill Gates, Harvard University, The Founder (2016 film),
      Apple Inc., Steve Jobs, Stanford University, Yale University, Amazon,
      Jeff Bezos, Intel, Reed College, Paul Allen, Redmond, Cambridge), the
      rest blank to suggest scale. Inert grey until touched. NO LINES BETWEEN
      TILES — there is no link structure, and its absence must be visible.
    - a counter: queries / docs found out of 3
    - space below for its result

RIGHT COLUMN — "ITERATIVE, ENTITY-BRIDGED"
  Built as a VERTICAL STACK OF ROUNDS. Three round-slots, drawn from the
  start, so the viewer can see there is room for more than one and that two
  of them are empty and waiting. Each round-slot is one row:

      ROUND n   [ query ]  ->  [ document retrieved ]  ->  [ new word found ]

  Round 1's query is pre-filled with the question. Rounds 2 and 3 have EMPTY
  query boxes, dashed, visibly awaiting content.

  Plus the same document store grid, the same counter, and — important —

  AN EVIDENCE PANEL that persists and ACCUMULATES. After round 1 it holds one
  document, after round 2 two, after round 3 three. It never empties. This
  panel is how "combine what we found and search again" becomes visible; it
  is the memory the loop is building, and it should grow in an obvious,
  physical way.

Below both columns: one large caption line that changes per frame, then
prev / step dots / next / replay / autoplay.

### COLOUR — fixed, and never decorative

    A  words the question contains            (the question's highlight)
    B  words discovered inside a document     (every bridge term)
    correct / wrong / neutral / rejected

Colour B is the whole idea. Every time a word moves from a document into a
query, it is in colour B. A viewer who notices only "things in colour B came
out of documents, not out of the question" has understood the system.

### THE FRAMES — twelve

One idea per frame. Roughly 15 seconds of narration each.

FRAME 1 — the board, before anything moves
  Question in, "Microsoft" highlighted in A. Both stores grey. Left has one
  query box; right has three round-slots, round 1 filled with the question,
  rounds 2 and 3 empty and dashed. Evidence panel empty. Counters 0, 0/3.
  Caption: "Three documents needed. The question names one."

FRAME 2 — both run round 1, identically
  A beam in colour A from the question into both stores. The SAME tiles light
  in both columns and rise as cards:
      "Microsoft"                 correct
      "The Founder (2016 film)"   wrong
      "Microsoft Research"        neutral
  Both counters -> 1 query, 1/3 docs. Right column's evidence panel gains its
  first document.
  The two columns are identical here, and that must be obvious: nothing so
  far is down to a better search engine.
  Caption: "Same search. Same result."

FRAME 3 — why the wrong document came back
  Zoom "The Founder (2016 film)" large enough to read. "Founder" inside it
  lights in colour A, with a line to the matching word in the question. Show
  enough text for the absurdity to land: a 2016 film about Ray Kroc and a
  hamburger franchise.
  Caption: "Matched on 'founder'. A hamburger franchise."

FRAME 4 — the left column is finished
  It dims and freezes with a terminal marker. Counter holds at 1 query,
  1 of 3 documents, no answer.
  The right column stays bright with two empty round-slots still visible.
  Caption: "One search, and it is over. Two documents still missing."

FRAME 5 — ROUND 1, reading
  On the right, the "Microsoft" card expands to full text. Candidate entities
  underline one at a time in colour B: "Redmond", "Bill Gates", "Paul Allen",
  "Altair 8800".
  Caption: "Reading gives words the question never had."

FRAME 6 — BRIDGE 1  (establish the motion)
  "Bill Gates" LIFTS OUT of the document text, travels along a visible path,
  and lands in ROUND 2's empty query box, assembling into:
      bill gates university attend
  Grey the question out for a beat: this query could not have come from it.
  Counter -> 2 queries.
  Caption: "A query built from the document, not the question."

FRAME 7 — ROUND 2 retrieves, and the proof
  A beam in colour B from round 2's query into the store. The "Bill Gates"
  tile lights and rises, with "Harvard University" highlighted in its text.
  Evidence panel grows to TWO documents. Counter -> 2/3.
  Then show the absence: highlight "the software company he had founded" and
  mark that "Microsoft" does not occur in this document — a struck-through
  "Microsoft" chip, or the question's highlighted word visibly failing to
  match here. Pick one and make it unambiguous.
  Draw a persistent edge from the Microsoft card to this card.
  Caption: "This document never says 'Microsoft'."

FRAME 8 — ROUND 2, reading again
  The Bill Gates card's entities underline in colour B: "Harvard University",
  "Washington". Deliberately the same visual treatment as frame 5.
  Caption: "Read again. New words again."

FRAME 9 — BRIDGE 2  (THE MOST IMPORTANT FRAME ON THE PAGE)
  "Harvard University" lifts out and lands in ROUND 3's empty query box:
      harvard university city located
  This motion must be VISUALLY IDENTICAL to frame 6 — same path style, same
  colour, same timing. The viewer has to recognise it as the same operation
  happening again. If frames 6 and 9 look different, the page reads as two
  one-off tricks instead of a loop, and the whole thing fails.
  Consider showing the two completed bridges stacked together here, so the
  repetition is visible in one glance.
  Counter -> 3 queries.
  Caption: "The same move, again."

FRAME 10 — ROUND 3 retrieves the answer
  Beam in colour B. The "Harvard University" tile lights and rises, with
  "Cambridge, Massachusetts" highlighted. Evidence panel grows to THREE
  documents. Counter -> 3/3. A third edge joins the chain, so the three cards
  now form a visible line: Microsoft -> Bill Gates -> Harvard University.
  Caption: "Three rounds. Three documents. A chain."

FRAME 11 — stopping
  A gauge fills, crosses a marked threshold, and the loop halts itself. Show
  a measurement being taken and a decision following; do not explain it in
  text. Round-slots are now all full, so there is nowhere left to go anyway.
  Caption: "Stops when another round would add nothing."

FRAME 12 — the answer, and the two columns side by side
  Right column answers "Cambridge", marked correct, with citation chips in
  exactly this format:
      25e081610f91::p0::s1      d8c0128ba827::p0::s1      f970f3994adc::p0::s0
  Hovering a chip draws a line back to the highlighted sentence in its card.
  Make this interaction work — it is the proof that citations are locations
  and not decoration.
  Final comparison, held:
      LEFT   1 query  | 1 of 3 docs | no answer
      RIGHT  3 queries| 3 of 3 docs | "Cambridge"  correct
  Caption: none. The frame speaks.

### One panel below the fold

Two measured bars and nothing else:

    title link graph   30.4%   [REJECTED]
    mention index      90.3%

Label: "reachable gold evidence, 831 links, measured both ways".
No paragraph. The link graph was built first and killed by that number.

## TAB 2 — RESULTS

Charts, not prose. Exact numbers as labels on the marks.

One line of context: the Microsoft chain is an illustration; these numbers are
from MuSiQue-Answerable v1.0 dev, sha256-pinned, 500 questions, of which 54
were dropped because all their gold evidence sits inside one document.

Lead with the 350 HELD-OUT questions; the first 150 were development data.
All-500 numbers behind a toggle, never the headline.

Metric meanings in tooltips only:
    AllDocs - 1 only if EVERY required document is in context (no partial credit)
    DocCov  - fraction of required documents in context
    SupAll  - 1 only if every required passage is in context
    EM / F1 - answer exact match and token F1

### Retrieval, 350 held-out questions

| System          | AllDocs | DocCov | SupAll |
|-----------------|---------|--------|--------|
| closed_book     |   0.0   |   0.0  |   0.0  |
| bm25            |  13.7   |  50.6  |   8.6  |
| dense           |  17.1   |  55.2  |  14.6  |
| hybrid          |  20.0   |  57.7  |  16.3  |
| hybrid_rerank   |  24.9   |  62.6  |  21.4  |
| SOUL (ours)     |  38.9   |  70.3  |  34.0  |
| oracle          | 100.0   | 100.0  | 100.0  |

SOUL vs hybrid_rerank on AllDocs: +14.0 points, 95% CI [+9.4, +18.9],
Holm-adjusted p = 0.0012, Cohen's d = 0.31.

oracle is gold documents handed in directly — draw it as a reference line,
not a competing bar.

### Answers, 350 held-out questions, SAME retrieval and SAME LLM reader

Retrieval is identical between these rows; only the reader differs. Make that
visually explicit — the rows should look like they share one context.

| System            |  EM  |  F1  |
|-------------------|------|------|
| closed_book_llm   |  0.0 |  0.0 |
| hybrid_rerank_llm | 23.1 | 30.7 |
| SOUL (ours_llm)   | 34.0 | 41.4 |
| oracle_llm        | 65.7 | 74.4 |

EM +10.9, 95% CI [+6.9, +15.4], Holm-adjusted p = 0.0012.
F1 +10.7, 95% CI [+6.6, +15.1], Holm-adjusted p = 0.0012.

closed_book_llm scores 0.0 and abstains on every question — the leakage
control. Draw it as a zero bar; it earns its place by being empty.

Small print: paired permutation tests, 10,000 resamples, Holm-Bonferroni
corrected across the metric family.

### AllDocs by hop count, all 500 questions

Grouped bars. The win and the failure in one chart.

| System        | 2-hop | 3-hop | 4-hop |
|---------------|-------|-------|-------|
| bm25          | 26.0  | 12.4  |  1.2  |
| hybrid_rerank | 46.9  | 25.5  |  1.2  |
| SOUL          | 74.0  | 37.9  |  4.7  |

### Limitation — a full section, same visual weight as the wins

On 4-hop questions AllDocs is 4.7%, against 74.0% on 2-hop. Partial coverage
is 46.4% — it finds part of the chain and rarely all of it. Four hops means
three consecutive bridges, and per-hop error compounds.

A bridge-diversity fix was tried and made held-out results WORSE: AllDocs
-3.1 points, p = 0.026. It was reverted and the negative result kept.

Draw the -3.1 as a bar going the wrong way. Not a footnote, not a collapsed
accordion, not greyed out.

## TAB 3 — ARCHITECTURE

A vertical pipeline of seven stages, each block expandable to show its real
components and parameters. Collapsed, it reads in ten seconds.

1. INGEST
   sha256-pinned source -> 5,848 documents -> 6,793 passages -> 24,064 anchors
   Whitespace-only normalisation, because citations re-resolve byte offsets
   into this exact string.
   Nest the three levels visually:
     document = what multi-document evidence is counted over
     passage  = what retrieval ranks
     anchor   = what a citation points at

2. INDEX — three indexes, drawn in PARALLEL, not in sequence
     BM25F     k1 0.9, b_title 0.4, b_body 0.75, title weight 2.5 vs body 1.0
               (Wikipedia paragraphs are anaphoric and omit their subject)
     Dense     BAAI/bge-small-en-v1.5, 384-dim, L2-normalised
     Mention   surface forms -> passages; every capitalised span and noun
               phrase, not just exact titles

3. QUESTION ANALYSIS
   anchor entities, relation terms, focus type. A small stage; keep it small.

4. SINGLE-SHOT RETRIEVAL  — label this "= the strongest baseline"
     BM25F + Dense -> RRF fusion (k 60, 100 candidates per signal)
                   -> cross-encoder rerank
                      cross-encoder/ms-marco-MiniLM-L-6-v2, top 60 in

5. ITERATIVE RETRIEVAL — draw as a LOOP. The only stage that cycles, and that
   must be obvious at a glance. This is the stage tab 1 animates.
     READ     top 5 frontier passages
     EXTRACT  entity candidates from their text
     GATE     reachability: min_entity_df >= 2, or resolves to a title
     RANK     w_rel 0.6 + w_parent 0.3 + w_entity 0.1, hop_decay 0.92
     REWRITE  new query from entity + relation terms
     SEARCH   back into stage 4 (per_round_k 20)
     STOP?    three gates:
                gain_threshold 0.55   fires on 87.8% of questions;
                                      mean 1.73 rounds, budget 3
                min_frontier_rel 0.25 a weak round spawns no bridges
                max_barren_rounds 2   halt after two empty rounds

6. CONTEXT ASSEMBLY — not top-k truncation
     budget 900 words, max 10 passages
     maximise relevance + coverage (0.35) + diversity (MMR lambda 0.65)
     constraints: max_per_doc 2, min_docs 2
     Why the cap: without it the budget fills with one document, the exact
     failure mode in a multi-hop setting.

7. READ, CITE, VERIFY — show the two readers as an interchangeable SWAP,
   since that is what isolates retrieval quality from reader quality
     extractive  deepset/roberta-base-squad2 (local)
     LLM         gpt-4.1-mini-2025-04-14, temperature 0, pinned version
   Then per claim: attach anchor doc_id::p0::s1, re-resolve its byte offsets
   against the source, compare byte-for-byte, reject on mismatch.
   24,064 anchors re-resolve exactly, 0 mismatches.

### The matrix — include this, it matters

    bm25           stage 4, lexical only
    dense          stage 4, dense only
    hybrid         stage 4, fusion, no rerank
    hybrid_rerank  stage 4 complete          <- strongest baseline
    SOUL           stages 4 + 5 + 6
    oracle         gold injected directly    <- ceiling, not a system

Every evaluated system runs through the same pipeline; behaviour comes
entirely from a config object, and there is no separate code path for SOUL.
This is what makes the comparison fair.

## GLOBAL DESIGN DIRECTION

- Restrained and technical. A research artifact, not a product launch. No
  gradient hero, no emoji, no marketing superlatives.
- Colour carries meaning and is never decorative.
- Motion is purposeful: things move to show where information came from. No
  fades or slides that mean nothing.
- Respect prefers-reduced-motion: expose the same frames as static steps.
- Must work at 1920x1080 for recording AND on a phone (columns stack there).
  Correct in light and dark mode; dark is what gets recorded, so check it
  first.
- Plain descriptive title. No author name.
