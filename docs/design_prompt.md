Build me a three-tab interactive artifact. Tabs: Mechanism, Results,
Architecture. Mechanism opens by default.

THIS PAGE WILL BE SCREEN-RECORDED FOR A VIDEO and narrated live. That governs
every decision below. It is not a page someone reads at their own pace — it is
a visual aid someone talks over, at 1080p, in a video player, possibly on a
laptop screen. Design for that.

## WHAT RECORDING MEANS IN PRACTICE

- The Mechanism canvas must FILL the viewport at 1920x1080. No large empty
  margins, no small cluster of content floating in a dark field. If the two
  tracks occupy less than about 80% of the width and 60% of the height, it is
  wrong.
- Nothing smaller than 14px. Document tiles that carry a title need readable
  type; the caption line under the canvas is large, 24–28px, because it is
  effectively a subtitle.
- Video compression destroys thin lines and low-contrast greys. Use solid
  strokes of at least 2px and keep contrast high.
- The step control must be reachable without scrolling, and the current step
  must be obvious at a glance.
- No autoplay by default. A presenter advances each frame by hand and talks
  over it. Include an autoplay toggle, but off.

## THE EXAMPLE

    Question: "Which university did the founder of Microsoft attend?"
    Answer:   "Harvard University"

    Document A - "Microsoft"
      "Microsoft Corporation is an American multinational technology company
       headquartered in Redmond, Washington. It was founded by Bill Gates and
       Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for
       the Altair 8800."

    Document B - "Bill Gates"
      "William Henry Gates III is an American businessman and philanthropist.
       He enrolled at Harvard University in 1973 and left two years later,
       before completing his degree, to work full time on the software company
       he had founded."

Everything turns on what each document does and does not say:

  - Document A names the founder and mentions no university.
  - Document B names the university and NEVER SAYS "MICROSOFT" -- it says
    "the software company he had founded".

So the second document cannot be found by searching the question: the
question's distinctive word is "Microsoft", and that word does not occur in
Document B. The only route is to read Document A, take "Bill Gates" out of it,
and search again.

CRITICAL: do not rewrite Document B to read more naturally. If the word
"Microsoft" ever appears in it, the page stops making its point. Frame 7
exists specifically to show that absence.

## TAB 1 — MECHANISM

### Layout

Two tracks side by side, filling the canvas, sharing one question at the top.

    LEFT TRACK  - "SINGLE-SHOT RETRIEVAL"
    RIGHT TRACK - "ITERATIVE, ENTITY-BRIDGED"

Each track contains, top to bottom:

  1. A counter row: "queries 0   docs 0/2"
  2. QUERY SLOTS -- empty rounded boxes. The LEFT track has ONE. The RIGHT
     track has TWO, and the second is visibly empty and waiting. This
     asymmetry must be noticeable in frame 1, before anything moves: the
     narrator points at it and says "nothing in the question can fill that
     box". Give the empty slot a dashed or outlined treatment so it reads as
     a slot awaiting content rather than as a blank panel.
  3. THE DOCUMENT STORE -- a compact grid of ~45 tiles. About 15 carry titles
     (Microsoft, Bill Gates, Harvard University, The Founder (2016 film),
     Apple Inc., Steve Jobs, Stanford University, Yale University, Amazon,
     Jeff Bezos, Intel, Reed College, Paul Allen, Redmond, Cambridge); the
     rest are unlabelled, to suggest scale. All inert grey until touched.
     No lines between tiles -- there is no link structure, and the absence
     must be visible.
  4. Space below the grid where retrieved cards rise and the answer appears.

Below both tracks: one caption line, large, changing per frame.
Below that: prev / step dots / next / replay / autoplay toggle.

In the question, highlight "Microsoft" from frame 1 in a distinct colour.
That colour means "words the question actually contains" for the whole
animation and is never used for anything else.

### The ten frames

Ten frames, about 18 seconds of narration each. One idea per frame; do not
combine them.

FRAME 1 — the board
  Question in, "Microsoft" highlighted. Both tracks idle, stores grey, query
  slots empty, counters at zero.
  Caption: "One question. Two documents needed."

FRAME 2 — both fire, same result
  A beam in the question-words colour travels from the question into each
  store. The SAME three tiles light in both and rise as cards:
      "Microsoft"                 correct (green edge)
      "The Founder (2016 film)"   wrong (red edge)
      "Microsoft Research"        neutral
  Both counters -> queries 1, docs 1/2.
  The tracks have not diverged, and that must be visually obvious.
  Caption: "Same search. Same result."

FRAME 3 — the wrong document
  Zoom "The Founder (2016 film)" large enough to read. Inside its text,
  "Founder" lights in the question-words colour and a line connects it to the
  matching word in the question above. Show enough text for the absurdity to
  land: a 2016 film about Ray Kroc and a hamburger franchise.
  This is the most memorable frame. Give it the most room.
  Caption: "Matched on 'founder'. A hamburger franchise."

FRAME 4 — the left track ends
  Left track dims and freezes, with a terminal marker. Counter holds at
  1 query, 1/2 docs. No answer.
  Right track stays bright and live. The asymmetry is the point: one side is
  over, the other is still running.
  Caption: "It cannot know what it missed."

FRAME 5 — reading
  On the right, the "Microsoft" card expands to its full text. Entity
  candidates underline one at a time: "Redmond", "Bill Gates", "Paul Allen",
  "Altair 8800".
  Use a SECOND colour for these -- words the question did NOT contain. The
  colour change carries the whole idea and must be unmistakable.
  Caption: "Reading gives words the question never had."

FRAME 6 — THE BRIDGE
  The single most important motion on the page.
  "Bill Gates" physically LIFTS OUT of the card text, travels up and across
  along a visible path, and lands in the empty second query slot, where it
  assembles into:
      bill gates university attend
  The slot was empty; now it is filled, and the viewer saw where the words
  came from. At the same moment, grey out the question at the top for a beat
  -- this query could not have come from it.
  Then a beam in the SECOND colour fires from that slot into the store.
  Counter -> queries 2.
  Caption: "A query built from the document, not the question."

FRAME 7 — THE PROOF
  The frame that closes the argument. Do not shorten it.
  A new tile lights and rises: "Bill Gates", green edge, text shown with
  "Harvard University" highlighted.
  Then show the absence. Highlight the phrase "the software company he had
  founded", and show that "Microsoft" does not occur in this document: a
  struck-through "Microsoft" chip, a "not found in this document" marker, or
  the question's highlighted word visibly failing to find a match here.
  Pick one and make it unambiguous.
  Draw a persistent edge from the "Microsoft" card to this card -- a link
  that did not exist in the store and was created by reading.
  Counter -> docs 2/2.
  Caption: "This document never says 'Microsoft'."

FRAME 8 — stopping
  A gauge on the right track fills, crosses a marked threshold, and the track
  halts itself. Show a measurement being taken and a decision following from
  it; do not explain it in words.
  Caption: "Stops when another round would add nothing."

FRAME 9 — the answer and its citation
  Right track answer: "Harvard University", marked correct.
  Below it one citation chip, in exactly this format:
      d8c0128ba827::p0::s1
  Hovering or focusing the chip draws a line back to the exact highlighted
  sentence in the Bill Gates card. Make this interaction work -- it is the
  proof that citations are locations, not decoration.
  Caption: "Every claim traced to a sentence."

FRAME 10 — final state
  Both tracks held side by side:
      LEFT   1 query  | 1/2 docs | no answer
      RIGHT  2 queries| 2/2 docs | "Harvard University"  correct
  Caption: none. The frame speaks.

### One panel below the fold

Two measured bars, nothing else:

    title link graph   30.4%   [REJECTED]
    mention index      90.3%

Label: "reachable gold evidence, 831 links, measured both ways".

No paragraph. The two bars and the REJECTED mark are the whole story: the
link graph was built first and killed by that number.

## TAB 2 — RESULTS

Charts, not prose. Exact numbers visible as labels on the marks.

One line of context: the Microsoft example is an illustration; these numbers
are from MuSiQue-Answerable v1.0 dev, sha256-pinned, 500 questions, of which
54 were dropped because all their gold evidence sits inside one document.

Lead with the 350 HELD-OUT questions. The first 150 were development data.
All-500 numbers behind a toggle, never as the headline.

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

oracle is gold documents handed in directly -- draw it as a reference line,
not a competing bar.

### Answers, 350 held-out questions, SAME retrieval and SAME LLM reader

Retrieval is identical between these rows; only the reader differs. Make that
visually explicit -- the rows should look like they share one context.

| System            |  EM  |  F1  |
|-------------------|------|------|
| closed_book_llm   |  0.0 |  0.0 |
| hybrid_rerank_llm | 23.1 | 30.7 |
| SOUL (ours_llm)   | 34.0 | 41.4 |
| oracle_llm        | 65.7 | 74.4 |

EM +10.9, 95% CI [+6.9, +15.4], Holm-adjusted p = 0.0012.
F1 +10.7, 95% CI [+6.6, +15.1], Holm-adjusted p = 0.0012.

closed_book_llm scores 0.0 and abstains on every question -- the leakage
control. Draw it as a zero bar; it earns its place by being empty.

Small print: paired permutation tests, 10,000 resamples, Holm-Bonferroni
corrected across the metric family.

### AllDocs by hop count, all 500 questions

Grouped bars. Shows the win and the failure in one chart.

| System        | 2-hop | 3-hop | 4-hop |
|---------------|-------|-------|-------|
| bm25          | 26.0  | 12.4  |  1.2  |
| hybrid_rerank | 46.9  | 25.5  |  1.2  |
| SOUL          | 74.0  | 37.9  |  4.7  |

### Limitation — a full section, same visual weight as the wins

On 4-hop questions AllDocs is 4.7%, against 74.0% on 2-hop. Partial coverage
is 46.4% -- it finds part of the chain and rarely all of it. Four hops means
three consecutive bridges, and per-hop error compounds.

A bridge-diversity fix was tried and made held-out results WORSE: AllDocs
-3.1 points, p = 0.026. It was reverted and the negative result kept.

Draw the -3.1 as a bar going the wrong way. Not a footnote, not a collapsed
accordion, not greyed out. A viewer must not be able to miss it.

## TAB 3 — ARCHITECTURE

A vertical pipeline of seven stages, each block expandable to show its real
components and parameters. Collapsed, the page reads in ten seconds.

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

5. ITERATIVE RETRIEVAL  — draw as a LOOP. The only stage that cycles, and
   that must be visually obvious at a glance.
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

A small grid of which stages each system uses:

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
- Colour carries meaning and is never decorative. Fix the palette at the
  start and keep it: one colour for words from the question, a second for
  words discovered inside documents, plus correct / wrong / neutral / rejected.
- Motion must be purposeful -- things move to show where information came
  from. No fades or slides that mean nothing.
- Respect prefers-reduced-motion: expose the same frames as static steps.
- Must work at 1920x1080 for recording AND on a phone (tracks stack
  vertically there). Correct in both light and dark mode; dark is likely what
  gets recorded, so check it first.
- Plain descriptive title. No author name.
