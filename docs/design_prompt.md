Build me a two-page interactive artifact.

PAGE 1 is a single animated visualisation, full-screen, that shows how my
retrieval system works and how it differs from standard RAG. It is a
visualisation, NOT an explainer page. PAGE 2 is the evaluation results.

## HARD CONSTRAINT ON PAGE 1

Page 1 must contain NO explanatory paragraphs. No intro text, no "how it
works" section, no bullet list of stages, no body copy. Nothing the reader has
to read in order to understand the mechanism.

The mechanism must be understood by WATCHING IT HAPPEN.

The only text permitted anywhere on the canvas:
  - the question, once, at the top
  - short labels ON objects (a document title, a query string, a stage name)
  - one caption line under the canvas, max 10 words, which changes per frame
  - the two outcome values at the end

If a reader could learn the mechanism by reading page 1 instead of watching
it, you have built the wrong thing. Budget your effort accordingly: this
animation is 80% of the work and the results page is 20%.

## THE EXAMPLE TO ANIMATE

Use this one throughout. It is deliberately a familiar example, chosen so the
reader spends no effort on the subject matter and all of it on the mechanism.

    Question: "Which university did the founder of Microsoft attend?"
    Answer:   "Harvard University"

    Document A - "Microsoft":
      "Microsoft Corporation is an American multinational technology company
       headquartered in Redmond, Washington. It was founded by Bill Gates and
       Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for
       the Altair 8800."

    Document B - "Bill Gates":
      "William Henry Gates III is an American businessman and philanthropist.
       He enrolled at Harvard University in 1973 and left two years later,
       before completing his degree, to work full time on the software company
       he had founded."

Read those two documents carefully, because the animation turns on exactly
what each one does and does not say:

  - Document A names the founder. It says nothing about any university.
  - Document B names the university. It NEVER SAYS "MICROSOFT" -- it says
    "the software company he had founded".

So neither document answers the question alone. And the second document cannot
be found by searching the question, because the question's distinctive word is
"Microsoft" and that word does not occur in Document B. The only way in is to
read Document A first, take the name "Bill Gates" out of it, and search again.

That is the mechanism, and the animation exists to make that one fact visible.
Do not soften it: if Document B is ever drawn containing the word "Microsoft",
the whole page stops making its point.

## THE CANVAS

Two tracks side by side, sharing one question at the top. Both tracks stay
visible throughout so the divergence is seen rather than remembered.

  LEFT TRACK  - labelled "single-shot retrieval"
  RIGHT TRACK - labelled "iterative, entity-bridged"

Behind each track: the document store, drawn as a field of ~45 small dim
tiles, a handful of them labelled (Microsoft, Bill Gates, Harvard University,
The Founder (2016 film), Apple Inc., Steve Jobs, Stanford University, Yale
University, Amazon, Jeff Bezos, Intel, Reed College, Paul Allen, Redmond,
Cambridge). Tiles are inert grey until touched. The store must look like many
independent articles with no connections drawn between them -- there is no
link structure to follow, and the reader should see that there is none.

Highlight the word "Microsoft" inside the question from frame 1, in a distinct
colour, and keep that colour meaning "words the question actually contains"
for the whole animation.

## STORYBOARD - the frames

Driven by a step control (prev / next / replay) AND an autoplay toggle.
Default to the step control so a presenter can hold a frame. One idea per
frame. Do not combine them.

FRAME 1 - the question arrives
  Question fades in at top. "Microsoft" lights up. Both tracks idle.
  Caption: "One question. Two documents needed."

FRAME 2 - both tracks fire one query
  A beam travels from the question down into each track's tile field, in the
  question-words colour: both systems can only search for words the question
  contains.
  Caption: "Both search what the question says."

FRAME 3 - both retrieve the same thing
  In each field the same 3 tiles light up and rise as cards. The tracks have
  not diverged yet, and that matters.
    "Microsoft"                 -> correct (green edge)
    "The Founder (2016 film)"   -> wrong (red edge)
    "Microsoft Research"        -> neutral
  Caption: "Same result so far."

FRAME 4 - why the wrong card came back
  Zoom "The Founder (2016 film)". Inside its text, "Founder" lights up in the
  question-words colour, matching the question above; draw a thin line between
  the two. Show enough of the card for the absurdity to land: it is a 2016
  film about Ray Kroc and a hamburger franchise.
  This is the most memorable frame in the animation. Give it room.
  Caption: "Matched on 'founder'. A hamburger franchise."

FRAME 5 - LEFT TRACK ENDS
  Left track dims and freezes; a terminal marker appears. Its counter settles
  at 1 query, 1 of 2 documents. Right track stays live and bright.
  The asymmetry is the point: one side is over, the other is still running.
  Caption: "Single-shot is finished. It cannot know what it missed."

FRAME 6 - RIGHT TRACK READS what it retrieved
  The "Microsoft" card expands to its full text. Entity candidates underline
  one at a time: "Redmond", "Bill Gates", "Paul Allen", "Altair 8800".
  Use a NEW colour for these -- words the question did NOT contain. The colour
  change carries the whole idea.
  Caption: "Reading gives words the question never had."

FRAME 7 - THE BRIDGE
  "Bill Gates" physically LIFTS OUT of the card text, travels up and across,
  and lands in an empty query slot, where it assembles into:
      bill gates university attend
  Show the slot empty beforehand. Trace the path visibly. This single motion
  is the contribution of the entire system; the reader must see where those
  words came from.
  At the same moment, grey out the question at the top for a beat -- this
  query could not have come from it.
  Caption: "A query built from the document, not the question."

FRAME 8 - round 2 fires
  A beam in the NEW colour travels from the bridge query into the tile field.
  Visually distinct from frame 2's beam: same mechanism, different origin.
  Caption: "Round 2."

FRAME 9 - the second document arrives, and the proof
  A new tile lights and rises: "Bill Gates", green edge, showing its text with
  "Harvard University" highlighted.
  Then do the thing that proves the point: highlight the phrase "the software
  company he had founded" and show that the word "Microsoft" is ABSENT from
  this document -- a struck-through "Microsoft" marker, a "not found" tag, or
  the question's highlighted word visibly failing to find a match here.
  Draw a persistent edge from the "Microsoft" card to this card: the bridge,
  made into a visible link that did not exist in the store.
  Caption: "This document never says 'Microsoft'."

FRAME 10 - the stop decision
  A small gauge on the right track fills, crosses a threshold, and the track
  halts itself. Do not explain it in words -- show a measurement being taken
  and a decision following from it.
  Right track counter settles at 2 queries, 2 of 2 documents.
  Caption: "Stops when another round would add nothing."

FRAME 11 - the answer, with provenance
  Right track answer renders: "Harvard University", marked correct.
  Beneath it, one citation chip in this exact format:
      d8c0128ba827::p0::s1
  Hovering or focusing the chip draws a line back to the exact highlighted
  sentence in the Bill Gates card. Make that interaction work: it is the proof
  that citations are real locations and not decoration.
  Caption: "Every claim traced to a sentence."

FRAME 12 - final state, both tracks held side by side
  LEFT:  1 query  | 1 of 2 docs | no answer
  RIGHT: 2 queries| 2 of 2 docs | "Harvard University" | correct
  Caption: "" (none -- the frame speaks)

## PAGE 1 - ONE SECONDARY PANEL, BELOW THE FOLD

One panel only, also visual, minimal text.

A rejected design, as two measured bars:

    title link graph   30.4%
    mention index      90.3%

Label: "reachable gold evidence, 831 links, measured both ways".
Mark the 30.4% bar REJECTED.

I built the link graph first, measured it, and killed it. Show the number that
killed it. No paragraph: the two bars and the REJECTED mark are the story.

Nothing else on page 1.

## PAGE 2 - RESULTS

Charts, not prose. Minimal text, numbers visible as labels.

One line of context, not a paragraph: the Microsoft example above is an
illustration; all numbers below come from MuSiQue-Answerable v1.0 dev,
sha256-pinned, 500 questions, of which 54 were dropped because all their gold
evidence sits inside a single document -- several chunks of one source is not
multi-document evidence.

The first 150 questions were held out as development data. Lead with the 350
held-out questions. All-500 numbers behind a toggle, never as the headline.

Metric meanings in tooltips only:
    AllDocs - 1 only if EVERY required document is in context (all-or-nothing)
    DocCov  - fraction of required documents in context (partial credit)
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
| MINE (ours)     |  38.9   |  70.3  |  34.0  |
| oracle          | 100.0   | 100.0  | 100.0  |

ours vs hybrid_rerank, AllDocs: +14.0 points, 95% CI [+9.4, +18.9],
Holm-adjusted p = 0.0012, Cohen's d = 0.31.

oracle is gold documents handed in directly -- a ceiling, not a competitor.
Draw it as a reference line, not a bar.

### Answers, 350 held-out questions - SAME retrieval, LLM reader

Retrieval is IDENTICAL between these two rows; only the reader differs, so the
gain is synthesis over the same context. Make that visually explicit: the two
rows should be shown sharing one context.

| System            |  EM  |  F1  |
|-------------------|------|------|
| closed_book_llm   |  0.0 |  0.0 |
| hybrid_rerank_llm | 23.1 | 30.7 |
| MINE (ours_llm)   | 34.0 | 41.4 |
| oracle_llm        | 65.7 | 74.4 |

EM +10.9 points, 95% CI [+6.9, +15.4], Holm-adjusted p = 0.0012.
F1 +10.7 points, 95% CI [+6.6, +15.1], Holm-adjusted p = 0.0012.

closed_book_llm scores 0.0 and abstains on every question -- the leakage
control. Show it as a zero bar; it earns its place by being empty.

Statistics, small print: paired permutation tests, 10,000 resamples,
Holm-Bonferroni corrected across the metric family.

### By hop count, AllDocs, all 500 questions

Grouped bar chart. Shows where the method wins and where it breaks.

| System        | 2-hop | 3-hop | 4-hop |
|---------------|-------|-------|-------|
| bm25          | 26.0  | 12.4  |  1.2  |
| hybrid_rerank | 46.9  | 25.5  |  1.2  |
| MINE          | 74.0  | 37.9  |  4.7  |

### LIMITATION - full section, same visual weight as the wins

On 4-hop questions AllDocs is 4.7%, against 74.0% on 2-hop. Partial coverage
is 46.4%, so it finds part of the chain and rarely all of it: three
consecutive bridges, and per-hop error compounds.

I tried a bridge-diversity fix. Held-out results got WORSE: AllDocs -3.1
points, p = 0.026. I reverted it and kept the negative result.

Show the -3.1 as a bar going the wrong way. Not a footnote, not a collapsed
accordion, not greyed out. A reader must not be able to miss it.

## GLOBAL DESIGN DIRECTION

- Restrained and technical. A research artifact, not a product launch. No
  gradient hero, no emoji, no marketing superlatives.
- Colour carries meaning and stays consistent: one colour = words from the
  question, a second = words discovered inside documents, plus correct /
  wrong / neutral. Never use colour decoratively.
- Motion must be purposeful: things move to show where information came from.
  No fades or slides that carry no meaning.
- Respect prefers-reduced-motion: expose the same frames as static steps.
- Works on a phone (tracks may stack vertically) and correct in light and dark
  mode.
- Plain descriptive title. No author name.
