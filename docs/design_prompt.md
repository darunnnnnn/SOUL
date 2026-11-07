Build me a two-page interactive artifact.

PAGE 1 is a single animated visualisation, full-screen, that shows how my
retrieval system works and how it differs from standard RAG. It is a
visualisation, NOT an explainer page. PAGE 2 is the evaluation results.

## HARD CONSTRAINT ON PAGE 1

Page 1 must contain NO explanatory paragraphs. No intro text, no "how it
works" section, no bullet list of stages, no body copy. Nothing the reader
has to read in order to understand the mechanism.

The mechanism must be understood by WATCHING IT HAPPEN.

The only text permitted anywhere on the canvas:
  - the question, once, at the top
  - short labels ON objects (a document title, a query string, a stage name)
  - one caption line under the canvas, max 10 words, which changes per frame
  - the two outcome values at the end

If a reader could learn the mechanism by reading page 1 instead of watching
it, you have built the wrong thing. Budget your effort accordingly: this
animation is 80% of the work and the results page is 20%.

## THE CANVAS

Two tracks, side by side, sharing one question at the top. Both tracks are
always visible so the divergence is seen rather than remembered.

  LEFT TRACK  - labelled "single-shot retrieval"
  RIGHT TRACK - labelled "iterative, entity-bridged"

Behind each track: the corpus, drawn as a field of ~250 small dim dots
representing 6,793 passages (do not draw all of them; suggest scale). Dots
are inert grey until touched.

The question, shown once at the top, centred, feeding both tracks:

    "Who was in charge of the state where Graft-De Rijp is located?"

Highlight the token "Graft-De Rijp" inside the question in a distinct colour
from frame 1 and keep that colour consistent for the rest of the animation.
The reader should come to associate that colour with "words the question
actually contains".

## STORYBOARD - the frames

Driven by a step control (prev / next / replay) AND an autoplay toggle.
Default to step control so a presenter can pause on a frame. Each frame has
one idea. Do not combine them.

FRAME 1 - the question arrives
  Question fades in at top. "Graft-De Rijp" lights up. Both tracks idle.
  Caption: "One question. Two documents needed."

FRAME 2 - both tracks fire one query
  From the question, a beam travels down into each track's dot field.
  The beam is coloured with the "question words" colour - this is the visual
  claim that both systems can only search for words the question contains.
  Caption: "Both search what the question says."

FRAME 3 - both retrieve
  In each field, 3 dots light up and rise out of the field as cards.
  Both tracks retrieve the SAME cards at this point. This matters - the
  systems have not diverged yet.
    card: "Graft-De Rijp"          -> mark correct (green edge)
    card: "The Case Against Brooklyn (1958)"  -> mark wrong (red edge)
    card: "Alkmaar"                -> neutral
  Caption: "Same result so far."

FRAME 4 - why the wrong card was retrieved
  Zoom the "The Case Against Brooklyn" card. Inside its text, the word
  "graft" lights up in the question-words colour, visually matching the
  highlighted token in the question above. A thin line connects the two.
  Caption: "Matched on 'graft'. Political graft."
  This frame is the single best moment in the animation. Give it room.

FRAME 5 - LEFT TRACK ENDS
  Left track dims and freezes. A terminal marker appears on it.
  Its answer renders: "Alkmaar", struck through or marked wrong.
  A small counter on the left track settles at: 1 query, 1 of 2 documents.
  Right track stays live and bright. The asymmetry is the whole point -
  one side is over, one side is still running.
  Caption: "Single-shot is finished. It cannot know what it missed."

FRAME 6 - RIGHT TRACK READS what it retrieved
  The "Graft-De Rijp" card expands to show its real text:
    "Graft-De Rijp () is a former municipality in the Netherlands, in the
     province of North Holland. Since 2015 it has been a part of Alkmaar."
  Entity candidates underline one by one: "Netherlands", "North Holland",
  "Alkmaar". Use a NEW colour for these - these are words the question did
  NOT contain. The colour change carries the entire idea.
  Caption: "Reading gives words the question never had."

FRAME 7 - THE BRIDGE
  The phrase "North Holland" physically LIFTS OUT of the card text, travels
  up and across, and lands in an empty query slot, where it assembles into:
    north holland charge state located
  Show the slot was empty before. Show it being filled from the document,
  not from the question. Trace the path visibly - this single motion is the
  contribution of the whole system and the reader must see where the words
  came from.
  Simultaneously, grey out the question at the top for a beat, to show this
  query could not have come from it.
  Caption: "A query built from the document, not the question."

FRAME 8 - round 2 fires
  A beam in the NEW colour travels from the bridge query into the dot field.
  Visually distinct from frame 2's beam. Same mechanism, different origin.
  Caption: "Round 2."

FRAME 9 - the second document arrives
  A new dot lights up and rises: card "North Holland", green edge, showing:
    "...The King's Commissioner of North Holland is Johan Remkes, serving
     since 2010."
  "Johan Remkes" highlights inside it.
  A line draws from the "Graft-De Rijp" card to this card - the bridge made
  visible as a persistent edge.
  Caption: "Found via the first document."

FRAME 10 - the stop decision
  A small gauge or meter on the right track fills and crosses a threshold,
  then the track halts itself. Do not explain it in text - show a measure
  being taken and a decision being made.
  Right track counter settles at: 13 queries, 2 of 2 documents.
  Caption: "Stops when another round would add nothing."

FRAME 11 - the answer, with provenance
  Right track answer renders: "Johan Remkes", marked correct.
  Two citation chips appear beneath it:
    47797f5b0cc7::p0::s0      17617c818820::p0::s1
  Hovering or focusing a chip draws a line back to the exact highlighted
  sentence in its card. Make that interaction work - it is the proof that
  citations are real and not decorative.
  Caption: "Every claim traced to a sentence."

FRAME 12 - final state, both tracks held side by side
  LEFT:  1 query   | 1 of 2 docs | "Alkmaar"      | wrong
  RIGHT: 13 queries| 2 of 2 docs | "Johan Remkes" | correct
  Caption: "" (none - the frame speaks)

## PAGE 1 - ONE SECONDARY PANEL, BELOW THE FOLD

After the animation, one panel only. Also visual, minimal text.

A rejected design, shown as two measured bars:

    title link graph   30.4%
    mention index      90.3%

Label: "reachable gold evidence, 831 links, measured both ways".
Mark the 30.4% bar as REJECTED.

I built the link graph first, measured it, and killed it. Show the number
that killed it. Do not write a paragraph about it - the two bars and the
REJECTED mark are the whole story.

Nothing else on page 1.

## PAGE 2 - RESULTS

Charts, not prose. Same rule: minimal text, numbers visible as labels.

Context: 500 questions from MuSiQue-Answerable v1.0 dev, sha256-pinned. 54
questions were dropped because all their gold evidence sits inside a single
document - several chunks of one source is not multi-document evidence.
State that as one line, not a paragraph.

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

oracle is gold documents handed in directly - a ceiling, not a competitor.
Style it as a reference line, not a bar.

### Answers, 350 held-out questions - SAME retrieval, LLM reader

Retrieval is IDENTICAL between these two rows; only the reader differs, so
the gain is synthesis over the same context. Make that visually explicit -
the two rows should be shown sharing one context.

| System            |  EM  |  F1  |
|-------------------|------|------|
| closed_book_llm   |  0.0 |  0.0 |
| hybrid_rerank_llm | 23.1 | 30.7 |
| MINE (ours_llm)   | 34.0 | 41.4 |
| oracle_llm        | 65.7 | 74.4 |

EM +10.9 points, 95% CI [+6.9, +15.4], Holm-adjusted p = 0.0012.
F1 +10.7 points, 95% CI [+6.6, +15.1], Holm-adjusted p = 0.0012.

closed_book_llm scores 0.0 and abstains on every question - the leakage
control. Show it as a zero bar; it earns its place by being empty.

Statistics, in small print: paired permutation tests, 10,000 resamples,
Holm-Bonferroni corrected across the metric family.

### By hop count, AllDocs, all 500 questions

Grouped bar chart. This shows where the method wins and where it breaks.

| System        | 2-hop | 3-hop | 4-hop |
|---------------|-------|-------|-------|
| bm25          | 26.0  | 12.4  |  1.2  |
| hybrid_rerank | 46.9  | 25.5  |  1.2  |
| MINE          | 74.0  | 37.9  |  4.7  |

### LIMITATION - full section, same visual weight as the wins

On 4-hop questions AllDocs is 4.7%. Partial coverage is 41% - it finds some
of the chain and rarely all of it. Three consecutive bridges compound
per-hop error.

I tried a beam-diversity fix. Held-out results got WORSE: AllDocs -3.1
points, p = 0.030. I reverted it and kept the negative result.

Show the -3.1 as a bar going the wrong way. Not a footnote, not a collapsed
accordion, not greyed out. A reader must not be able to miss it.

## GLOBAL DESIGN DIRECTION

- Restrained and technical. A research artifact, not a product launch. No
  gradient hero, no emoji, no marketing superlatives.
- Colour carries meaning and must stay consistent throughout:
  one colour = words from the question, a second = words discovered in
  documents, plus correct / wrong / neutral. Do not use colour decoratively.
- Motion must be purposeful: things move to show where information came
  from. No fades or slides that carry no meaning.
- Respect prefers-reduced-motion: expose the same frames as static steps.
- Works on a phone (tracks may stack vertically there) and correct in both
  light and dark mode.
- Plain descriptive title. No author name.
