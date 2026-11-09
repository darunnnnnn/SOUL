The Mechanism tab needs a structural change. It currently shows the loop
running once, which is the wrong idea: a viewer sees "it does one extra
search". The actual claim is that this is a LOOP — each round discovers the
term that makes the next round possible — and that is only visible when it
happens twice. Please rework it to a three-document chain.

Keep the Results and Architecture tabs as they are.

## 1. New question and chain

Replace the current question with:

    "In which city is the university attended by the founder of Microsoft
     located?"          Answer: Cambridge

Three documents, found in order, each containing the search term for the next:

    Document 1 — "Microsoft"
      "Microsoft Corporation is an American multinational technology company
       headquartered in Redmond, Washington. It was founded by Bill Gates and
       Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for
       the Altair 8800."
      -> gives BILL GATES

    Document 2 — "Bill Gates"
      "William Henry Gates III is an American businessman and philanthropist.
       He enrolled at Harvard University in 1973 and left two years later,
       before completing his degree, to work full time on the software company
       he had founded."
      -> gives HARVARD UNIVERSITY

    Document 3 — "Harvard University"
      "Harvard University is a private Ivy League research university in
       Cambridge, Massachusetts. Established in 1636, it is the oldest
       institution of higher learning in the United States."
      -> gives the answer, CAMBRIDGE

The question contains: city, university, attended, founder, Microsoft.
It does NOT contain "Bill Gates", "Harvard" or "Cambridge". Every term needed
to reach documents 2 and 3 exists only inside an earlier document. That is the
whole argument.

Do not reword these documents to read more smoothly. Document 2 must never
contain the word "Microsoft" — it says "the software company he had founded",
and that absence is shown on screen as proof.

## 2. Right column becomes a stack of rounds

Instead of one or two query slots, the right column is a vertical stack of
THREE round-slots, all drawn from frame 1 so the viewer sees there is room for
more than one round:

    ROUND 1   [ the question        ]  ->  [ document ]  ->  [ new word ]
    ROUND 2   [ empty, dashed       ]  ->  [          ]  ->  [          ]
    ROUND 3   [ empty, dashed       ]  ->  [          ]  ->  [          ]

Round 1 is pre-filled with the question. Rounds 2 and 3 are visibly empty and
awaiting content — dashed outlines, not blank panels.

The left column keeps its single query box. It will finish early and then sit
frozen while the right column keeps working; that asymmetry is a large part of
the argument, so make it obvious.

## 3. Add an accumulating evidence panel

New, and important. The right column gets a panel that persists and GROWS:
one document after round 1, two after round 2, three after round 3. It never
empties.

This is how "combine what we found and search again" becomes something the
viewer watches instead of infers. Make the growth physical and obvious.

## 4. The twelve frames

FRAME 1  — the board. Question in, "Microsoft" highlighted. Stores grey.
           Three round-slots, two empty. Evidence panel empty. Counters 0, 0/3.
           Caption: "Three documents needed. The question names one."

FRAME 2  — both columns run round 1 and get the SAME three documents:
           "Microsoft" (correct), "The Founder (2016 film)" (wrong),
           "Microsoft Research" (neutral). Counters 1 query, 1/3.
           Caption: "Same search. Same result."

FRAME 3  — zoom "The Founder (2016 film)". "Founder" inside it lights in the
           question colour, with a line to the question. Show enough text that
           the absurdity lands: a 2016 film about Ray Kroc and a hamburger
           franchise.
           Caption: "Matched on 'founder'. A hamburger franchise."

FRAME 4  — left column dims and freezes. 1 query, 1 of 3 documents, no answer.
           Right column stays bright with two empty rounds still visible.
           Caption: "One search, and it is over. Two documents still missing."

FRAME 5  — right column reads the Microsoft card. Entities underline one at a
           time in the discovery colour: "Redmond", "Bill Gates", "Paul Allen".
           Caption: "Reading gives words the question never had."

FRAME 6  — BRIDGE 1. "Bill Gates" lifts out of the document text along a
           visible path and lands in ROUND 2's empty query box, assembling
           into: bill gates university attend
           Grey the question out for a beat.
           Caption: "A query built from the document, not the question."

FRAME 7  — round 2 retrieves the "Bill Gates" document. Evidence panel grows
           to TWO. Counter 2/3. Then show the absence: highlight "the software
           company he had founded" and mark that "Microsoft" does not occur in
           this document. Draw a persistent edge between the two cards.
           Caption: "This document never says 'Microsoft'."

FRAME 8  — read the new card. "Harvard University" underlines in the discovery
           colour. Use deliberately the same treatment as frame 5.
           Caption: "Read again. New words again."

FRAME 9  — BRIDGE 2. THE MOST IMPORTANT FRAME.
           "Harvard University" lifts out and lands in ROUND 3's query box:
           harvard university city located
           This motion must be VISUALLY IDENTICAL to frame 6 — same path
           style, same colour, same timing. The viewer has to recognise it as
           the same operation happening again. If frames 6 and 9 look
           different, the page reads as two one-off tricks instead of a loop,
           and the tab fails at its only job.
           Consider showing both completed bridges stacked here, so the
           repetition is visible in a single glance.
           Caption: "The same move, again."

FRAME 10 — round 3 retrieves "Harvard University" with "Cambridge,
           Massachusetts" highlighted. Evidence panel grows to THREE.
           Counter 3/3. A third edge joins, so the cards form a visible chain.
           Caption: "Three rounds. Three documents. A chain."

FRAME 11 — a gauge fills, crosses a threshold, and the loop halts itself.
           Caption: "Stops when another round would add nothing."

FRAME 12 — answer "Cambridge", correct, with three citation chips:
             25e081610f91::p0::s1   d8c0128ba827::p0::s1   f970f3994adc::p0::s0
           Hovering a chip draws a line back to its highlighted sentence.
           Final comparison held side by side:
             LEFT   1 query  | 1 of 3 docs | no answer
             RIGHT  3 queries| 3 of 3 docs | "Cambridge"  correct
           Caption: none.

## 5. Fix the sizing — this page gets screen-recorded

The current layout is too small and too sparse for video. It will be recorded
at 1920x1080 and watched in a video player, so:

- The canvas must FILL the viewport. If the two columns occupy less than ~80%
  of the width and ~60% of the height, it is too small.
- Nothing below 14px. Document tiles that carry a title need readable type.
- The caption line is effectively a subtitle: 24-28px.
- Video compression destroys thin lines and low-contrast grey. Strokes 2px
  minimum, high contrast throughout.
- Check dark mode first — that is what will be recorded.
- Keep the step control reachable without scrolling, and keep autoplay off by
  default.

## 6. Colour discipline, unchanged but worth restating

Two meanings, used for nothing else:
  colour A — words the question contains
  colour B — words discovered inside a document

Every bridge term is in colour B. A viewer who notices only "things in colour
B came out of documents, not out of the question" has understood the system.
