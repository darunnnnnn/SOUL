Build me a two-page interactive artifact that explains a retrieval system I
built and shows its evaluation results. Page 1 explains the mechanism and how
it differs from standard RAG, centred on a diagram. Page 2 is the results.
Navigation between the two pages, no scrolling marathon.

Everything below is real measured data. Do not invent numbers, do not round
them differently, and do not add metrics I have not given you.

## THE PROBLEM (this is what the whole page has to make obvious)

A knowledge base of 5,848 independent Wikipedia articles -> 6,793 passages ->
24,064 sentence-level anchors. No links between documents, no folder
structure, no table of contents. Nothing in the corpus says which documents
belong together.

Some questions cannot be answered by any single document. The hard part is
not that you need two documents. The hard part is that THE QUESTION DOES NOT
CONTAIN THE WORDS YOU NEED TO FIND THE SECOND DOCUMENT.

Worked example, use this exact one throughout the diagram:

    Question: "Who was in charge of the state where Graft-De Rijp is located?"
    Correct answer: "Johan Remkes"

    Document A - "Graft-De Rijp":
      "Graft-De Rijp () is a former municipality in the Netherlands, in the
       province of North Holland. Since 2015 it has been a part of Alkmaar."

    Document B - "North Holland":
      "The capital and seat of the provincial government is Haarlem... The
       King's Commissioner of North Holland is Johan Remkes, serving since
       2010."

The question says "Graft-De Rijp". It never says "North Holland". But
"Johan Remkes" only exists in Document B. So the search term you need exists
nowhere in the question - it exists only INSIDE Document A.

## WHAT STANDARD RAG DOES (left side of the diagram)

One embedding of the question, one search, top-k passages, feed to reader. It
is a single shot. Better embeddings, a bigger k, or a cross-encoder reranker
do not fix this, because the system is searching for a string it does not
have yet.

On this exact question my strongest baseline (BM25 + dense + cross-encoder
reranker) retrieved "The Case Against Brooklyn", a 1958 film noir. It matched
on the word "graft" - political graft. It answered "Alkmaar". Wrong. It found
1 of the 2 required documents.

Show that wrong retrieval in the diagram. The film noir is the single most
memorable detail on the page - it makes the failure concrete instead of
abstract.

## WHAT MINE DOES (right side of the diagram)

An iterative loop. Four stages, show them as a cycle, not a straight line:

    1. RETRIEVE - hybrid search (BM25F + dense + cross-encoder rerank)
    2. READ     - extract candidate entities from the passages just retrieved
    3. BRIDGE   - build a NEW query from an entity that was not in the
                  original question
    4. STOP?    - estimate whether another round would add anything; if not,
                  stop

On the example: round 1 retrieves the Graft-De Rijp passage. Reading it
yields the entity "North Holland". That becomes the query
`north holland charge state located` - a query the original question could
not have produced. Round 2 retrieves the North Holland passage. Both gold
documents are now in context. Answer: Johan Remkes, with two citations from
two different documents.

13 queries issued instead of 1.

### The design decision worth showing as its own panel

My first version connected documents by a title link graph - if document A's
text contains document B's title, link them. I measured how much of the
required evidence that could ever reach, and it was 30.4%. A mention index
over surface forms instead - every capitalised span and noun phrase, not just
exact titles - reaches 90.3%. Same 831 gold links, measured both ways.

Show 30.4% vs 90.3% as a direct comparison. The point is that I built the
obvious thing first, measured it, and killed it with a number. Label it
honestly as a rejected design, not as a feature.

### Verifiable citations

Every claim in an answer carries an anchor of the form `doc_id::p0::s1` -
document, passage, sentence - with byte offsets. The anchor is re-read from
the source corpus and compared byte-for-byte. All 24,064 anchors re-resolve
exactly; 0 mismatches. A fabricated anchor is rejected. Show the anchor
format itself somewhere; the triple-colon form is distinctive.

## PAGE 2 - RESULTS

500 questions from MuSiQue-Answerable v1.0 dev, sha256-pinned. 54 questions
were DROPPED because all their gold evidence sits inside a single document -
several chunks of one source is not multi-document evidence. Say this
explicitly on the page; it is a deliberate choice to decline easy credit.

The first 150 questions (sorted by id) were held out as development data.
Report the 350 held-out questions as the headline, and make that prominent.
Numbers from all 500 can be a secondary row or a toggle, but held-out is the
honest one and should be what the reader sees first.

Metric definitions to show in a tooltip or small print, not as a wall:

    AllDocs - 1 only if EVERY required document is in the context (all-or-nothing)
    DocCov  - fraction of required documents in the context (partial credit)
    SupAll  - 1 only if every required passage is in the context
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

oracle is gold documents handed in directly - it is the ceiling, not a
competitor. Style it differently from the real systems.

### Answers, 350 held-out questions, same retrieval, LLM reader

Retrieval is IDENTICAL between these two rows. Only the reader differs. So
the gain is synthesis over the same context, which is the cleanest way to
show the context layer is what is doing the work.

| System            |  EM  |  F1  |
|-------------------|------|------|
| closed_book_llm   |  0.0 |  0.0 |
| hybrid_rerank_llm | 23.1 | 30.7 |
| MINE (ours_llm)   | 34.0 | 41.4 |
| oracle_llm        | 65.7 | 74.4 |

EM +10.9 points, 95% CI [+6.9, +15.4], Holm-adjusted p = 0.0012.
F1 +10.7 points, 95% CI [+6.6, +15.1], Holm-adjusted p = 0.0012.

closed_book_llm scores 0.0 and abstains on every question - that is the
leakage control, proving the model is not answering from memory.

Statistics: paired permutation tests, 10,000 resamples, Holm-Bonferroni
corrected across the metric family. Say so in small print.

### By hop count, AllDocs, all 500 questions

This is where an iterative retriever should separate from single-shot, and it
does - but it also shows where mine breaks. Chart this.

| System        | 2-hop | 3-hop | 4-hop |
|---------------|-------|-------|-------|
| bm25          | 26.0  | 12.4  |  1.2  |
| hybrid_rerank | 46.9  | 25.5  |  1.2  |
| MINE          | 74.0  | 37.9  |  4.7  |

### LIMITATION - give this real space, do not bury it

On 4-hop questions AllDocs is 4.7%. Partial coverage is 41%, so it finds some
of the chain and rarely all of it. Three consecutive bridges means per-hop
error compounds.

I tried a beam-diversity fix. It made held-out results WORSE: AllDocs -3.1
points, p = 0.030. So I reverted it and documented the negative result
instead of keeping a change I could not defend.

Present this as a section with a heading, in the normal body text, same
visual weight as the wins. Not a footnote, not a collapsed accordion, not
greyed out. A reader should not be able to miss it.

## DESIGN DIRECTION

- The centrepiece of page 1 is one diagram: standard RAG on the left failing
  on this question, my loop on the right succeeding, the same question
  entering both. If the diagram works, the page works.
- Animate or step through the loop if you can - round 1, entity extracted,
  round 2 - so the bridge is something the reader watches happen rather than
  reads about. A step control beats autoplay.
- Restrained and technical. This is a research write-up, not a product
  launch. No marketing superlatives, no "revolutionary", no emoji, no
  gradient hero banner.
- Charts over tables where a chart reads faster, but keep the exact numbers
  visible as labels - a reader of this page wants the digits.
- Must work on a phone and must look right in both light and dark mode.
- Title it something plain and descriptive. Do not put my name on it.
