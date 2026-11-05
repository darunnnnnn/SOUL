# WALKTHROUGH — 5–10 minutes

A scripted tour of the system, ending with a successful multi-document answer
and a real failure surfaced by the evaluation. Everything here runs on CPU with
no API key.

**To record it, run the single command in §0 and talk through the seven
sections it prints.** The script is ordered so the narration follows naturally.
If you prefer to drive it by hand, §1–§7 below give the individual commands.

---

## 0. The whole walkthrough, one command

```bash
python scripts/walkthrough.py unified
```

Prerequisites (once):

```bash
pip install -e .
python -m ctxlayer.cli download          # ~30 s, verifies sha256
python -m ctxlayer.cli ingest --subset 500
python -m ctxlayer.cli index             # ~8 min first time (dense encoding)
python -m ctxlayer.cli diagnose reachability
python -m ctxlayer.cli eval --setting unified --limit 100   # ~25 min
python -m ctxlayer.cli report --setting unified --write-md
```

The script picks its examples **from the raw evaluation outputs**, choosing a
question where our system retrieved every gold document and the strongest
baseline did not — so the demo is selected by the data, not cherry-picked.

---

## 1. The problem, in one question (≈1 min)

> *"Where was the creator of the Lap Engine educated?"* — answer:
> **University of Glasgow**

The evidence spans two documents:

| hop | document | what it contributes |
|---|---|---|
| 1 | **Lap Engine** | "a beam engine designed by **James Watt**" |
| 2 | **James Watt** | "While working as an instrument maker at the **University of Glasgow**…" |

The question never says "James Watt". So the second gold document shares almost
no vocabulary with the question, and **no single-shot retriever can rank it** —
reranking only reorders what recall proposed, and recall never proposed it.
This is the ceiling the whole design attacks.

```bash
python -m ctxlayer.cli ask --qid 2hop__115515_779396 --system ours
```

Watch the `BRIDGES` and `QUERIES` lines: the system retrieves the *Lap Engine*
paragraph, extracts `james watt` as a bridge entity, and issues a second query
`"jame watt creator educated"` — the bridge entity plus the question's relation
terms. That second query is what reaches the *James Watt* article.

---

## 2. Why a mention index, not a link graph (≈1 min)

```bash
python -m ctxlayer.cli diagnose reachability
```

The intuitive design is a document-link graph keyed on article titles. I built
it first, then measured the only thing that matters for a candidate generator:
*can it even propose the rest of the gold evidence chain?*

```
hop type          title graph    mention index      delta
2hop                    29.1%            89.8%     +60.6
3hop1                   29.6%            91.2%     +61.7
3hop2                   23.2%            93.9%     +70.7
4hop1                   38.5%            92.3%     +53.8
4hop2                   43.6%            92.3%     +48.7
4hop3                   20.0%            76.7%     +56.7
OVERALL                 30.4%            90.3%     +59.8
```

The title graph fails **structurally**, and the diagnostic prints the root
cause: only **22.5 %** of gold bridge answers are themselves article titles, so
a title-keyed graph has no node to traverse to. The bridge entity is a
*mention inside* the next gold paragraph. That measurement is why the component
was rebuilt.

---

## 3. A successful multi-document answer (≈2 min)

The walkthrough prints, for the chosen question:

- the **query plan** — anchor entities (what the question gives), relation terms
  (what it asks), expected answer type;
- the **bridge entities** followed and the queries issued;
- the **assembled context** — passages, distinct document count, word count,
  with gold passages flagged;
- `gold DOCUMENTS in context: 2/2`;
- the **answer**, and each claim with its citation:

```
CLAIM  : "University of Glasgow"   supported=True
  cites b5c220e30ca8::p0::s0
    source   : James Watt, sentence 0, chars 0-125
    verified : True  (text byte-identical to source)
    quote    : "While working as an instrument maker at the University of
                Glasgow, Watt became interested in the technology of steam
                engines."
```

The citation is not a label attached after the fact: the anchor id resolves to a
specific sentence at specific character offsets, and `verified: True` means the
text was re-read from the source and matched byte for byte.

---

## 4. The same question, strongest baseline (≈1 min)

Section 4 reruns the identical question through `hybrid_rerank` — BM25F + dense
+ RRF + cross-encoder reranking, the strongest single-shot pipeline in common
use, sharing our indexes, our reader and our token budget.

It retrieves the first gold document and misses the second. This is the point of
the whole submission: the gap is not a ranking gap that a better reranker
closes, it is a **candidate generation** gap.

---

## 5. A real failure (≈1.5 min)

```bash
python -m ctxlayer.cli diagnose failures
```

Section 5 shows the most common failure, and it is deliberately *not* a
flattering one: the context contains **every** gold document and the fixed
extractive reader still answers wrongly.

That is a reader limitation, not a context-layer success — and it is exactly why
the `oracle` row is reported in EVAL.md, so reading error is never silently
counted as retrieval quality. The honest framing: with a fixed extractive
reader, `EM`/`F1` measure *"did the context layer put the answer-bearing
paragraph and its supporting chain in front of the reader?"*, not *"can the
reader chain facts?"*.

A second, subtler failure worth narrating is visible in §3's context: once a hop
discovers the answer entity (*University of Glasgow*), that entity becomes a
candidate bridge, and expanding it spends budget on passages *about the answer*
(*Beatrice Heuser*, who holds a chair there). The relevance floor and
idf-weighted coverage contain the damage; removing the cause is item 3 in
DESIGN.md §7.

---

## 6. Provenance, checked mechanically (≈1 min)

```bash
python -m ctxlayer.cli diagnose anchors
```

```
anchors checked          : 24064
offset/text mismatches   : 0
passages under-covered   : 0
mean anchors per passage : 3.54

OK: every anchor re-resolves to byte-identical source text.
```

Exhaustive, not sampled. And a fabricated id is rejected rather than accepted:

```python
kb.verify_anchor("deadbeef::p0::s0")   # -> False
```

`tests/test_provenance.py` covers the adversarial cases: fabricated ids,
tampered offsets, and a claim whose *numbers* disagree with its source
("1953" vs "1954") — rejected despite near-identical wording, because
paraphrase-tolerant scoring is most dangerous exactly where numbers are
concerned.

---

## 7. Results (≈1.5 min)

```bash
python -m ctxlayer.cli report --setting unified
```

The headline table and, against the strongest single-shot baseline, the delta
with a 95 % bootstrap confidence interval and a Holm-adjusted p-value from a
paired permutation test.

Two things to point at while narrating:

1. **`AllDocs`** — 1 if *every* gold **document** is in the context. Defined
   over distinct documents, so stacking several paragraphs of one article cannot
   inflate it. This is the Hard-Requirement-#2 metric.
2. **`Words`** — our context is *smaller* than the baseline's. The gain is not
   bought with a bigger prompt; the cost we pay is retrieval compute (more
   queries per question), which is also in the table.

Full numbers, ablations and failure analysis: **[EVAL.md](EVAL.md)**.
Architecture and reasoning: **[DESIGN.md](DESIGN.md)**.
