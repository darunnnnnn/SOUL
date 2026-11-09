# SOUL — recording script

Target 9:20. Hard ceiling 10:00. Every command below is real and has been run;
every number comes from a committed artefact.

## Set up before recording

```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
python scripts/show_corpus.py
python scripts/show_corpus.py --qid 2hop__130668_126089
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system hybrid_rerank
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system ours_llm
```

Run all five once to warm the caches — cold start is ~60s of silence, warm is
14–24s. Then clear the terminal.

Open in tabs: the artifact page, `EVAL.md` at §5, the GitHub repo.

---

# 1 · Opening — 0:00–0:35

**ON SCREEN** Title slide.

> Hi — this is SOUL, my solution to the Sarvam AI "Knowledge Base as a Context
> Layer" assignment.
>
> The goal is a system that retrieves information from multiple documents,
> connects it, and produces grounded answers with citations you can check.
>
> The problem is that evidence is distributed. A question can need two, three
> or four different documents — and retrieving a few relevant paragraphs
> doesn't mean you've collected all the evidence the question needs.

---

# 2 · The problem and the mechanism — 0:35–3:30

**ON SCREEN** Artifact page, Mechanism tab. Step through with the arrows.
Never autoplay.

### Frame 1 — the board

*Hold twenty seconds. This frame sets up everything.*

> One question, two systems side by side. On the left, standard single-shot
> retrieval. On the right, mine.
>
> "Microsoft" is highlighted in the question. That colour means one thing
> throughout: words the question actually contains.
>
> The big box in each track is the document store. Everything's grey because
> nothing's been retrieved yet — and notice there are no lines between the
> documents. These are independent articles. Nothing says which ones belong
> together, so every connection has to be found in the text.
>
> Now the detail to watch. Above each store are query slots. The left system
> has one, because it will only ever search once. The right has two, and the
> second is empty.
>
> Keep your eye on that empty box. Nothing in the question can fill it.

### Frame 2 — both fire, same result

> Both search, and both get the same three documents back. They haven't
> diverged yet — so whatever happens next isn't because one has a better
> search engine.

### Frame 3 — the wrong document

> Look at what came back second: "The Founder", a 2016 film about Ray Kroc and
> a hamburger franchise.
>
> It's here because it matched the word "founder". The retriever did its job
> correctly and returned something useless.

### Frame 4 — the left side ends

> And the left system is done. One query, one of two documents.
>
> The problem isn't that it was wrong. It's that it has no way of knowing it
> was wrong — no signal that something's missing, so nothing triggers a second
> attempt. It's over. The right side is still running.

### Frame 5 — reading

> The right system reads what it retrieved instead of handing it straight to a
> model. The entities it finds are underlined in a new colour — Redmond, Bill
> Gates, Paul Allen.
>
> That colour change is the whole idea. These are words the question did not
> contain.

### Frame 6 — the bridge

*Slow down. This is the contribution.*

> Here's the bridge. "Bill Gates" lifts out of the document text and lands in
> that empty query slot.
>
> Watch where it comes from — not the question, but inside a document. The
> question greys out to make the point: this query could not have been
> produced from it.
>
> Then a second search fires, using the same indexes and the same reranker.
> The only difference is where the query came from.

### Frame 7 — the proof

*The frame that closes the argument.*

> The second document arrives, with "Harvard University" in it.
>
> But look at this phrase — "the software company he had founded." This
> document never says the word "Microsoft". Not once.
>
> That's why no single search could reach it. The distinctive word in the
> question doesn't appear in the document holding the answer. You can't search
> your way there — you have to read the first document and carry a name out of
> it.
>
> And that line between the two documents didn't exist in the store. The
> system created it by reading.

### Frame 8 — stopping

> It also has to know when to stop. A gauge measures how much another round
> would add; it crosses the threshold and the loop halts itself.
>
> On the real benchmark that fires on about 88% of questions, keeping the
> average under two rounds instead of spending the full budget every time.

### Frame 9 — the citation

> The answer, and underneath it a citation: document, passage, sentence.
>
> That's not a label. The byte offsets get re-resolved against the source and
> compared character by character — if the cited text doesn't match exactly,
> the citation is rejected. Across the real corpus that's 24,064 anchors, zero
> mismatches.

### Frame 10 — side by side

> One query against two. One document against two. No answer against the right
> answer with a verified citation.
>
> Same question, same indexes, same reranker. The only difference is that one
> system read what it retrieved and searched again.

### The panel below

> One design decision. My first version linked documents by title — if A
> contains B's title, link them. I measured how much evidence that could ever
> reach: 30%. A mention index over surface forms reaches 90%. Same 831 links,
> measured both ways.
>
> So I threw the link graph away. That's the version I rejected, with the
> number that killed it.

---

# 3 · Architecture — 3:30–4:05

**ON SCREEN** Artifact page, Architecture tab.

> The system as a whole. Ingestion splits documents into passages, and
> passages into sentence anchors with byte offsets.
>
> Three indexes in parallel: BM25F with the title weighted two and a half
> times the body, because Wikipedia paragraphs often omit their own subject; a
> dense index; and the mention index.
>
> Retrieval fuses lexical and dense, then reranks with a cross-encoder — that
> stage alone is the strongest baseline. This is the loop, the only stage that
> cycles. Then assembly, which maximises relevance, coverage and document
> diversity together under a 900-word budget, capped at two passages per
> document. Then the reader and citation verification.

**Point at the matrix:**

> And this is the part worth looking at. Every baseline is this same pipeline
> with stages switched off. There's no separate code path for my system — one
> class, one config object, and a test that enforces it. That's what makes the
> comparison fair.

---

# 4 · The knowledge base — 4:05–5:00

**ON SCREEN** Terminal.

```powershell
python scripts/show_corpus.py
```

> That was the illustration. Everything from here is the real benchmark.
>
> I used MuSiQue-Answerable, a published multi-hop QA benchmark, and the
> source file's sha256 is pinned — if the data changes, the evaluator refuses
> to run. I didn't pick a corpus that flatters my system.

**Point at the funnel:**

> 2,417 questions in the split. I dropped 54, because in those all the gold
> evidence sits inside a single document — several chunks of one source isn't
> multi-document evidence, and I'd rather lose 54 easy questions than take
> credit I didn't earn. 2,363 eligible, 500 selected with a fixed seed,
> stratified by hop count.

**Scroll down:**

> 5,848 documents, 6,793 passages, 24,064 anchors. The document is what
> multi-document evidence is counted over, the passage is what retrieval
> ranks, and the anchor is the sentence a citation points at.
>
> And these are the titles — a treaty, a footballer, a fighter aircraft.
> 5,848 unrelated articles with no link structure between them.

---

# 5 · Live demonstration — 5:00–6:35

```powershell
python scripts/show_corpus.py --qid 2hop__130668_126089
```

> A real question: who was in charge of the state where Graft-De Rijp is
> located? It needs two of those 5,848 documents.
>
> Same trap as the Microsoft example — the question says "Graft-De Rijp" and
> never says "North Holland", but Johan Remkes only appears in the North
> Holland document. Unlike Microsoft and Bill Gates, no model has this
> association memorised, which is exactly why the benchmark uses obscure
> entities.

```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system hybrid_rerank
```

> The strong baseline — BM25, dense, and the cross-encoder reranker. One
> query.
>
> And it retrieved *The Case Against Brooklyn*, a 1958 film noir. It matched
> on "graft" — political graft. The answer is Alkmaar, which is wrong, and it
> found one of the two documents.
>
> This isn't a weak baseline failing. It's a well-built one hitting a wall it
> structurally can't get past.

```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system ours_llm
```

> Same question, SOUL. Thirteen queries instead of one.
>
> And in that list — "north holland charge state located". That query couldn't
> have come from the question. It was built from an entity inside a passage
> the system had already retrieved. That's the bridge, on a real question.
>
> Answer: Johan Remkes. Two claims, two citations, two different documents,
> both anchors verified byte-exact against the source.

---

# 6 · Results — 6:35–7:40

**ON SCREEN** Artifact page, Results tab.

> Two things about how this is measured.
>
> The first 150 questions are held out — every hyperparameter I tuned was
> tuned on those, so these numbers are from the 350 no tuning decision
> touched. And both systems use the same pinned LLM reader, so the comparison
> is about retrieval quality, not about swapping models.

| Metric | Hybrid RAG | SOUL |
|---|---|---|
| AllDocs | 24.9% | 38.9% |
| Exact Match | 23.1% | 34.0% |
| F1 | 30.7% | 41.4% |

> AllDocs scores 1 only if *every* required document made it into the context
> — no partial credit. That goes from 24.9 to 38.9 percent, a 14-point gain,
> confidence interval 9.4 to 18.9.
>
> Exact match 23.1 to 34. F1 30.7 to 41.4. All at a Holm-adjusted p of 0.0012,
> from paired permutation tests with ten thousand resamples.
>
> The closed-book row scores zero and abstains on every question — that's the
> control showing the model isn't answering from memory.
>
> And the oracle row is gold documents handed in directly. That's a ceiling,
> not a competitor.

**Point at the hop chart:**

> By hop count you can see both the win and the problem. 2-hop, 47 to 74
> percent. 3-hop, 25 to 38. 4-hop, almost nothing.

---

# 7 · Ablations and the limitation — 7:40–8:40

**ON SCREEN** `EVAL.md` §5.

> Which components earn their place? Each ablation changes exactly one field,
> run on the 150-question development split — never the held-out set.
>
> Removing iterative retrieval drops AllDocs from 78 to 48 percent. That
> thirty-point gap is the empirical argument for the component existing. The
> reachability gate is next at seventeen points, then reranking at fifteen.
> Three of them change nothing at all, and I left those in the table.

**Scroll to §9:**

> And the main weakness, which I want to be direct about.
>
> On 4-hop questions AllDocs is 4.7 percent, against 74 on 2-hop. Partial
> coverage is 46 percent — it finds part of the chain and rarely all of it.
> Four hops means three consecutive bridges, and error compounds.
>
> I tried to fix it by capping bridges per passage. It made the held-out
> results worse — down 3.1 points, p of 0.026. So I reverted it and kept the
> measurement instead of the change.

---

# 8 · Reproducibility and close — 8:40–9:20

**ON SCREEN** GitHub repo, then README.

> The repository has the source, the design and evaluation documents, the raw
> per-question outputs, 164 tests, and the scripts behind every measurement
> I've shown.
>
> The extractive evaluation reproduces with one documented command. The LLM
> evaluation is separate and needs its own API key — I've documented it as a
> separate step rather than implying everything runs at once.
>
> Every run records the commit, the config fingerprint and the corpus hash
> that produced it, and the report refuses to combine runs that disagree on
> any of them.

**ON SCREEN** Title slide.

> So — SOUL discovers connected information across sources, assembles the
> evidence a question needs, and produces grounded answers with traceable
> citations. The evaluation and ablations let me measure what each component
> contributes, and show where it still falls short.
>
> Thank you for reviewing my work.

---

## Checklist

- [ ] Title slide
- [ ] Artifact page open — Mechanism, Results, Architecture tabs
- [ ] `show_corpus.py`
- [ ] `show_corpus.py --qid 2hop__130668_126089`
- [ ] `ask --system hybrid_rerank` — the film noir, wrong answer, 1 of 2
- [ ] `ask --system ours_llm` — 13 queries, the bridge, 2 citations
- [ ] `EVAL.md` §5 ablations and §9 limitations
- [ ] GitHub repo and README
- [ ] `OPENAI_API_KEY` set, all five commands pre-warmed

## If you overrun

Cut in this order, and no further:

1. §3 Architecture — 35s. The Mechanism tab already showed the loop.
2. The design-decision panel at the end of §2 — 20s.
3. §4's title sample — 15s.

**Never cut** frames 6 and 7 (the bridge and the proof), the live demo, or the
limitation. Those are the submission.

## Three rules

1. **The Microsoft example explains; the benchmark question demonstrates.**
   Never run the live system on the Microsoft store in `data/demo`. A sentence
   encoder has already memorised that Bill Gates founded Microsoft, so
   single-shot retrieval does *not* fail there, and the iterative retriever
   does worse — it ranks bridge candidates by inverse document frequency,
   which carries no signal across 45 documents. Both caveats are in the
   docstring of `scripts/build_demo_kb.py`. If a reviewer raises it, say so:
   it's why the benchmark uses obscure entities.
2. **Don't mix the splits.** 23.1 → 34.0 is held-out, n = 350. 78 → 48 is the
   development split, n = 150. Say which is which each time.
3. **Say the limitation out loud.** 4-hop at 4.7%, and the fix that failed.
   Volunteering it is worth more than the numbers it costs.
