# SOUL — recording script

9:00 of speech at a normal pace, measured, plus roughly 1:15 of command
runtime on screen. Unedited that lands at about 10:15, which is over the
10:00 ceiling — see Timing at the end. Trimming the dead air while commands
run brings it comfortably inside, and costs nothing.

Every command is real and has been run. Every number is from a committed
artefact. The narration is deliberately terse — the page and the terminal are
doing the explaining, and anything you add on top comes out of your margin.

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

# 1 · Opening — 0:00–0:25

**ON SCREEN** Title slide.

> This is SOUL, my solution to the Sarvam AI "Knowledge Base as a Context
> Layer" assignment.
>
> The goal is a system that retrieves information from multiple documents,
> connects it, and produces grounded answers with citations you can check.
>
> The problem is that evidence is distributed. Retrieving a few relevant
> paragraphs doesn't mean you've collected all the evidence a question needs.

---

# 2 · The problem and the mechanism — 0:25–3:45

**ON SCREEN** Artifact page, Mechanism tab. Step through by hand.

### Frame 1 — the board

*The only frame worth lingering on. Everything else moves briskly.*

> In which city is the university attended by the founder of Microsoft
> located? The answer is Cambridge, and it needs three documents.
>
> Left is standard single-shot retrieval. Right is mine.
>
> "Microsoft" is highlighted — that colour means words the question contains.
> And that's all the question gives you. Not Bill Gates, not Harvard, not
> Cambridge.
>
> The grid is the document store, with no lines between documents — these are
> independent articles.
>
> And on the right, three round-slots. Two are empty. Nothing in the question
> can fill them.

### Frame 2

> Both search, both get the same three documents. They haven't diverged, so
> nothing that follows is down to a better search engine.

### Frame 3

> And look what came back — "The Founder", a 2016 film about a hamburger
> franchise. It matched the word "founder". The retriever worked correctly
> and returned something useless.

### Frame 4

> The left system is done. One query, one of three documents, no answer.
>
> The problem isn't that it was wrong. Nothing tells it two documents are
> missing, so nothing triggers a second attempt.

### Frame 5

> Mine reads what it retrieved. The entities underline in a new colour —
> Redmond, Bill Gates, Paul Allen. Words the question did not contain.

### Frame 6

> So "Bill Gates" lifts out of the document and lands in round two's empty
> query box. Watch where it came from — not the question, but inside a
> document we just read.

### Frame 7

> Round two finds the Bill Gates document, and the evidence grows to two.
>
> And here's the proof. This phrase — "the software company he had founded."
> This document never says the word "Microsoft". So no search of the question
> could ever have reached it.

### Frame 8

> Then the same thing again. We read the new document, and it gives us
> "Harvard University".

### Frame 9 — the one that matters

*Slow down here.*

> And this is the same move, a second time. "Harvard University" lifts out
> and lands in round three.
>
> That's the system. It isn't one extra search — it's a loop. Search, read,
> take a new word, search again.
>
> Each round finds the term that makes the next one possible. "Bill Gates"
> existed only inside document one. "Harvard" only inside document two.

### Frame 10

> Round three finds Harvard, and there's the answer — Cambridge. Three
> rounds, three documents, and a chain the system built by reading. None of
> those links existed in the store.

### Frame 11

> It also has to stop. A gauge measures what another round would add, crosses
> the threshold, and halts. On the real benchmark that fires on 88% of
> questions.

### Frame 12

> Cambridge, with three citations — document, passage, sentence. The byte
> offsets are re-resolved against the source and compared character by
> character, and mismatches are rejected. Across the corpus that's 24,064
> anchors with zero failures.
>
> Side by side: one query against three, one document against three.

### The panel below

> One design decision. My first version linked documents by title. I measured
> what that could ever reach — 30%. A mention index over surface forms
> reaches 90%. So I threw the link graph away.

---

# 3 · Architecture — 3:45–4:25

**ON SCREEN** Artifact page, Architecture tab.

> The whole system. Documents split into passages, passages into sentence
> anchors with byte offsets. Three indexes in parallel — BM25F, dense, and
> the mention index. Retrieval fuses lexical and dense and reranks with a
> cross-encoder, and that stage alone is the strongest baseline. This is the
> loop, the only stage that cycles. Then assembly under a word budget, then
> the reader and citation checking.

**Point at the matrix:**

> And this is the part worth looking at. Every baseline is this same pipeline
> with stages switched off. There's no separate code path for my system — one
> class, one config object, and a test that enforces it.

---

# 4 · The knowledge base — 4:25–5:40

**ON SCREEN** Terminal.

```powershell
python scripts/show_corpus.py
```

> That was the illustration. Everything from here is the real benchmark:
> MuSiQue-Answerable, with the source file's sha256 pinned, so if the data
> changes the evaluator refuses to run.

**Point at the funnel:**

> 2,417 questions in the split. I dropped 54, because in those all the gold
> evidence sits inside one document — several chunks of one source isn't
> multi-document evidence, and I'd rather lose 54 easy questions than take
> credit I didn't earn. 500 selected with a fixed seed, stratified by hops.

**Scroll down:**

> 5,848 documents, 6,793 passages, 24,064 anchors. The document is what
> multi-document evidence is counted over, the passage is what retrieval
> ranks, and the anchor is the sentence a citation points at.
>
> And these are the titles — a treaty, a footballer, a fighter aircraft.
> Unrelated articles with no link structure between them.

---

# 5 · Live demonstration — 5:40–7:45

```powershell
python scripts/show_corpus.py --qid 2hop__130668_126089
```

> A real question: who was in charge of the state where Graft-De Rijp is
> located? It needs two of those 5,848 documents.
>
> Same shape as the Microsoft chain, one link shorter. The question says
> "Graft-De Rijp" and never says "North Holland", but Johan Remkes only
> appears in the North Holland document.

```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system hybrid_rerank
```

> The strong baseline — BM25, dense, cross-encoder reranker. One query.
>
> And it retrieved *The Case Against Brooklyn*, a 1958 film noir, on the word
> "graft" — political graft. The answer is Alkmaar, which is wrong, and it
> found one of the two documents.
>
> That's not a weak baseline failing. It's a well-built one hitting a wall it
> can't get past.

```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system ours_llm
```

> Same question, SOUL. Thirteen queries instead of one.
>
> And in that list — "north holland charge state located". That query
> couldn't have come from the question. It was built from an entity inside a
> passage already retrieved.
>
> Answer: Johan Remkes. Two claims, two citations, two documents, both
> anchors verified byte-exact.

---

# 6 · Results — 7:45–8:45

**ON SCREEN** Artifact page, Results tab.

> Two things about the measurement. The first 150 questions are held out —
> every hyperparameter was tuned on those, so these numbers come from the 350
> that no tuning decision touched. And both systems use the same pinned LLM
> reader, so this compares retrieval, not models.

| Metric | Hybrid RAG | SOUL |
|---|---|---|
| AllDocs | 24.9% | 38.9% |
| Exact Match | 23.1% | 34.0% |
| F1 | 30.7% | 41.4% |

> AllDocs scores 1 only if every required document made it into the context —
> no partial credit. That goes from 24.9 to 38.9 percent, a 14-point gain.
> Exact match 23.1 to 34, F1 30.7 to 41.4. All at a Holm-adjusted p of
> 0.0012, from paired permutation tests.
>
> Closed-book scores zero and abstains on everything — that's the control
> showing the model isn't answering from memory. And oracle is gold documents
> handed in directly, so it's a ceiling, not a competitor.

**Point at the hop chart:**

> By hop count you see the win and the problem. 2-hop, 47 to 74 percent.
> 3-hop, 25 to 38. 4-hop, almost nothing.

---

# 7 · Ablations and the limitation — 8:45–9:35

**ON SCREEN** `EVAL.md` §5.

> Which components earn their place? Each ablation changes exactly one field,
> on the 150-question development split, never the held-out set.
>
> Removing iterative retrieval drops AllDocs from 78 to 48 percent. That
> thirty-point gap is the argument for the component existing. The
> reachability gate is next, then reranking. Three of them change nothing,
> and I left those in the table.

**Scroll to §9:**

> And the main weakness. On 4-hop questions AllDocs is 4.7 percent, against
> 74 on 2-hop. Four hops means three consecutive bridges, and error
> compounds.
>
> I tried to fix it by capping bridges per passage. It made the held-out
> results worse — down 3.1 points, p of 0.026. So I reverted it and kept the
> measurement instead of the change.

---

# 8 · Reproducibility and close — 9:35–10:15

**ON SCREEN** GitHub repo, then README.

> The repository has the source, the design and evaluation documents, the raw
> per-question outputs, 164 tests, and the scripts behind every measurement
> here. Every run records the commit, the config fingerprint and the corpus
> hash, and the report refuses to combine runs that disagree.
>
> The extractive evaluation reproduces with one command. The LLM evaluation
> is separate and needs its own API key, and I've documented it that way
> rather than implying everything runs at once.

**ON SCREEN** Title slide.

> So — SOUL discovers connected information across sources, assembles the
> evidence a question needs, and produces grounded answers you can trace.
>
> Thank you for reviewing my work.

---

## Checklist

- [ ] Title slide
- [ ] Artifact page — Mechanism, Results, Architecture tabs
- [ ] `show_corpus.py`
- [ ] `show_corpus.py --qid 2hop__130668_126089`
- [ ] `ask --system hybrid_rerank` — the film noir, wrong answer, 1 of 2
- [ ] `ask --system ours_llm` — 13 queries, the bridge, 2 citations
- [ ] `EVAL.md` §5 ablations and §9 limitations
- [ ] GitHub repo and README
- [ ] `OPENAI_API_KEY` set, all five commands pre-warmed

## Timing

Measured, not estimated:

    speech, at 145 words per minute          9:00
    command runtime on screen (4 commands)  ~1:15
                                            -----
    unedited                                10:15

That is over the ceiling, and the fix is editing rather than cutting content.
Almost all of the 1:15 is you sitting still while a warm command prints. Cut
those waits in post and the recording lands near 9:30.

Two other things that protect the margin:

- Do not improvise. Every sentence beyond what is written comes straight out
  of the margin, and the script is already trimmed close.
- Pre-warm all five commands. Cold starts are ~60s each and would put the
  recording past twelve minutes on their own.

If you still overrun, cut in this order and no further:

1. §3 Architecture — 40s. The Mechanism tab already showed the loop.
2. The design-decision panel at the end of §2 — 20s.
3. §4's title sample — 15s.

**Never cut** frames 6, 7 and 9 (the two bridges and the proof), the live
demo, or the limitation. Those are the submission.

## Three rules

1. **The Microsoft chain explains; the benchmark question demonstrates.**
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
