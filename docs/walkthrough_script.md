# SOUL — recording script

Target 9–10 minutes. Every command is real and has been run. Every number is
from a committed artefact.

**Before you start**

```powershell
$env:OPENAI_API_KEY = "sk-proj-..."      # else ours_llm silently uses the weak reader
python scripts/show_corpus.py            # warm the caches: run each command
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system hybrid_rerank
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system ours_llm
python scripts/walkthrough.py
```

Cold start is ~60s of silence per command; warm is 14–24s. Have EVAL.md open in
a second tab, scrolled to §4 and §5.

---

## 1 · What SOUL is — 0:00–0:50

**ON SCREEN** Title slide.

> Hi, today I'll be presenting SOUL, my solution to the Sarvam AI "Knowledge
> Base as a Context Layer" assignment.
>
> The objective is a system that retrieves information from multiple
> documents, connects that information, and uses it to generate accurate,
> grounded answers with citations.
>
> The problem I set out to solve is that useful information is distributed.
> A question might need evidence from two, three or four different documents.
>
> A traditional search system retrieves relevant paragraphs — but retrieving a
> few relevant paragraphs doesn't mean you have collected all the evidence the
> question needs.
>
> So SOUL is a context layer that focuses on discovering connected
> information, assembling the required evidence, and generating answers that
> can be traced back to their sources.

---

## 2 · The problem, with a simple example — 0:50–1:35

**ON SCREEN** A slide with these two documents. Not the terminal.

> **Document 1 — Microsoft**
> Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4,
> 1975, to develop and sell BASIC interpreters for the Altair 8800.
>
> **Document 2 — Bill Gates**
> William Henry Gates III enrolled at Harvard University in 1973 and left two
> years later, before completing his degree, to work full time on **the
> software company he had founded**.
>
> **Question** Which university did the founder of Microsoft attend?
> **Answer** Harvard University

> Let's understand the problem with a simple example.
>
> Imagine our knowledge base has two documents. The first tells us Microsoft
> was founded by Bill Gates and Paul Allen. The second tells us Bill Gates
> attended Harvard University.
>
> If someone asks which university the founder of Microsoft attended, we need
> both. The first identifies the founder; the second gives the university.
> Finding only one is not enough — we have to connect them.

**Now point at Document 2 and say the sentence that matters:**

> And look closely at the second document. It never says the word "Microsoft".
> It says "the software company he had founded".
>
> That's what makes this hard. The distinctive word in the question is
> "Microsoft", and that word does not appear in the document that holds the
> answer. So no search of the original question can reach it. You have to read
> the first document, take the name "Bill Gates" out of it, and search again.
>
> This is the central problem SOUL is designed to solve.

*This example is an illustration for explaining the algorithm. The live demo
later uses a real benchmark question.*

---

## 3 · Architecture — 1:35–2:20

**ON SCREEN** Architecture diagram: Ingestion → Indexing → Retrieval →
Iterative retrieval → Context assembly → Synthesis and citations.

> Before the implementation, the overall architecture.
>
> First we ingest and index the source documents. When a question arrives, we
> retrieve relevant passages using several search methods.
>
> The important stage is iterative retrieval, where information discovered in
> one document is used to search for further evidence in others.
>
> Once the evidence is collected, we assemble the context and pass it to the
> reader. Finally we attach citations and verify them, so the answer can be
> traced back to the source.
>
> Now let's look at how each stage actually works.

---

## 4 · The knowledge base — 2:20–3:10

**ON SCREEN**
```powershell
python scripts/show_corpus.py
```

> For the knowledge base and the evaluation I used MuSiQue-Answerable, a
> published multi-hop QA benchmark. I chose it because answering its questions
> correctly requires combining information from several documents.
>
> The source file's sha256 is pinned — if the data changes, the evaluator
> refuses to run. I didn't pick a corpus that flatters my system.

**Point at the selection funnel:**

> 2,417 questions in the dev split. I dropped 54 of them, because in those,
> all the gold evidence sits inside a single document. Several chunks of one
> source is not multi-document evidence, and I would rather lose 54 easy
> questions than take credit I didn't earn. That leaves 2,363 eligible, and I
> selected 500 with a fixed seed, stratified by hop count.

**Scroll to section 2:**

> The knowledge base is 5,848 documents, 6,793 passages and 24,064 citable
> anchors. Three levels, each with a job: the document is what
> multi-document evidence is counted over, the passage is what retrieval
> ranks, and the anchor is a single sentence with byte offsets — that's what a
> citation points at.

**Scroll to section 3, the sampled titles:**

> And these are 5,848 unrelated Wikipedia articles — a treaty, a footballer, a
> fighter aircraft. Nothing in the corpus says which documents belong
> together. There is no link graph to walk. Every connection has to be
> discovered from the text.

---

## 5 · How it works — 3:10–6:10

Keep the Document 1 / Document 2 slide visible for Steps 1–3. This is the
longest section and the most important one.

### Step 1 — the question enters

> So what happens when a question is submitted.
>
> SOUL first analyses the question and identifies what has to be found: the
> founder of Microsoft, and then that person's education.
>
> Instead of asking a language model directly, we search the indexed knowledge
> base to collect the supporting evidence first.

### Step 2 — the first retrieval

> In the first stage, SOUL searches for passages related to the question, using
> several methods together.
>
> BM25F finds passages by keyword relevance. Dense retrieval finds them by
> semantic similarity, even when the words differ. We fuse those results and
> then apply a cross-encoder reranker, which scores each passage against the
> question directly and reorders the candidates.
>
> In our example this retrieves the Microsoft passage: founded by Bill Gates
> and Paul Allen. We've now discovered something important — Bill Gates is a
> founder of Microsoft. But we still don't know which university he attended.

### Step 3 — the bridge (the core of the project)

**ON SCREEN** Highlight "Bill Gates" in Document 1, then draw an arrow to a
new search box.

> Now the most important part of SOUL.
>
> Instead of stopping after the first retrieval, we use what that document told
> us to drive another search.
>
> From the retrieved passage we extract a new entity: Bill Gates. That name was
> not in the question — it came out of a document we had just read.
>
> SOUL builds a second query from it and searches again. An entity index maps
> surface forms — every capitalised span and noun phrase, not just exact
> titles — back to the passages that mention them, so a name found in one
> document locates the other documents that discuss it.
>
> Through this, SOUL retrieves the second document: Bill Gates enrolled at
> Harvard University.
>
> We now have two pieces of evidence. The first connects Microsoft to Bill
> Gates; the second gives the university.
>
> This is iterative retrieval — finding new information and using it to
> discover further evidence. It lets SOUL follow connections between documents
> instead of relying on one search.
>
> It also has to know when to stop. There's a relevance floor, so a round
> whose results are too weak doesn't spawn more queries, and a gain estimate
> that halts the loop when another round would add nothing. On the benchmark
> that stop fires on about 88% of questions, which keeps the mean at under two
> rounds instead of running the full budget every time.

### Step 4 — context assembly

> Once retrieval has collected the evidence, we assemble the context.
>
> We don't send the whole knowledge base to the model — that's wasteful and it
> buries the answer in noise. Instead we select the passages that matter and
> pack them into a fixed budget.
>
> Assembly maximises three things at once: relevance, coverage of the question's
> terms, and document diversity — with a cap per document, because ten passages
> from one source is exactly the failure mode we're trying to avoid in a
> multi-hop setting.

### Step 5 — the answer and its citations

> Finally the assembled context goes to the reader, which produces the answer.
>
> But we don't stop at generation. Every factual claim is attached to a source
> location — a specific sentence in a specific passage in a specific document.
>
> Then we verify them. The anchor's byte offsets are re-resolved against the
> source corpus and compared byte for byte. If a cited sentence doesn't match
> the source exactly, the citation is rejected.
>
> We don't just want an answer that sounds right. We want one that can be
> checked against the source material.

---

## 6 · Live demonstration — 6:10–7:00

**ON SCREEN** This is the real system on a real benchmark question.

> That was the illustration. Now the actual system, on a question from the
> benchmark.

**First, show the problem is real:**
```powershell
python scripts/show_corpus.py --qid 2hop__130668_126089
```

> The question: who was in charge of the state where Graft-De Rijp is located?
> Answer: Johan Remkes. It needs two of the 5,848 documents — Graft-De Rijp
> and North Holland.
>
> And the same trap as before: the question says "Graft-De Rijp" and never says
> "North Holland", but Johan Remkes only appears in the North Holland
> document. Unlike Microsoft and Bill Gates, no model has this association
> memorised — which is exactly why the benchmark uses obscure entities.

**Then the baseline:**
```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system hybrid_rerank
```

> This is the strong baseline — BM25 plus dense plus the cross-encoder
> reranker. One query.
>
> And look what it retrieved: *The Case Against Brooklyn*, a 1958 film noir. It
> matched on the word "graft" — political graft. The answer is Alkmaar, which
> is wrong, and it found one of the two documents it needed.
>
> This isn't a weak baseline failing. It's a well-built one hitting a wall it
> structurally cannot get past.

**Then SOUL:**
```powershell
python -m ctxlayer.cli ask "Who was in charge of the state where Graft-De Rijp is located?" --system ours_llm
```

> Same question, SOUL. Thirteen queries instead of one.
>
> And in that list — "north holland charge state located". That query could not
> have been produced from the question. It was built from an entity found
> inside a passage the system had already retrieved. That's the bridge, on a
> real question.
>
> Answer: Johan Remkes. Two claims, two citations, two different documents, and
> both anchors verified as byte-exact against the source.

---

## 7 · Evaluation — 7:00–8:05

**ON SCREEN** EVAL.md §4, the held-out LLM table.

> Building the system is one part of the assignment. Showing it performs better
> than standard approaches is the other.
>
> I evaluated SOUL against BM25, dense retrieval, hybrid retrieval, and hybrid
> with reranking — plus two controls: closed-book, which gets no context at
> all, and an oracle that is handed the gold documents directly as a ceiling.
>
> Two things about how this is measured.
>
> First, the first 150 questions are held out. Any hyperparameter I tuned was
> tuned on those, so the numbers I'm about to give are from the 350 questions
> no tuning decision ever touched.
>
> Second, both rows use the same pinned LLM reader, so the comparison is about
> retrieval quality and not about swapping the model.

| Metric | Hybrid RAG | SOUL |
|---|---|---|
| Exact Match | 23.1% | 34.0% |
| F1 | 30.7% | 41.4% |
| AllDocs | 24.9% | 38.9% |

> Exact match goes from 23.1 to 34 percent. F1 from 30.7 to 41.4.
>
> And AllDocs — which is 1 only if *every* required document made it into the
> context — goes from 24.9 to 38.9 percent.
>
> These are paired permutation tests, ten thousand resamples, Holm-corrected
> across the metric family. The adjusted p-value is 0.0012.
>
> The closed-book control scores zero and abstains on every question, which is
> the check that the model isn't answering from memory rather than from the
> context we gave it.

---

## 8 · Ablations and failure analysis — 8:05–9:00

**ON SCREEN** EVAL.md §5, the ablation table.

> I also wanted to know which components actually earn their place, so each
> ablation changes exactly one field and is run on the 150-question
> development split — never on the held-out set.
>
> Removing iterative retrieval drops AllDocs from 78 to 48 percent. That
> thirty-point gap is the empirical argument for the component existing.
>
> The reachability gate is next, at seventeen points, then reranking at
> fifteen. And three of them change nothing at all, which I've left in the
> table because a component that costs nothing and contributes nothing is
> worth knowing about.

**Scroll to the design measurement:**

> One decision worth showing. My first design connected documents by a title
> link graph — if document A's text contains document B's title, link them. I
> measured how much of the required evidence that could ever reach: 30.4%.
>
> A mention index over surface forms instead reaches 90.3%. Same 831 gold
> links, measured both ways. So I threw the link graph away. I'm showing you
> the version I rejected, with the number that killed it.

**Scroll to §9, Limitations:**

> And the main weakness. On 4-hop questions, AllDocs is 4.7 percent, against 74
> percent on 2-hop. Partial coverage is 46 percent, so it finds part of the
> chain and rarely all of it — a 4-hop question needs three consecutive
> bridges, and per-hop error compounds.
>
> I tried a fix for it: capping bridges per passage and discarding nested
> candidates. It made the held-out results worse — AllDocs down 3.1 points,
> p equals 0.026. So I reverted it and kept the measurement instead of the
> change.

**Optional, if time allows:**
```powershell
python scripts/ab_bridge_diversity.py
```

---

## 9 · Reproducibility — 9:00–9:30

**ON SCREEN** GitHub repository, then README.

> The repository has the source, the design and evaluation documents, the raw
> per-question outputs, the test suite — 164 tests — and the scripts that
> produce every measurement I've shown.
>
> The extractive evaluation reproduces with one documented command. The LLM
> evaluation is separate and needs its own API key, and I've documented it as a
> separate step rather than implying everything runs from one command.
>
> Every run records the commit, the config fingerprint, and the corpus hash
> that produced it, and the report refuses to combine runs that disagree on
> any of them.

---

## 10 · Conclusion — 9:30–9:50

**ON SCREEN** Title slide.

> To conclude. SOUL goes beyond single-shot document retrieval: it discovers
> connected information across sources, assembles the evidence a complex
> question needs, and generates grounded answers with traceable citations.
>
> Through the benchmark evaluation and the ablations I was able to measure what
> each component contributes, and to identify where the system still falls
> short.
>
> Thank you for taking the time to review my work.

---

## Checklist

- [ ] Title slide
- [ ] Two-document slide, with "the software company he had founded" visible
- [ ] Architecture diagram
- [ ] `show_corpus.py` — funnel, three levels, sampled titles
- [ ] `show_corpus.py --qid 2hop__130668_126089` — the two gold documents
- [ ] `ask --system hybrid_rerank` — the film noir, wrong answer, 1 of 2 docs
- [ ] `ask --system ours_llm` — 13 queries, the bridge query, 2 citations
- [ ] EVAL.md §4 held-out LLM table
- [ ] EVAL.md §5 ablation table + the 30.4 / 90.3 measurement
- [ ] EVAL.md §9 limitations
- [ ] GitHub repo and README
- [ ] `OPENAI_API_KEY` set, all commands pre-warmed

## Three rules

1. **The Microsoft example explains; the benchmark question demonstrates.**
   Never run the live system on the Microsoft store. Its documents are in
   `data/demo`, and a sentence encoder has already memorised that Bill Gates
   founded Microsoft — so single-shot retrieval does *not* fail there, and the
   iterative retriever actually does worse, because it ranks bridge candidates
   by inverse document frequency and that carries no signal across 45
   documents. Both caveats are in the docstring of `scripts/build_demo_kb.py`.
2. **Don't mix the splits.** The 23.1 → 34.0 results are held-out, n = 350.
   The 78 → 48 ablation is the development split, n = 150. Say which is which.
3. **Say the limitation out loud.** 4-hop at 4.7%, and the fix that failed.
   Volunteering it is worth more than the numbers it costs.
