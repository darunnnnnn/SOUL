# Narrating the artifact page

For the three-tab page: Mechanism (12 frames), Results, Architecture.
Roughly 5 minutes. Use the step control, never autoplay — you want to talk
over each frame, not chase it.

This replaces the whiteboard parts of the video. The terminal demo on a real
benchmark question still happens afterwards, and it is not optional.

---

## MECHANISM TAB

### Frame 1 — the board, before anything moves

*Hold here for twenty seconds. Do not rush it.*

> Here's the setup. One question at the top, feeding two systems side by side.
> On the left, standard single-shot retrieval. On the right, mine.
>
> In the question, "Microsoft" is highlighted. That colour will mean one thing
> for the whole animation: words the question actually contains.
>
> The big box in each track is the document store — every document either
> system can search. They're grey because nothing has been retrieved yet. And
> notice there are no lines drawn between them. These are independent
> articles. Nothing in the store says which documents belong together, so
> every connection has to be discovered from the text.
>
> Now the detail I want you to watch. Above each store there are query slots.
> The left system has one, because it will only ever search once. The right
> system has two — and the second one is empty.
>
> Keep your eye on that empty box. Nothing in the question can fill it. The
> only thing that can fill it is a word taken out of a document the system
> hasn't read yet.
>
> And two counters: searches run, and how many of the two required documents
> have been found.

### Frame 2 — both fire

> Both systems search. The beam is in the question's colour, because that's
> the constraint they share: at this point, both can only look for words the
> question gave them.

### Frame 3 — the same result

> And they retrieve the same three documents. The systems have not diverged
> yet — that matters. Whatever happens next isn't because one has a better
> search engine.

### Frame 4 — the wrong document

*The best frame on the page. Let it breathe.*

> Now look at what came back second. "The Founder", a 2016 film about Ray
> Kroc and a hamburger franchise.
>
> It's here because it matched the word "founder" — you can see the match
> drawn back up to the question. The retriever did its job correctly and
> returned something completely useless.
>
> This is what keyword and semantic matching do when the question's wording
> points somewhere the answer isn't.

### Frame 5 — the left side ends

> And the left system is finished. One query, one of two documents, and it
> stops.
>
> The problem isn't that it was wrong. The problem is that it has no way of
> knowing it was wrong. It has no signal that anything is missing, so there's
> nothing to trigger a second attempt. It's over.
>
> The right side is still running.

### Frame 6 — reading

> The right system reads what it just retrieved, instead of handing it
> straight to a model.
>
> And the entities it finds are underlined in a new colour — Redmond, Bill
> Gates, Paul Allen, Altair 8800.
>
> That colour change is the whole idea. These are words the question did not
> contain. A moment ago the system could only search for "Microsoft" and
> "founder". Now it has names it didn't start with.

### Frame 7 — the bridge

*This is the contribution. Say it slowly.*

> And here's the bridge.
>
> "Bill Gates" lifts out of the document text, travels up, and lands in that
> empty query slot. Watch where it comes from — not from the question, from
> inside a document.
>
> The question greys out for a moment to make the point: this query could not
> have been produced from the question. "Bill Gates" isn't in it.
>
> That single motion is the contribution of the entire system.

### Frame 8 — round two

> A second search, in the new colour. Same retrieval machinery as round one —
> same indexes, same reranker. The only difference is where the query came
> from.

### Frame 9 — the proof

*The second most important frame. This is the one that closes the argument.*

> The second document arrives: Bill Gates, with "Harvard University" in it.
>
> But look at this phrase — "the software company he had founded." This
> document never says the word "Microsoft". Not once.
>
> That's why no single search could have found it. The distinctive word in
> the question is "Microsoft", and it does not appear anywhere in the document
> that holds the answer. You cannot search your way to it. You have to read
> the first document and carry a name out of it.
>
> And that line now drawn between the two documents — that link did not exist
> in the store. The system created it by reading.

### Frame 10 — stopping

> It also has to know when to stop, or it would search forever.
>
> There's a gauge measuring how much a further round would actually add. It
> crosses the threshold, and the loop halts itself.
>
> On the real benchmark that stop fires on about 88% of questions, which
> brings the average down to under two rounds instead of spending the full
> budget on every question.

### Frame 11 — the citation

> The answer: Harvard University.
>
> And underneath it, a citation — document, passage, sentence. If I hover it,
> it draws a line back to the exact sentence it came from.
>
> That isn't a label. The byte offsets get re-resolved against the source
> corpus and compared character by character. If the cited text doesn't match
> the source exactly, the citation is rejected. Across the real corpus that's
> twenty-four thousand anchors, with zero mismatches.

### Frame 12 — side by side

> So: one query against two. One document against two. No answer against the
> right answer with a verified citation.
>
> Same question, same indexes, same reranker. The only difference is that one
> system read what it retrieved and searched again.

### The panel below the fold

> One design decision worth showing.
>
> My first version connected documents by a title link graph — if document A
> contains document B's title, link them. I measured how much of the required
> evidence that could ever reach: 30%.
>
> A mention index over surface forms — every capitalised span, not just exact
> titles — reaches 90%. Same 831 links, measured both ways.
>
> So I threw the link graph away. That's the rejected version, with the number
> that killed it.

**A note if someone asks:** this example is an illustration. Microsoft and
Bill Gates are so strongly associated that a sentence encoder has already
memorised the connection, so on this example single-shot retrieval does not
actually fail. That is why the benchmark uses obscure entities, and why the
live demo uses one. Say this if asked — it is a strength, not a weakness.

---

## RESULTS TAB

> Now the measurements. Everything from here is the real benchmark, not the
> illustration.
>
> MuSiQue-Answerable, sha256-pinned, 500 questions. I dropped 54 questions
> whose evidence all sits inside a single document, because several chunks of
> one source is not multi-document evidence.
>
> And the first 150 questions are held out. Every hyperparameter I tuned was
> tuned on those, so the numbers here are from the 350 that no tuning decision
> ever touched.

**Retrieval chart:**

> AllDocs is the strict metric — it scores 1 only if *every* required document
> made it into the context. Partial credit doesn't count.
>
> The strongest baseline, hybrid retrieval with a cross-encoder reranker, gets
> 24.9%. Mine gets 38.9%. Fourteen points, confidence interval 9.4 to 18.9,
> Holm-adjusted p of 0.0012.
>
> The oracle row is gold documents handed in directly. That's a ceiling, not a
> competitor.

**Answer chart:**

> And the answers, with the same LLM reader on both rows. Retrieval is
> identical between these two — only the reader changes — so this gain is
> about the quality of the context, not the model.
>
> Exact match 23.1 to 34. F1 30.7 to 41.4. Both at p 0.0012.
>
> The closed-book row scores zero and abstains on every question. That's the
> control that shows the model isn't answering from memory.
>
> All of this is paired permutation tests, ten thousand resamples,
> Holm-corrected across the metric family.

**Hop chart:**

> By hop count, you can see both the win and the problem. On 2-hop questions,
> 47% to 74%. On 3-hop, 25 to 38. On 4-hop, almost nothing.

**Limitation section:**

> Which brings me to the main weakness, and I want to be direct about it.
>
> On 4-hop questions AllDocs is 4.7%. Partial coverage is 46%, so it finds
> part of the chain and rarely all of it. Four hops means three consecutive
> bridges, and error at each hop compounds.
>
> I tried to fix it by capping bridges per passage and discarding nested
> candidates. It made the held-out results worse — down 3.1 points, p of
> 0.026. So I reverted it and kept the measurement instead of the change.
>
> That bar going the wrong way is a fix of mine that failed.

---

## ARCHITECTURE TAB

*Keep this brisk. The mechanism tab already did the explaining.*

> Finally, the system as a whole.
>
> Ingestion splits documents into passages and passages into sentence anchors
> with byte offsets. Normalisation is whitespace-only on purpose — anything
> cleverer would be a transformation a citation has to survive.
>
> Three indexes, built in parallel. BM25F, fielded, with the title weighted
> two and a half times the body, because Wikipedia paragraphs are anaphoric
> and often omit their own subject. A dense index. And the mention index.
>
> Retrieval fuses lexical and dense with reciprocal rank fusion, then reranks
> with a cross-encoder. That stage on its own is the strongest baseline.
>
> This is the loop — the only stage that cycles. Read, extract entities, gate
> them for reachability, rank the bridge candidates, rewrite a query, search
> again, and check whether to stop.
>
> Assembly isn't top-k truncation. It maximises relevance, query-term coverage
> and document diversity together under a 900-word budget, with a cap of two
> passages per document — because otherwise the budget fills with one
> document, which is the exact failure mode you're trying to avoid here.
>
> Then the reader, and citation verification.

**The matrix — point at it:**

> And this is the part I'd most like you to look at. Every baseline is this
> same pipeline with stages switched off. BM25 is the lexical path alone.
> Hybrid-rerank is the retrieval stage complete. Mine adds the loop and the
> assembler.
>
> There is no separate code path for my system. One class, one config object,
> and the config decides the behaviour. That's what makes the comparison fair,
> and it's enforced by a test.

---

## Transition into the terminal

> That's the design and the numbers. Now let me run the real thing on a real
> benchmark question.

Then go to the terminal section of the main script.
