# SOUL – Sarvam AI Engineering Take-Home Assignment

**Target duration:** 9–9½ minutes
**Format:** Screen recording with voiceover and live terminal demonstration.

## Before recording

```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
python scripts/show_corpus.py
python scripts/show_corpus.py --qid 2hop__95773_51329
python -m ctxlayer.cli ask --qid 2hop__95773_51329 --system hybrid_rerank
python -m ctxlayer.cli ask --qid 2hop__95773_51329 --system ours_llm
```

Run all four once to warm the caches — a cold start is about 60 seconds of
silence per command, warm is 15–25 seconds. Then clear the terminal.

Open in tabs: the artifact page, `EVAL.md` at section 5, the GitHub repo.

---

## 1. Introduction – 0:00–0:25

**ON SCREEN:** SOUL title slide.

**SAY:**

"Hi, this is SOUL, my solution to the Sarvam AI Knowledge Base as a Context Layer assignment.

First of all, thank you for giving me this opportunity. I enjoyed working on this problem, especially exploring how information from multiple documents can be connected to generate more reliable answers.

Let me walk you through the approach, implementation and evaluation."

---

## 2. Animated explanation – 0:25–1:10

**ON SCREEN:** Play your 35–40 second animated video.

**VOICEOVER:**

"Let's understand the difference between a standard RAG system and SOUL using a simple example.

The question is: in which city is the university attended by the founder of Microsoft located? The answer is Cambridge, but finding it requires connecting three different documents.

On the left is standard single-shot retrieval. It searches using the original question, and it can even return misleading results by matching a word like 'founder'. Once that retrieval finishes, it stops.

On the right, SOUL reads the first document and discovers Bill Gates. It uses that to search again. The second document reveals Harvard University, so it searches again, and the third document gives us the city.

So the difference is this: standard RAG searches with the original question. SOUL searches with what it discovers."

**[After the video:]**

"Now that we've seen the core idea, let's look at how I implemented it."

---

## 3. Architecture – 1:10–1:55

**ON SCREEN:** Open the Architecture tab.

**SAY:**

"This is the overall architecture of SOUL.

During ingestion, documents are divided into passages, and passages are further represented using sentence-level anchors with byte offsets.

We maintain three indexes: BM25F for lexical search, dense embeddings for semantic search, and a mention index for discovering entities across documents.

The initial retrieval combines lexical and dense search, followed by a cross-encoder reranker.

After that comes the iterative retrieval loop, followed by context assembly, answer generation and citation checking.

One important design decision is that the baselines and SOUL use the same underlying pipeline. We change the configuration to switch individual stages on or off, rather than maintaining completely separate implementations."

---

## 4. The knowledge base – 1:55–2:50

**ON SCREEN:** Open the terminal.

Run:

```powershell
python scripts/show_corpus.py
```

**SAY:**

"Now let's look at the actual knowledge base used for evaluation.

I selected MuSiQue-Answerable because it contains questions that require combining evidence from multiple documents.

The original split contains 2,417 questions. I excluded 54 questions where all the supporting evidence was available in just one document. Multiple passages from one document do not satisfy the multi-document requirement.

From the remaining questions, I selected 500 using a fixed random seed and stratified the selection by the number of reasoning hops.

The final corpus contains 5,848 documents, 6,793 passages and 24,064 sentence-level anchors.

These three levels have different purposes.

The document is the unit used to measure multi-document evidence. The passage is what the retrieval system ranks. The anchor is the specific sentence that a citation points to.

The source dataset is also hash-pinned, so changes to the underlying data are detected during evaluation."

---

## 5. Live demonstration – 2:50–5:10

### Part A – Show the benchmark question

**ON SCREEN:** Run:

```powershell
python scripts/show_corpus.py --qid 2hop__95773_51329
```

**SAY:**

"Now let's move from the illustration to an actual question from the benchmark.

'What movie did the person who acted in the comedy film Crime School win his only Oscar for?'

The answer is The African Queen, and it needs two documents. Crime School tells us Humphrey Bogart acted in it. The Bogart article tells us he won one Academy Award, for The African Queen.

But the question never mentions Humphrey Bogart. That name exists only inside the first document, so the system has to discover the connection itself."

### Part B – Run the standard RAG baseline

**ON SCREEN:** Run:

```powershell
python -m ctxlayer.cli ask --qid 2hop__95773_51329 --system hybrid_rerank
```

**SAY:**

"First, the hybrid retrieval baseline — BM25F, dense retrieval and a cross-encoder reranker. A reasonably strong system, but it performs only one retrieval round, using the original question.

It does find Crime School. But look at everything else: Fred Gwynne, The Omen, the 70th Academy Awards, the 84th, The King's Speech, a list of Matt Damon's nominations.

The baseline understood the question correctly. It recognised this was about an Oscar, so it retrieved documents about Oscars. What it never retrieved was Humphrey Bogart, because the question does not contain his name.

One of the two required documents, and it abstains rather than answer.

That is the limitation of a single retrieval round when the question requires discovering something first."

### Part C – Run SOUL

**ON SCREEN:** Run:

```powershell
python -m ctxlayer.cli ask --qid 2hop__95773_51329 --system ours_llm
```

**SAY:**

"Now the same question using SOUL. Nine queries instead of one.

Look at the bridge entities. The first is 'humphrey bogart', and the query built from it: 'humphrey bogart movie person acted comedy film win only'.

The question never contained that name. It was discovered inside the Crime School passage retrieved in the first round, and SOUL used it to construct another search.

Both required documents are now in the context — two out of two. The answer is The African Queen, with citations pointing to the source anchors in both.

That is the complete process: retrieve the first piece of evidence, use it to discover the next, and combine them to answer the question."

---

## 6. Evaluation and results – 5:10–6:10

**ON SCREEN:** Open the Results tab and show the held-out comparison table.

**SAY:**

"Now how we measured the improvement.

I compared SOUL against BM25, dense retrieval, hybrid retrieval, and hybrid with reranking. Both systems use the same pinned LLM reader, so this isolates retrieval rather than comparing language models.

The 500 questions were split into a development set of 150 and a held-out set of 350. Tuning used the development set; everything here is from the held-out set.

SOUL reaches 38.9% AllDocs against 24.9% for the hybrid baseline. AllDocs gives credit only when every required document makes it into the context.

Exact Match improved from 23.1% to 34%, and F1 from 30.7% to 41.4%, with a Holm-adjusted p-value of 0.0012 from paired permutation tests.

So SOUL improves the completeness of the evidence available for answer generation."

---

## 7. Ablation and limitations – 6:10–7:10

**ON SCREEN:** Open EVAL.md, section 5, and show the ablation table.

**SAY:**

"I also wanted to understand which components actually contribute to the improvement.

So I conducted ablation experiments on the 150-question development set, where I removed individual components and measured their impact.

For example, when iterative retrieval was removed, AllDocs dropped from 78% to 48%.

That's a 30-percentage-point reduction, showing how important iterative retrieval is to SOUL.

The reachability gate and reranking also contribute to the overall performance.

However, I also wanted to understand where the system fails."

**ON SCREEN:** Scroll to the hop-count chart and failure analysis.

**SAY:**

"One of the biggest limitations is complex multi-hop retrieval.

On four-hop questions, AllDocs is only 4.7%, compared with 74% on two-hop questions.

This happens because longer reasoning chains require multiple successful retrieval steps, and errors can accumulate at every stage.

I also tested a change that limited the number of bridges per passage. However, it made the held-out results worse by 3.1 percentage points, so I reverted that change.

I wanted the evaluation to reflect what actually improved performance, rather than keeping a change just because it sounded like a good idea."

---

## 8. Reproducibility and engineering – 7:10–8:00

**ON SCREEN:** Show the GitHub repository and README.

**SAY:**

"Finally, the repository contains the complete source code, design documentation, evaluation methodology, raw per-question outputs, 164 tests and the scripts used to generate the measurements.

Every evaluation run records the Git commit, configuration fingerprint and corpus hash. The reporting pipeline checks these before combining results.

The extractive evaluation can be reproduced using the documented one-command workflow.

The separate LLM evaluation requires its own API key and execution, and this is documented in the repository.

The objective was to make the experiments reproducible, not just present a table of results without the underlying evidence."

---

## 9. Conclusion – 8:00–8:20

**ON SCREEN:** Return to the SOUL title slide.

**SAY:**

"To conclude, SOUL is designed to go beyond single-shot document retrieval.

It discovers connected information across independent sources, assembles the evidence required to answer complex questions and produces grounded answers with traceable citations.

The evaluation also helped identify both the components that contribute to its improvement and the limitations that still need to be addressed.

Thank you for taking the time to review my work."

---

## Checklist

- [ ] SOUL title slide
- [ ] Animated video exported, 35–40 seconds
- [ ] Artifact page open — Architecture and Results tabs
- [ ] `show_corpus.py`
- [ ] `show_corpus.py --qid 2hop__95773_51329`
- [ ] `ask --system hybrid_rerank` — the Oscar documents, no Bogart, 1 of 2
- [ ] `ask --system ours_llm` — 9 queries, the `humphrey bogart` bridge, 2 of 2
- [ ] `EVAL.md` section 5 ablations, section 9 limitations
- [ ] GitHub repo and README
- [ ] `OPENAI_API_KEY` set, all four commands pre-warmed

## Timing

Measured by word count at a normal speaking pace, not estimated:

    1. Introduction            0:24
    2. Animated explanation    0:59
    3. Architecture            0:44
    4. The knowledge base      0:58
    5. Live demonstration      2:13
    6. Evaluation and results  0:50
    7. Ablation and limits     1:09
    8. Reproducibility         0:38
    9. Conclusion              0:26
                               ----
       speech                  8:26
       command runtime        ~1:00   (4 commands, warm)
                               ----
       unedited                9:26

Inside the 9.5 target, with almost no slack. Trimming the dead air while
commands print is worth about thirty seconds and costs nothing.

**Section 2 needs a 60-second animation, not 35-40.** The narration there
measures 0:59. If your export is shorter, either hold the final frame or cut
the sentence about misleading results — do not speed up the delivery.

Do not improvise. The margin is seconds, not minutes. And pre-warm all four
commands: a cold start is about a minute each and would put the recording
past twelve minutes on its own.

If you need to claw back time, cut in this order:

1. Section 3, the architecture tab — 44s. Section 2 already showed the loop.
2. Section 4's three-levels explanation — 20s.
3. Section 8's second paragraph — 15s.

Never cut the bridge query in section 5 Part C, or the limitation in
section 7. Those are the submission.

## Three things to hold to

1. **The Microsoft chain explains; the Bogart question demonstrates.** Never
   run the live system on the Microsoft store in `data/demo`. A sentence
   encoder has already memorised that Bill Gates founded Microsoft, so
   single-shot retrieval does not fail there, and the iterative retriever
   does worse — it ranks bridge candidates by inverse document frequency,
   which carries no signal across 45 documents. Both caveats are in the
   docstring of `scripts/build_demo_kb.py`. If a reviewer raises it, say so:
   it is exactly why the benchmark uses obscure entities.
2. **Do not mix the splits.** 23.1 → 34.0 is held-out, n = 350. 78 → 48 is
   the development set, n = 150. Section 6 and section 7 say which each time.
3. **Say the limitation out loud.** Four-hop at 4.7%, and the change that
   made things worse. Volunteering it is worth more than the numbers it
   costs.
