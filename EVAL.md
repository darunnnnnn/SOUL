# EVAL.md -- Evaluation of the Context Layer

> **This file is generated.** Every number below is recomputed from the
> per-question records in `eval/raw_outputs/` by
> `python -m ctxlayer.cli report --setting unified --write-md`. Do not edit it by hand.

## 1. Benchmark, version and subset

**Benchmark: MuSiQue-Answerable v1.0, validation split** (Trivedi, Balasubramanian, Khot & Sabharwal, *MuSiQue: Multihop Questions via Single-hop Question Composition*, TACL 2022).

| Property | Value |
|---|---|
| Source file | `musique_ans_v1.0_dev.jsonl` |
| Source sha256 | `15fa63794d18a94ce12411aca6e2327e...` |
| Questions in full dev split | 2417 |
| Excluded: gold evidence in a single document | 54 |
| Eligible (>= 2 distinct gold documents) | 2363 |
| **Evaluated subset** | **500** |
| Selection seed | `20251105` |
| Stratified by hop type | True |
| Knowledge base | 6793 passages / 5848 documents |

### Why MuSiQue

Three properties make it the right benchmark for a *context layer* rather than for a reader:

1. **It genuinely requires multiple documents.** MuSiQue is composed bottom-up from single-hop questions and then *filtered against single-hop shortcut models*. HotpotQA, the obvious alternative, is answerable from a single paragraph far more often than its labels imply, so a gain there can be a reading gain in disguise.
2. **Paragraph-level supporting labels.** Support recall and multi-document coverage are measured directly against gold labels, not via a proxy.
3. **It has 3- and 4-hop questions**, so an iterative retriever's hop budget is tested rather than assumed.

### Subset selection (reproducible)

Selection is a pure function of (source file, subset size, seed=`20251105`, `min_gold_docs`, stratification) and is re-derived from scratch on every run -- nothing is cached, so the subset cannot silently drift. Two filters apply:

- **`min_gold_docs = 2`** drops the 54 dev questions whose gold supporting paragraphs all come from **one** document. They are real multi-hop questions, but their evidence lives in a single source, so counting them would let a single-document answer score as a multi-document success -- which Hard Requirement #2 forbids. Verify with `python -m ctxlayer.cli diagnose corpus`.
- **Hop-type stratification** with largest-remainder apportionment, so the 2-hop majority does not drown out 3- and 4-hop questions.

Subset composition by hop type: `2hop` 254, `3hop1` 120, `3hop2` 41, `4hop1` 52, `4hop2` 13, `4hop3` 20.

### Retrieval settings

The same ingest and the same indexes serve two settings, selected per query, so the two sets of numbers are guaranteed to come from one index:

| Setting | Candidate space | Why |
|---|---|---|
| `unified` *(headline)* | all 6793 passages of the subset | A real knowledge base. Distractors from every other question are live, so retrieval is genuinely open-domain. |
| `pool` | each question's own 20-paragraph pool | The standard MuSiQue distractor setting; keeps our numbers legible against published work. |

## 2. Metrics

| Metric | Definition | Why it is here |
|---|---|---|
| `AllDocs` (`ctx_all_docs`) | 1 if **every** gold *document* is represented in the assembled context | **The Hard-Requirement-#2 metric.** Defined over distinct documents, so stacking several paragraphs of one article cannot inflate it. |
| `SupAll` (`ctx_support_all`) | 1 if **every** gold passage is in the context | Predicts answerability. Far harsher than mean recall: on a 4-hop question, recall 0.75 means the chain is broken and the answer is unreachable, which mean recall hides. |
| `SupRec` (`ctx_support_recall`) | fraction of gold passages in the context | The standard MuSiQue support measure; comparable to published numbers. |
| `DocCov` (`ctx_doc_coverage`) | fraction of gold documents represented | Partial-credit companion to `AllDocs`. |
| `EM` / `F1` | official MuSiQue answer Exact Match and token F1, max over answer aliases | The end-to-end outcome. |
| `Words` | words in the assembled context | Budget parity is auditable, not merely asserted. A gain bought with 3x the tokens is not a gain. |
| `#Docs` | distinct documents in the context | Shows whether a system assembles multi-document context at all. |

Citation metrics (`citation_validity`, `citation_precision`, `citation_recall`, `multi_doc_citation`) are reported in section 6.

## 3. Baselines

All baselines share our indexes, our reader, our token budget and our code path -- they are produced by *config*, not by separate implementations, so none can be accidentally crippled.

| System | What it is | Why it is a fair comparison |
|---|---|---|
| `closed_book` | No retrieval at all. | Measures how much of the benchmark is answerable from the reader's parameters. Without this row, a retrieval gain cannot be distinguished from memorisation. |
| `bm25` | BM25F top-k (title + body fields). | The classic sparse baseline, and a strong one on entity-heavy Wikipedia text. |
| `dense` | Dense bi-encoder (bge-small-en-v1.5) top-k. | The standard single-vector embedding baseline. |
| `hybrid` | BM25F + dense, weighted RRF. | A very widely deployed production configuration. |
| `hybrid_rerank` | Hybrid RRF + cross-encoder reranking. | **Our primary comparison point:** the strongest single-shot pipeline in common use. Beating BM25 alone would prove little. |
| `oracle` | Gold supporting paragraphs only. | Not a baseline but a *ceiling*: separates retrieval error from reading error. |
| `ours` | Iterative entity-bridged multi-hop retrieval with graph expansion, cross-encoder reranking and coverage-aware multi-document context assembly. | -- |

## 4. Results

**unified setting, n = 500 questions.** Percentages except `Words` and `#Docs`.

| System | SupAll | AllDocs | SupRec | DocCov | EM | F1 | Words | #Docs |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 12.8 | 17.4 | 50.5 | 53.5 | 11.0 | 18.9 | 746.3 | 8.4 |
| `dense` | 22.8 | 25.0 | 58.1 | 59.4 | 12.0 | 20.4 | 784.5 | 8.0 |
| `hybrid` | 22.6 | 25.6 | 59.1 | 60.7 | 14.0 | 21.3 | 760.5 | 8.1 |
| `hybrid_rerank` | 29.6 | 32.2 | 64.4 | 65.7 | 13.4 | 20.2 | 756.3 | 8.1 |
| `ours` | 46.4 | 50.6 | 72.4 | 75.5 | 19.0 | 26.1 | 790.0 | 8.4 |
| `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 36.4 | 46.2 | 231.2 | 2.6 |

With 95% bootstrap confidence intervals:

| System | SupAll | AllDocs | SupRec | DocCov | EM | F1 | Words | #Docs |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| `bm25` | 12.8 [10.0, 15.8] | 17.4 [14.2, 20.6] | 50.5 [48.3, 52.7] | 53.5 [51.2, 55.8] | 11.0 [8.4, 13.6] | 18.9 [16.1, 21.8] | 746.3 [735.1, 757.1] | 8.4 [8.3, 8.6] |
| `dense` | 22.8 [19.2, 26.4] | 25.0 [21.2, 28.4] | 58.1 [55.7, 60.4] | 59.4 [57.0, 61.8] | 12.0 [9.4, 15.0] | 20.4 [17.4, 23.8] | 784.5 [777.0, 792.3] | 8.0 [7.8, 8.2] |
| `hybrid` | 22.6 [19.2, 26.4] | 25.6 [21.8, 29.4] | 59.1 [56.7, 61.6] | 60.7 [58.2, 63.1] | 14.0 [10.8, 17.2] | 21.3 [18.1, 24.4] | 760.5 [750.1, 771.2] | 8.1 [7.9, 8.3] |
| `hybrid_rerank` | 29.6 [25.4, 33.4] | 32.2 [28.0, 36.0] | 64.4 [62.1, 66.7] | 65.7 [63.4, 68.0] | 13.4 [10.6, 16.4] | 20.2 [17.2, 23.5] | 756.3 [745.9, 766.2] | 8.1 [7.9, 8.3] |
| `ours` | 46.4 [41.8, 51.0] | 50.6 [45.8, 55.2] | 72.4 [69.6, 75.0] | 75.5 [72.9, 77.9] | 19.0 [15.6, 22.6] | 26.1 [22.7, 29.7] | 790.0 [780.7, 799.5] | 8.4 [8.3, 8.6] |
| `oracle` | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 36.4 [32.2, 40.6] | 46.2 [41.8, 50.2] | 231.2 [220.9, 241.5] | 2.6 [2.6, 2.7] |

### Significance

Paired permutation test over per-question scores (10,000 resamples), with Holm-Bonferroni correction across the whole family of comparisons below. Paired because every system sees the same questions; permutation rather than a t-test because these metrics are binary and their differences are nowhere near normal. `**` marks significance at adjusted p < 0.05.

| Metric | vs baseline | Baseline | Ours | Delta | 95% CI | p (raw) | p (Holm) | d |
|---|---|---|---|---|---|---|---|---|
| ctx_support_all | `closed_book` | 0.0 | 46.4 | +46.4 ** | [+42.0, +51.0] | 0.0001 | 0.0030 | 0.93 |
| ctx_support_all | `bm25` | 12.8 | 46.4 | +33.6 ** | [+29.0, +38.2] | 0.0001 | 0.0030 | 0.67 |
| ctx_support_all | `dense` | 22.8 | 46.4 | +23.6 ** | [+19.2, +28.2] | 0.0001 | 0.0030 | 0.47 |
| ctx_support_all | `hybrid` | 22.6 | 46.4 | +23.8 ** | [+19.4, +28.4] | 0.0001 | 0.0030 | 0.48 |
| ctx_support_all | `hybrid_rerank` | 29.6 | 46.4 | +16.8 ** | [+12.6, +21.4] | 0.0001 | 0.0030 | 0.35 |
| ctx_all_docs | `closed_book` | 0.0 | 50.6 | +50.6 ** | [+46.0, +55.2] | 0.0001 | 0.0030 | 1.01 |
| ctx_all_docs | `bm25` | 17.4 | 50.6 | +33.2 ** | [+28.8, +37.8] | 0.0001 | 0.0030 | 0.66 |
| ctx_all_docs | `dense` | 25.0 | 50.6 | +25.6 ** | [+21.0, +30.4] | 0.0001 | 0.0030 | 0.51 |
| ctx_all_docs | `hybrid` | 25.6 | 50.6 | +25.0 ** | [+20.4, +29.6] | 0.0001 | 0.0030 | 0.51 |
| ctx_all_docs | `hybrid_rerank` | 32.2 | 50.6 | +18.4 ** | [+14.2, +22.8] | 0.0001 | 0.0030 | 0.38 |
| ctx_support_recall | `closed_book` | 0.0 | 72.4 | +72.4 ** | [+69.7, +75.1] | 0.0001 | 0.0030 | 2.49 |
| ctx_support_recall | `bm25` | 50.5 | 72.4 | +21.9 ** | [+19.4, +24.4] | 0.0001 | 0.0030 | 0.77 |
| ctx_support_recall | `dense` | 58.1 | 72.4 | +14.3 ** | [+11.9, +16.8] | 0.0001 | 0.0030 | 0.52 |
| ctx_support_recall | `hybrid` | 59.1 | 72.4 | +13.3 ** | [+11.0, +15.6] | 0.0001 | 0.0030 | 0.50 |
| ctx_support_recall | `hybrid_rerank` | 64.4 | 72.4 | +8.0 ** | [+5.7, +10.3] | 0.0001 | 0.0030 | 0.31 |
| ctx_doc_coverage | `closed_book` | 0.0 | 75.5 | +75.5 ** | [+73.0, +78.0] | 0.0001 | 0.0030 | 2.72 |
| ctx_doc_coverage | `bm25` | 53.5 | 75.5 | +21.9 ** | [+19.5, +24.5] | 0.0001 | 0.0030 | 0.80 |
| ctx_doc_coverage | `dense` | 59.4 | 75.5 | +16.1 ** | [+13.8, +18.5] | 0.0001 | 0.0030 | 0.61 |
| ctx_doc_coverage | `hybrid` | 60.7 | 75.5 | +14.8 ** | [+12.6, +17.1] | 0.0001 | 0.0030 | 0.59 |
| ctx_doc_coverage | `hybrid_rerank` | 65.7 | 75.5 | +9.8 ** | [+7.7, +11.9] | 0.0001 | 0.0030 | 0.41 |
| answer_em | `closed_book` | 0.0 | 19.0 | +19.0 ** | [+15.6, +22.6] | 0.0001 | 0.0030 | 0.48 |
| answer_em | `bm25` | 11.0 | 19.0 | +8.0 ** | [+4.6, +11.8] | 0.0002 | 0.0030 | 0.20 |
| answer_em | `dense` | 12.0 | 19.0 | +7.0 ** | [+4.0, +10.4] | 0.0001 | 0.0030 | 0.20 |
| answer_em | `hybrid` | 14.0 | 19.0 | +5.0 ** | [+1.6, +8.4] | 0.0057 | 0.0114 | 0.13 |
| answer_em | `hybrid_rerank` | 13.4 | 19.0 | +5.6 ** | [+3.0, +8.4] | 0.0001 | 0.0030 | 0.18 |
| answer_f1 | `closed_book` | 0.0 | 26.1 | +26.1 ** | [+22.7, +29.6] | 0.0001 | 0.0030 | 0.66 |
| answer_f1 | `bm25` | 18.9 | 26.1 | +7.3 ** | [+3.9, +10.9] | 0.0001 | 0.0030 | 0.18 |
| answer_f1 | `dense` | 20.4 | 26.1 | +5.7 ** | [+2.6, +9.1] | 0.0008 | 0.0030 | 0.16 |
| answer_f1 | `hybrid` | 21.3 | 26.1 | +4.8 ** | [+1.5, +8.4] | 0.0058 | 0.0114 | 0.12 |
| answer_f1 | `hybrid_rerank` | 20.2 | 26.1 | +5.9 ** | [+3.2, +8.7] | 0.0001 | 0.0030 | 0.19 |

Against the strongest single-shot baseline (`hybrid_rerank`):

- **SupAll**: 29.6 -> 46.4 (+16.8 points, 95% CI [+12.6, +21.4], d = 0.35) -- significant (adjusted p = 0.0030).
- **AllDocs**: 32.2 -> 50.6 (+18.4 points, 95% CI [+14.2, +22.8], d = 0.38) -- significant (adjusted p = 0.0030).
- **SupRec**: 64.4 -> 72.4 (+8.0 points, 95% CI [+5.7, +10.3], d = 0.32) -- significant (adjusted p = 0.0030).
- **DocCov**: 65.7 -> 75.5 (+9.8 points, 95% CI [+7.7, +11.9], d = 0.41) -- significant (adjusted p = 0.0030).
- **EM**: 13.4 -> 19.0 (+5.6 points, 95% CI [+3.0, +8.4], d = 0.18) -- significant (adjusted p = 0.0030).
- **F1**: 20.2 -> 26.1 (+5.9 points, 95% CI [+3.2, +8.7], d = 0.19) -- significant (adjusted p = 0.0030).

### Held-out split (no tuning decision used these questions)

Assembly hyperparameters were selected with `scripts/ab_assembly.py` on the first 150 questions by sorted id, so the table above is partly in-sample. Dropping them leaves **n = 350** questions that no tuning decision touched. Same artefacts, same code, only a filter: `python -m ctxlayer.cli report --holdout`.

| System | SupAll | AllDocs | SupRec | DocCov | EM | F1 | Words | #Docs |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 8.6 | 13.7 | 47.3 | 50.6 | 11.4 | 18.8 | 757.2 | 8.3 |
| `dense` | 14.6 | 17.1 | 53.6 | 55.2 | 11.1 | 18.9 | 793.9 | 7.8 |
| `hybrid` | 16.3 | 20.0 | 55.7 | 57.7 | 14.6 | 21.4 | 770.9 | 7.9 |
| `hybrid_rerank` | 21.4 | 24.9 | 60.9 | 62.6 | 14.3 | 20.2 | 772.4 | 7.9 |
| `ours` | 34.0 | 38.9 | 66.4 | 70.3 | 16.9 | 23.5 | 795.8 | 8.3 |
| `oracle` | 100.0 | 100.0 | 100.0 | 100.0 | 37.1 | 46.9 | 260.0 | 2.9 |

Against `hybrid_rerank` on the held-out split:

- **SupAll**: 21.4 -> 34.0 (+12.6 pts, 95% CI [+8.0, +17.1], adjusted p = 0.0030) -- significant.
- **AllDocs**: 24.9 -> 38.9 (+14.0 pts, 95% CI [+9.4, +18.9], adjusted p = 0.0030) -- significant.
- **EM**: 14.3 -> 16.9 (+2.6 pts, 95% CI [-0.6, +5.7], adjusted p = 0.3576) -- **not** significant.

### Budget parity

A context-quality gain is only meaningful at equal cost. Context size per system:

| System | Mean context words | Mean distinct documents | Median latency (ms) | Mean queries issued |
|---|---|---|---|---|
| `closed_book` | 0 | 0.00 | 0 | 0.0 |
| `bm25` | 746 | 8.43 | 0 | 1.0 |
| `dense` | 784 | 7.99 | 23 | 1.0 |
| `hybrid` | 760 | 8.08 | 1 | 1.0 |
| `hybrid_rerank` | 756 | 8.12 | 2 | 1.0 |
| `ours` | 790 | 8.43 | 153 | 7.9 |
| `oracle` | 231 | 2.65 | 0 | 0.0 |

The latency column above is measured with the cross-encoder and query-embedding caches **warm**, since those caches are what make a 22-ablation sweep affordable. That is the amortised cost inside a sweep, not the cost of a fresh user query, so cold-cache latency is reported separately.

`ours` uses **790** context words against **756** for `hybrid_rerank` (+4%). The gain is therefore not bought with a larger prompt. The cost `ours` *does* pay is retrieval compute: more queries per question, shown above.

### Breakdown by hop count

This is where an iterative context layer should and does separate from single-shot retrieval: the deeper the chain, the more of it is lexically invisible to the original question.

**AllDocs (`ctx_all_docs`) by hop count**

| System | 2hop | 3hop | 4hop | all |
|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 26.0 | 12.4 | 1.2 | 17.4 |
| `dense` | 38.6 | 16.8 | 0.0 | 25.0 |
| `hybrid` | 37.8 | 18.6 | 2.4 | 25.6 |
| `hybrid_rerank` | 46.9 | 25.5 | 1.2 | 32.2 |
| `ours` | 74.0 | 37.9 | 4.7 | 50.6 |
| `oracle` | 100.0 | 100.0 | 100.0 | 100.0 |

**SupAll (`ctx_support_all`) by hop count**

| System | 2hop | 3hop | 4hop | all |
|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 22.4 | 4.3 | 0.0 | 12.8 |
| `dense` | 37.0 | 12.4 | 0.0 | 22.8 |
| `hybrid` | 35.4 | 13.7 | 1.2 | 22.6 |
| `hybrid_rerank` | 46.1 | 19.3 | 0.0 | 29.6 |
| `ours` | 70.9 | 30.4 | 3.5 | 46.4 |
| `oracle` | 100.0 | 100.0 | 100.0 | 100.0 |

**EM (`answer_em`) by hop count**

| System | 2hop | 3hop | 4hop | all |
|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 8.7 | 18.0 | 4.7 | 11.0 |
| `dense` | 12.6 | 17.4 | 0.0 | 12.0 |
| `hybrid` | 11.4 | 23.0 | 4.7 | 14.0 |
| `hybrid_rerank` | 11.4 | 23.0 | 1.2 | 13.4 |
| `ours` | 20.1 | 26.1 | 2.4 | 19.0 |
| `oracle` | 37.4 | 43.5 | 20.0 | 36.4 |

### Synthesis with an LLM reader

The table above fixes a local extractive reader (`deepset/roberta-base-squad2`), which selects one span from one passage. It cannot combine facts across passages, and its `oracle` row shows the cost: handed perfect documents it still scores only 36.4 EM. A retrieval gain therefore has little room to appear in `EM`/`F1`, and on held-out questions it did not reach significance (EM +2.6, Holm p=0.36).

This table replaces **only** the reader, with `gpt-4.1-mini-2025-04-14` at temperature 0.0 -- a dated snapshot, so the numbers name the model that produced them. Retrieval, assembly, budget, questions and metrics are unchanged. Four systems are reported rather than one, because a row for our system alone would support no claim.

| System | SupAll | AllDocs | SupRec | DocCov | EM | F1 | Words | #Docs |
|---|---|---|---|---|---|---|---|---|
| `closed_book_llm` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `hybrid_rerank_llm` | 29.6 | 32.2 | 64.4 | 65.7 | 26.0 | 34.0 | 756.3 | 8.1 |
| `ours_llm` | 46.4 | 50.6 | 72.4 | 75.5 | 37.2 | 46.5 | 790.0 | 8.4 |
| `oracle_llm` | 100.0 | 100.0 | 100.0 | 100.0 | 62.8 | 73.7 | 231.2 | 2.6 |

| Metric | vs baseline | Baseline | Ours | Delta | 95% CI | p (raw) | p (Holm) | d |
|---|---|---|---|---|---|---|---|---|
| ctx_support_all | `closed_book_llm` | 0.0 | 46.4 | +46.4 ** | [+42.0, +51.0] | 0.0001 | 0.0012 | 0.93 |
| ctx_support_all | `hybrid_rerank_llm` | 29.6 | 46.4 | +16.8 ** | [+12.6, +21.4] | 0.0001 | 0.0012 | 0.35 |
| ctx_all_docs | `closed_book_llm` | 0.0 | 50.6 | +50.6 ** | [+46.0, +55.2] | 0.0001 | 0.0012 | 1.01 |
| ctx_all_docs | `hybrid_rerank_llm` | 32.2 | 50.6 | +18.4 ** | [+14.2, +22.8] | 0.0001 | 0.0012 | 0.38 |
| ctx_support_recall | `closed_book_llm` | 0.0 | 72.4 | +72.4 ** | [+69.7, +75.1] | 0.0001 | 0.0012 | 2.49 |
| ctx_support_recall | `hybrid_rerank_llm` | 64.4 | 72.4 | +8.0 ** | [+5.7, +10.3] | 0.0001 | 0.0012 | 0.31 |
| ctx_doc_coverage | `closed_book_llm` | 0.0 | 75.5 | +75.5 ** | [+73.0, +78.0] | 0.0001 | 0.0012 | 2.72 |
| ctx_doc_coverage | `hybrid_rerank_llm` | 65.7 | 75.5 | +9.8 ** | [+7.7, +11.9] | 0.0001 | 0.0012 | 0.41 |
| answer_em | `closed_book_llm` | 0.0 | 37.2 | +37.2 ** | [+33.0, +41.6] | 0.0001 | 0.0012 | 0.77 |
| answer_em | `hybrid_rerank_llm` | 26.0 | 37.2 | +11.2 ** | [+7.4, +15.2] | 0.0001 | 0.0012 | 0.26 |
| answer_f1 | `closed_book_llm` | 0.0 | 46.5 | +46.5 ** | [+42.3, +50.6] | 0.0001 | 0.0012 | 1.01 |
| answer_f1 | `hybrid_rerank_llm` | 34.0 | 46.5 | +12.5 ** | [+8.9, +16.5] | 0.0001 | 0.0012 | 0.29 |

Held out (n = 350, the same filter as above):

| System | SupAll | AllDocs | SupRec | DocCov | EM | F1 | Words | #Docs |
|---|---|---|---|---|---|---|---|---|
| `closed_book_llm` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `hybrid_rerank_llm` | 21.4 | 24.9 | 60.9 | 62.6 | 23.1 | 30.7 | 772.4 | 7.9 |
| `ours_llm` | 34.0 | 38.9 | 66.4 | 70.3 | 34.0 | 41.4 | 795.8 | 8.3 |
| `oracle_llm` | 100.0 | 100.0 | 100.0 | 100.0 | 65.7 | 74.4 | 260.0 | 2.9 |

| Metric | vs baseline | Baseline | Ours | Delta | 95% CI | p (raw) | p (Holm) | d |
|---|---|---|---|---|---|---|---|---|
| ctx_support_all | `closed_book_llm` | 0.0 | 34.0 | +34.0 ** | [+29.1, +38.9] | 0.0001 | 0.0012 | 0.72 |
| ctx_support_all | `hybrid_rerank_llm` | 21.4 | 34.0 | +12.6 ** | [+8.0, +17.1] | 0.0001 | 0.0012 | 0.28 |
| ctx_all_docs | `closed_book_llm` | 0.0 | 38.9 | +38.9 ** | [+33.7, +43.7] | 0.0001 | 0.0012 | 0.80 |
| ctx_all_docs | `hybrid_rerank_llm` | 24.9 | 38.9 | +14.0 ** | [+9.4, +18.9] | 0.0001 | 0.0012 | 0.31 |
| ctx_support_recall | `closed_book_llm` | 0.0 | 66.4 | +66.4 ** | [+63.5, +69.5] | 0.0001 | 0.0012 | 2.32 |
| ctx_support_recall | `hybrid_rerank_llm` | 60.9 | 66.4 | +5.5 ** | [+3.2, +8.0] | 0.0001 | 0.0012 | 0.24 |
| ctx_doc_coverage | `closed_book_llm` | 0.0 | 70.3 | +70.3 ** | [+67.4, +73.1] | 0.0001 | 0.0012 | 2.56 |
| ctx_doc_coverage | `hybrid_rerank_llm` | 62.6 | 70.3 | +7.7 ** | [+5.4, +10.0] | 0.0001 | 0.0012 | 0.35 |
| answer_em | `closed_book_llm` | 0.0 | 34.0 | +34.0 ** | [+29.1, +39.1] | 0.0001 | 0.0012 | 0.72 |
| answer_em | `hybrid_rerank_llm` | 23.1 | 34.0 | +10.9 ** | [+6.9, +15.4] | 0.0001 | 0.0012 | 0.26 |
| answer_f1 | `closed_book_llm` | 0.0 | 41.4 | +41.4 ** | [+36.6, +46.0] | 0.0001 | 0.0012 | 0.90 |
| answer_f1 | `hybrid_rerank_llm` | 30.7 | 41.4 | +10.7 ** | [+6.6, +15.1] | 0.0001 | 0.0012 | 0.26 |

Three things make this table interpretable rather than a demonstration that large models know trivia:

* **`hybrid_rerank_llm` holds the reader fixed.** The strongest single-shot baseline, same budget, same model, same prompt, so the gap to `ours_llm` is attributable to retrieval, not generation.

* **`closed_book_llm` is the leakage control.** Same model and prompt, no context at all. It scores 0.0 and abstains on every question, so nothing here is carried by parametric memory. Stated precisely: that shows the grounding instruction is obeyed, not that the model is ignorant of these facts -- the prompt directs abstention when the context is insufficient, and it complies.

* **`oracle_llm` bounds synthesis error** as `oracle` bounds reading error in the headline table.

Every citation here is checked by the verifier the extractive path uses: the anchor must resolve to a real location and its text must still match the source bytes. A model-emitted citation is not trusted because a model emitted it.

## 5. Ablations

Each ablation is the `ours` config with **exactly one field changed** (`SystemConfig.replace` raises on an unknown field, and `tests/test_pipeline.py::test_ablation_is_a_single_field_delta` asserts the delta is single-field). Run on **n = 150** questions -- the development split, so that any design decision informed by this table leaves the held-out split in section 4 untouched. Sorted by damage to `AllDocs`.

| Ablation | What changed | AllDocs | d(AllDocs) | SupAll | EM | Words |
|---|---|---|---|---|---|---|
| **`ours` (full)** | -- | **78.0** | -- | **75.3** | **24.0** | 776 |
| `no_iterative` | `iterative.enabled=False` | 48.0 | -30.0 | 47.3 | 11.3 | 716 |
| `no_reach_gate` | `iterative.min_entity_df=1` | 61.3 | -16.7 | 60.7 | 16.0 | 738 |
| `no_rerank` | `rerank.enabled=False` | 63.3 | -14.7 | 58.7 | 18.0 | 767 |
| `no_query_rewrite` | `iterative.use_query_rewrite=False` | 69.3 | -8.7 | 67.3 | 22.0 | 779 |
| `no_graph` | `iterative.use_graph=False` | 73.3 | -4.7 | 71.3 | 23.3 | 772 |
| `no_adaptive_stop` | `iterative.gain_threshold=0.0` | 75.3 | -2.7 | 74.0 | 24.0 | 791 |
| `no_assembly` | `assemble.enabled=False` | 78.0 | +0.0 | 74.7 | 24.0 | 763 |
| `no_barren_stop` | `iterative.max_barren_rounds=99` | 78.0 | +0.0 | 75.3 | 24.0 | 776 |
| `no_frontier_floor` | `iterative.min_frontier_rel=0.0` | 78.0 | +0.0 | 75.3 | 24.0 | 775 |

The most damaging single change is **`no_iterative`** (-30.0 points of `AllDocs`), which is the empirical argument for that component existing.

### How much gold evidence is reachable only by a bridge

Every retrieved passage records the hop at which it entered the pool, so the raw outputs answer the central claim directly. A passage that entered at hop >= 1 was never proposed by the original question, so **no single-shot retriever can reach it, however good its reranker** -- reranking only reorders what recall proposed.

| Hop at which gold evidence was found | Count | Share |
|---|---|---|
| round 0 (single-shot reachable) | 857 | 78.1% |
| hop 1 (bridged) | 221 | 20.1% |
| hop 2 (bridged) | 19 | 1.7% |

**21.9%** of the gold evidence we retrieve is bridge-only, and 228/500 questions (46%) depend on at least one bridged passage. Mean queries issued per question: 7.9.

That share is worth stating carefully, because it is smaller than the headline gain might suggest: most gold evidence *is* reachable from the question, and round 0 finds it. The bridged minority is disproportionately decisive only because `AllDocs` and `SupAll` are **all-or-nothing per question** -- one missing document fails the whole question, so recovering a single otherwise-unreachable passage can flip a question outright. The gain is therefore leverage on a minority of evidence, not a uniform lift, and the `no_iterative` ablation separates that contribution from the assembly component.

Adaptive stop, by reason: `gain_below_threshold` 439, `no_frontier` 6, `ran_full_budget` 55.

Reproduce: `python scripts/analyze_traces.py unified ours`.

## 6. Provenance and grounding

Citations are checked at three levels (see `src/ctxlayer/synth/cite.py`): **L1 resolvable** -- the anchor id exists and the text at its character offsets is byte-identical to the live source (mechanical, no threshold); **L2 supported** -- the claim's content words are present in the cited sentence, with numbers and dates required to match exactly; **L3 repair** -- a claim supported by a *different* anchor in the context has its pointer corrected, and the repair is recorded separately rather than folded into precision.

| System | Citation validity (L1) | Citation precision | Citation recall | Cited >= 2 documents |
|---|---|---|---|---|
| `closed_book` | 0.0 | 0.0 | 0.0 | 0.0 |
| `bm25` | 99.8 | 49.8 | 22.2 | 0.0 |
| `dense` | 99.4 | 51.8 | 23.3 | 0.0 |
| `hybrid` | 100.0 | 49.4 | 22.1 | 0.0 |
| `hybrid_rerank` | 100.0 | 54.2 | 23.8 | 0.0 |
| `ours` | 100.0 | 66.5 | 43.5 | 55.6 |
| `oracle` | 100.0 | 100.0 | 40.6 | 0.0 |

`citation_validity` is 100 by construction for the extractive reader and we report it anyway, because it is the mechanical check that the provenance claim rests on: an id that does not resolve, or whose offsets no longer match the source, is a fabrication and is counted as one. Exhaustive verification over the whole knowledge base: `python -m ctxlayer.cli diagnose anchors`.

## 7. Failure analysis

Outcome decomposition for `ours` over all 500 questions:

| Outcome | Count | Share |
|---|---|---|
| retrieval miss: at least one gold DOCUMENT absent | 247 | 49.4% |
| reading miss: complete evidence present, answer wrong | 161 | 32.2% |
| answered correctly with complete evidence | 71 | 14.2% |
| retrieval miss: right documents, wrong passage | 21 | 4.2% |

Reproduce, with examples: `python -m ctxlayer.cli diagnose failures`.

### reading miss: complete evidence present, answer wrong

- **[2hop]** Where in Zhejiang is the city where Protestants are especially notable?
  - gold answer: `Yongjia County`
  - predicted: `Wenzhou`
  - gold documents: Zhejiang, Sanjiang Church
  - context documents: Zhejiang, Estonia, Sanjiang Church, Protestantism, 2004 Summer Olympics, One Thrilling Night
  - bridge entities followed: wenzhou, protestantism, sanjiang church, zhejiang province, nordic, clarke
- **[2hop]** Who did the creator of Derech Mitzvosecha follow?
  - gold answer: `Dovber Schneuri`
  - predicted: `Rabbi Menachem Mendel Schneersohn`
  - gold documents: Chaya Mushka Schneersohn, Derech Mitzvosecha
  - context documents: Derech Mitzvosecha, Chaya Mushka Schneersohn, The Human Centipede (First Sequence), Orthodox Judaism, Jack Cohen (rabbi), David Wallechinsky
  - bridge entities followed: chabad, chabad hasidic, menachem, menachem mendel, chaya mushka schneersohn, menachem mendel schneersohn
- **[2hop]** Who is the child of Chiang Hsiao-wen's father?
  - gold answer: `Chiang Hsiao-wu`
  - predicted: `President Chiang Kai-shek`
  - gold documents: Chiang Hsiao-wen, Chiang Hsiao-wu
  - context documents: Chiang Hsiao-wen, Chiang Hsiao-wu, Chiang Hsiao-chang, Basketball Wives LA, Rabbit Ears Productions, Focus (Chico Freeman album)
  - bridge entities followed: chiang hsiao wu, chiang hsiao chang, john chiang, winston chang, lina medina, chico

### retrieval miss: at least one gold DOCUMENT absent

- **[2hop]** How many people work in the university that is related with Mark F. Green?
  - gold answer: `11,900`
  - predicted: `Oral Roberts University`
  - gold documents: Mark F. Green, Oklahoma City
  - context documents: Mark F. Green, Oklahoma, Mark McCullough, John Ellis Martineau, The Green (Dartmouth College), Monique Mehta
  - bridge entities followed: oklahoma u s, college of law, eastern district, juri doctorate
- **[2hop]** In which county is the city of KABG located?
  - gold answer: `Santa Barbara County`
  - predicted: `Bushehr County`
  - gold documents: Los Alamos, California, KABG
  - context documents: Kharg, Iran, KABG, Ban Pha Tang, Kharg Island, Muang Kasi, Kalldrun
  - bridge entities followed: bushehr, bushehr province, kharg, kharg district, klaipeda, klaipeda county
- **[2hop]** Who plays the person with the same name as he who was caught by surprise in Once Upon a Time?
  - gold answer: `Liam Thomas Garrigan`
  - predicted: `Josh Dallas`
  - gold documents: John, King of England, Liam Garrigan
  - context documents: List of Once Upon a Time characters, Once Upon a Time in China III, Michael Raymond-James, Julie Goodwin, Beverley Elliott, Once Upon a Time... Planet Earth
  - bridge entities followed: goodwin, list of once, list of once upon, prince charming

### retrieval miss: right documents, wrong passage

- **[2hop]** What was the population of the founder of New Amsterdam before influx of Huguenot refugees?
  - gold answer: `2 million`
  - predicted: `2 million`
  - gold documents: New York City, Huguenots
  - context documents: Huguenots, Jakarta, New York City, Dutch Republic, Flatbush, Brooklyn, Utrecht sodomy trials
  - bridge entities followed: dutch republic, dutch, anglo dutch war, second anglo dutch
- **[2hop]** What percent of Nigerians in 2010 were the same religion as Avicenna?
  - gold answer: `48.8 percent`
  - predicted: `40%`
  - gold documents: Nigeria, Avicenna
  - context documents: Nigeria, International rankings of India, Avicenna, The World's Billionaires, Jews, Agriculture in Nigeria
  - bridge entities followed: factbook, world factbook, the world, additionally
- **[2hop]** What specific part of the document that is the highest authority in Protestantism for morals reference Mary?
  - gold answer: `Genesis 3:15`
  - predicted: `sola scriptura`
  - gold documents: Protestantism, Immaculate Conception
  - context documents: Protestantism, Canon law, Catholic Church in England and Wales, Mary, mother of Jesus, Jude the Apostle, Immaculate Conception
  - bridge entities followed: roman catholic church, bible, mariology, catholic church

## 8. Reproducing these numbers

```bash
pip install -e .
python -m ctxlayer.cli reproduce            # end-to-end, one command
```

Or step by step:

```bash
python -m ctxlayer.cli download             # fetch + verify sha256
python -m ctxlayer.cli ingest  --subset 500 # deterministic subset
python -m ctxlayer.cli index                # BM25F / dense / entities
python -m ctxlayer.cli eval    --setting unified
python -m ctxlayer.cli ablate  --setting unified
python -m ctxlayer.cli report  --setting unified --write-md
```

`report` reads only `eval/raw_outputs/` -- it rebuilds every table above without retrieval, without a model, and offline. It refuses to aggregate runs whose question sets, corpus fingerprints or readers disagree, so a stale artefact produces an error instead of a wrong table.

### Environment that produced these numbers

| | |
|---|---|
| python | `3.11.9` |
| torch | `2.7.0+cpu` |
| transformers | `4.57.6` |
| numpy | `1.26.4` |
| platform | `Windows-10-10.0.26300-SP0` |
| git_commit | `b16cc5a1f290a7f042eafffd54a50ba4783e4edb` |

Results are CPU-deterministic given the same seeds and library versions. Small differences across machines are possible from floating-point reduction order in the encoders; the retrieval metrics are integer-valued set comparisons and are robust to that.

## 9. Limitations

- **Subset, not the full split.** 500 of 2363 eligible questions. Selection is seeded and stratified, and the confidence intervals above quantify the resulting uncertainty, but a full-split run would narrow them.
- **The reader extracts; it does not reason across passages.** In MuSiQue the answer is a span in the final hop's paragraph, so retrieve-then-extract is sound -- but `EM`/`F1` here measure *"did the context layer put the answer-bearing paragraph, and its supporting chain, in front of the reader?"*, not *"can the reader chain facts?"*. The `oracle` row bounds reading error separately. An LLM reader is implemented (`src/ctxlayer/synth/llm.py`) and shares the same citation verifier, but is deliberately not used for the headline numbers: a strong LLM answers many of these questions from parametric memory, which would confound a context-quality measurement with model knowledge. The `closed_book` row quantifies exactly that risk.
- **L2 citation support is lexical + numeric, not entailment.** A natural-language-inference model would judge paraphrase better, at ~1.4 GB and a dependency whose version can silently change the verdict on previously-committed citations. For the extractive reader the claim is a literal substring of its cited sentence, so L1+L2 is exact there; for free-form generation it is an approximation.
- **English Wikipedia prose only.** The proper-noun detector relies on capitalisation, which does not transfer to unsegmented or caseless scripts.

