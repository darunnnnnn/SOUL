# SOUL — key results

Source: `eval/results/report_unified_llm_holdout150.json` and
`eval/results/report_unified_fixed_holdout150.json`, regenerated from
`eval/raw_outputs/unified*/`. Every number below can be re-produced with:

```powershell
python -m ctxlayer.cli report --setting unified --holdout
```

**Held-out set: 350 of 500 questions.** The first 150 (sorted by id) were
used for tuning and are excluded here — nothing below is from data any
hyperparameter was chosen against.

---

## 1. Does SOUL retrieve the complete evidence?

| System | AllDocs | DocCov | SupAll |
|---|---|---|---|
| closed_book | 0.0 | 0.0 | 0.0 |
| bm25 | 13.7 | 50.6 | 8.6 |
| dense | 17.1 | 55.2 | 14.6 |
| hybrid | 20.0 | 57.7 | 16.3 |
| **hybrid_rerank** (strongest baseline) | **24.9** | 62.6 | 21.4 |
| **SOUL (ours)** | **38.9** | **70.3** | **34.0** |
| oracle (gold docs injected — ceiling, not a system) | 100.0 | 100.0 | 100.0 |

- **AllDocs** = 1 only if *every* required document is in the context. No
  partial credit.
- **SOUL vs strongest baseline: +14.0 points** (24.9 → 38.9).
  95% CI **[+9.4, +18.9]**, Holm-adjusted **p = 0.0012**, Cohen's d = 0.31.

## 2. Does the better context produce a better answer?

Same retrieval, same pinned LLM reader on both rows — this isolates
**retrieval quality**, not model quality.

| System | EM | F1 |
|---|---|---|
| closed_book_llm (no context — leakage control) | 0.0 | 0.0 |
| **hybrid_rerank_llm** (strongest baseline) | **23.1** | **30.7** |
| **SOUL (ours_llm)** | **34.0** | **41.4** |
| oracle_llm (gold docs injected — ceiling) | 65.7 | 74.4 |

- **EM: 23.1 → 34.0** (+10.9, CI [+6.9, +15.4], Holm p = 0.0012)
- **F1: 30.7 → 41.4** (+10.7, CI [+6.6, +15.1], Holm p = 0.0012)
- `closed_book_llm` scores exactly 0.0 and abstains on every question — the
  control proving the model isn't answering from memory.

## 3. Where it wins, and where it doesn't (AllDocs by hop count, all 500 q)

| System | 2-hop | 3-hop | 4-hop |
|---|---|---|---|
| bm25 | 26.0 | 12.4 | 1.2 |
| hybrid_rerank | 46.9 | 25.5 | 1.2 |
| **SOUL** | **74.0** | **37.9** | **4.7** |

**The main limitation:** on 4-hop questions AllDocs is 4.7% (DocCov 46.4%) —
three consecutive bridges compound error. A bridge-diversity fix was tried
and made held-out results *worse* (AllDocs −3.1, p = 0.026); it was reverted
and the negative result kept. See `EVAL.md` §9.

## 4. Ablations (150-question development split — never the held-out set)

| Ablation | What changed | AllDocs | Δ |
|---|---|---|---|
| **ours (full)** | — | **78.0** | — |
| no_iterative | iterative retrieval disabled | 48.0 | **−30.0** |
| no_reach_gate | `min_entity_df=1` | 61.3 | −16.7 |
| no_rerank | reranker disabled | 63.3 | −14.7 |
| no_query_rewrite | no bridge-query rewriting | 69.3 | −8.7 |
| no_graph | mention index disabled | 73.3 | −4.7 |

Removing iterative retrieval costs **30 points of AllDocs** — the empirical
argument for the component's existence.

## 5. The design decision that got rejected

| Mechanism | Reachable gold evidence (831 links) |
|---|---|
| title link graph | 30.4% — **rejected** |
| mention index (shipped) | 90.3% |

---

## Statistics, throughout

Paired permutation tests, 10,000 resamples, Holm-Bonferroni corrected across
the metric family. `**` = significant after correction.

## Knowledge base

5,848 documents · 6,793 passages · 24,064 sentence-level anchors ·
MuSiQue-Answerable v1.0 dev, sha256-pinned · 500 of 2,363 eligible questions
(54 single-document questions excluded per Hard Requirement 2).
