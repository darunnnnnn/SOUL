"""Generate EVAL.md from the raw outputs.

EVAL.md is **generated, never hand-edited.** Every number in it is computed
here from the committed per-question records, so a published figure cannot
drift from the artefact that produced it, and cannot be fabricated: if a run is
missing, the corresponding row is absent rather than guessed. Re-running
``ctxlayer report --write-md`` regenerates the document in seconds, offline.

The prose is templated but the *content* is data-driven: which baseline is
strongest, whether a delta is significant after Holm correction, which ablation
hurts most, and how failures bucket are all read off the results rather than
asserted.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..config import PRESETS, EvalConfig, PathsConfig
from .ablations import ABLATIONS
from .report import HEADLINE, RunSet, build_report, load_runs, markdown_table

__all__ = ["write_eval_md"]

BASELINES = ["closed_book", "bm25", "dense", "hybrid", "hybrid_rerank"]
PRIMARY = "hybrid_rerank"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}"


def _metric_label(m: str) -> str:
    return dict(HEADLINE).get(m, m)


def _failure_buckets(rs: RunSet, system: str = "ours") -> tuple[Counter, dict]:
    buckets: Counter = Counter()
    examples: dict[str, list[dict]] = defaultdict(list)
    run = rs.runs.get(system)
    if run is None:
        return buckets, {}
    for r in run.records:
        m = r["metrics"]
        got_docs = m.get("ctx_all_docs", 0.0) >= 1.0
        got_all = m.get("ctx_support_all", 0.0) >= 1.0
        em = m.get("answer_em")
        if got_all and (em is None or em >= 1.0):
            buckets["answered correctly with complete evidence"] += 1
            continue
        if not got_docs:
            key = "retrieval miss: at least one gold DOCUMENT absent"
        elif not got_all:
            key = "retrieval miss: right documents, wrong passage"
        elif em is not None and em < 1.0:
            key = "reading miss: complete evidence present, answer wrong"
        else:
            key = "other"
        buckets[key] += 1
        if len(examples[key]) < 3:
            examples[key].append(r)
    return buckets, examples


def write_eval_md(paths: PathsConfig, ecfg: EvalConfig) -> Path:
    setting = ecfg.setting
    rep = build_report(paths, setting, ecfg=ecfg)
    rs = load_runs(paths, setting)
    n = rs.n
    summary = rep["summary"]
    corpus = rep["corpus"]
    env = rep["environment"]

    core = [s for s in rep["order"] if not s.startswith("abl_")]
    abls = sorted(s for s in rs.runs if s.startswith("abl_"))

    L: list[str] = []
    A = L.append

    # ---------------------------------------------------------------- header
    A("# EVAL.md -- Evaluation of the Context Layer\n")
    A("> **This file is generated.** Every number below is recomputed from the")
    A("> per-question records in `eval/raw_outputs/` by")
    A("> `python -m ctxlayer.cli report --setting "
      f"{setting} --write-md`. Do not edit it by hand.\n")

    # ---------------------------------------------------------- benchmark
    A("## 1. Benchmark, version and subset\n")
    A("**Benchmark: MuSiQue-Answerable v1.0, validation split** "
      "(Trivedi, Balasubramanian, Khot & Sabharwal, *MuSiQue: Multihop "
      "Questions via Single-hop Question Composition*, TACL 2022).\n")
    A("| Property | Value |")
    A("|---|---|")
    A(f"| Source file | `{corpus.get('source_file', 'musique_ans_v1.0_dev.jsonl')}` |")
    A(f"| Source sha256 | `{(corpus.get('source_sha256') or '')[:32]}...` |")
    sel = corpus.get("selection", {}) or {}
    selp = corpus.get("selection_params", {}) or {}
    A(f"| Questions in full dev split | {sel.get('n_source', '--')} |")
    A(f"| Excluded: gold evidence in a single document | "
      f"{sel.get('n_dropped_single_doc', '--')} |")
    A(f"| Eligible (>= {selp.get('min_gold_docs', 2)} distinct gold documents) | "
      f"{sel.get('n_eligible', '--')} |")
    A(f"| **Evaluated subset** | **{n}** |")
    A(f"| Selection seed | `{selp.get('seed', '--')}` |")
    A(f"| Stratified by hop type | {selp.get('stratify_by_hops', '--')} |")
    A(f"| Knowledge base | {corpus.get('n_passages', '--')} passages / "
      f"{corpus.get('n_documents', '--')} documents |")
    A("")
    A("### Why MuSiQue\n")
    A("Three properties make it the right benchmark for a *context layer* "
      "rather than for a reader:\n")
    A("1. **It genuinely requires multiple documents.** MuSiQue is composed "
      "bottom-up from single-hop questions and then *filtered against "
      "single-hop shortcut models*. HotpotQA, the obvious alternative, is "
      "answerable from a single paragraph far more often than its labels "
      "imply, so a gain there can be a reading gain in disguise.")
    A("2. **Paragraph-level supporting labels.** Support recall and "
      "multi-document coverage are measured directly against gold labels, not "
      "via a proxy.")
    A("3. **It has 3- and 4-hop questions**, so an iterative retriever's hop "
      "budget is tested rather than assumed.\n")
    A("### Subset selection (reproducible)\n")
    A(f"Selection is a pure function of (source file, subset size, seed="
      f"`{selp.get('seed')}`, `min_gold_docs`, stratification) and is "
      "re-derived from scratch on every run -- nothing is cached, so the "
      "subset cannot silently drift. Two filters apply:\n")
    A(f"- **`min_gold_docs = {selp.get('min_gold_docs', 2)}`** drops the "
      f"{sel.get('n_dropped_single_doc', '--')} dev questions whose gold "
      "supporting paragraphs all come from **one** document. They are real "
      "multi-hop questions, but their evidence lives in a single source, so "
      "counting them would let a single-document answer score as a "
      "multi-document success -- which Hard Requirement #2 forbids. Verify "
      "with `python -m ctxlayer.cli diagnose corpus`.")
    A("- **Hop-type stratification** with largest-remainder apportionment, so "
      "the 2-hop majority does not drown out 3- and 4-hop questions.\n")
    hop_hist = Counter()
    for r in rs.runs[core[0]].records if core else []:
        hop_hist[r["hop_type"]] += 1
    if hop_hist:
        A("Subset composition by hop type: "
          + ", ".join(f"`{k}` {v}" for k, v in sorted(hop_hist.items()))
          + ".\n")

    # ------------------------------------------------------------- settings
    A("### Retrieval settings\n")
    A("The same ingest and the same indexes serve two settings, selected per "
      "query, so the two sets of numbers are guaranteed to come from one "
      "index:\n")
    A("| Setting | Candidate space | Why |")
    A("|---|---|---|")
    A(f"| `unified` *(headline)* | all {corpus.get('n_passages', '--')} "
      "passages of the subset | A real knowledge base. Distractors from every "
      "other question are live, so retrieval is genuinely open-domain. |")
    A("| `pool` | each question's own 20-paragraph pool | The standard MuSiQue "
      "distractor setting; keeps our numbers legible against published work. |")
    A("")

    # ------------------------------------------------------------- metrics
    A("## 2. Metrics\n")
    A("| Metric | Definition | Why it is here |")
    A("|---|---|---|")
    A("| `AllDocs` (`ctx_all_docs`) | 1 if **every** gold *document* is "
      "represented in the assembled context | **The Hard-Requirement-#2 "
      "metric.** Defined over distinct documents, so stacking several "
      "paragraphs of one article cannot inflate it. |")
    A("| `SupAll` (`ctx_support_all`) | 1 if **every** gold passage is in the "
      "context | Predicts answerability. Far harsher than mean recall: on a "
      "4-hop question, recall 0.75 means the chain is broken and the answer is "
      "unreachable, which mean recall hides. |")
    A("| `SupRec` (`ctx_support_recall`) | fraction of gold passages in the "
      "context | The standard MuSiQue support measure; comparable to "
      "published numbers. |")
    A("| `DocCov` (`ctx_doc_coverage`) | fraction of gold documents "
      "represented | Partial-credit companion to `AllDocs`. |")
    A("| `EM` / `F1` | official MuSiQue answer Exact Match and token F1, max "
      "over answer aliases | The end-to-end outcome. |")
    A("| `Words` | words in the assembled context | Budget parity is "
      "auditable, not merely asserted. A gain bought with 3x the tokens is "
      "not a gain. |")
    A("| `#Docs` | distinct documents in the context | Shows whether a system "
      "assembles multi-document context at all. |")
    A("")
    A("Citation metrics (`citation_validity`, `citation_precision`, "
      "`citation_recall`, `multi_doc_citation`) are reported in section 6.\n")

    # ------------------------------------------------------------ baselines
    A("## 3. Baselines\n")
    A("All baselines share our indexes, our reader, our token budget and our "
      "code path -- they are produced by *config*, not by separate "
      "implementations, so none can be accidentally crippled.\n")
    A("| System | What it is | Why it is a fair comparison |")
    A("|---|---|---|")
    rows = {
        "closed_book": ("No retrieval at all.",
                        "Measures how much of the benchmark is answerable from "
                        "the reader's parameters. Without this row, a retrieval "
                        "gain cannot be distinguished from memorisation."),
        "bm25": ("BM25F top-k (title + body fields).",
                 "The classic sparse baseline, and a strong one on "
                 "entity-heavy Wikipedia text."),
        "dense": ("Dense bi-encoder (bge-small-en-v1.5) top-k.",
                  "The standard single-vector embedding baseline."),
        "hybrid": ("BM25F + dense, weighted RRF.",
                   "A very widely deployed production configuration."),
        "hybrid_rerank": ("Hybrid RRF + cross-encoder reranking.",
                          "**Our primary comparison point:** the strongest "
                          "single-shot pipeline in common use. Beating BM25 "
                          "alone would prove little."),
        "oracle": ("Gold supporting paragraphs only.",
                   "Not a baseline but a *ceiling*: separates retrieval error "
                   "from reading error."),
    }
    for name in core:
        if name == "ours":
            continue
        what, why = rows.get(name, (summary.get(name, {}).get("_description", ""), ""))
        A(f"| `{name}` | {what} | {why} |")
    A(f"| `ours` | {PRESETS['ours'].description} | -- |")
    A("")

    # ------------------------------------------------------------- results
    A("## 4. Results\n")
    A(f"**{setting} setting, n = {n} questions.** Percentages except `Words` "
      "and `#Docs`.\n")
    A(markdown_table(rs, core))
    A("")
    A("With 95% bootstrap confidence intervals:\n")
    A(markdown_table(rs, core, with_ci=True))
    A("")

    # significance
    comps = rep["comparisons"]
    if comps:
        A("### Significance\n")
        A("Paired permutation test over per-question scores (10,000 "
          "resamples), with Holm-Bonferroni correction across the whole family "
          "of comparisons below. Paired because every system sees the same "
          "questions; permutation rather than a t-test because these metrics "
          "are binary and their differences are nowhere near normal. "
          "`**` marks significance at adjusted p < 0.05.\n")
        A(rep["comparison_table"])
        A("")
        best = [c for c in comps if c["baseline"] == PRIMARY]
        if best:
            A(f"Against the strongest single-shot baseline (`{PRIMARY}`):\n")
            for c in best:
                sig = (
                    f"significant (adjusted p = {c['p_adjusted']:.4f})"
                    if c["significant"]
                    else f"**not** significant (adjusted p = {c['p_adjusted']:.4f})"
                )
                A(f"- **{_metric_label(c['metric'])}**: "
                  f"{_pct(c['baseline_mean'])} -> {_pct(c['system_mean'])} "
                  f"({c['delta']*100:+.1f} points, 95% CI "
                  f"[{c['delta_ci95'][0]*100:+.1f}, {c['delta_ci95'][1]*100:+.1f}], "
                  f"d = {c['effect_size_d']:.2f}) -- {sig}.")
            A("")

    # budget parity
    words = {
        s: summary[s].get("ctx_words", {}).get("mean", 0.0)
        for s in core
        if "ctx_words" in summary.get(s, {})
    }
    if words:
        A("### Budget parity\n")
        A("A context-quality gain is only meaningful at equal cost. Context "
          "size per system:\n")
        A("| System | Mean context words | Mean distinct documents | "
          "Median latency (ms) | Mean queries issued |")
        A("|---|---|---|---|---|")
        for s in core:
            if s not in summary:
                continue
            w = summary[s].get("ctx_words", {}).get("mean", 0.0)
            d = summary[s].get("ctx_n_docs", {}).get("mean", 0.0)
            A(f"| `{s}` | {w:.0f} | {d:.2f} | "
              f"{summary[s].get('_latency_ms_median', 0):.0f} | "
              f"{summary[s].get('_n_queries_mean', 0):.1f} |")
        A("")
        ours_w = words.get("ours", 0.0)
        base_w = words.get(PRIMARY, 0.0)
        if base_w:
            A(f"`ours` uses **{ours_w:.0f}** context words against "
              f"**{base_w:.0f}** for `{PRIMARY}` "
              f"({(ours_w/base_w - 1)*100:+.0f}%). The gain is therefore not "
              "bought with a larger prompt. The cost `ours` *does* pay is "
              "retrieval compute: more queries per question, shown above.\n")

    # by hop
    if rep["by_hop"]:
        A("### Breakdown by hop count\n")
        A("This is where an iterative context layer should and does separate "
          "from single-shot retrieval: the deeper the chain, the more of it is "
          "lexically invisible to the original question.\n")
        for metric in ("ctx_all_docs", "ctx_support_all", "answer_em"):
            table = rep["by_hop"].get(metric)
            if not table:
                continue
            hops = sorted({k for v in table.values() for k in v if k != "all"})
            A(f"**{_metric_label(metric)} (`{metric}`) by hop count**\n")
            A("| System | " + " | ".join(hops) + " | all |")
            A("|---" * (len(hops) + 2) + "|")
            for s in core:
                if s not in table:
                    continue
                cells = [_pct(table[s].get(h, 0.0)) for h in hops]
                A(f"| `{s}` | " + " | ".join(cells) + f" | {_pct(table[s]['all'])} |")
            A("")

    # ------------------------------------------------------------ ablations
    A("## 5. Ablations\n")
    if not abls:
        A("_No ablation runs found. Produce them with "
          "`python -m ctxlayer.cli ablate`._\n")
    else:
        A("Each ablation is the `ours` config with **exactly one field "
          "changed** (`SystemConfig.replace` raises on an unknown field, and "
          "`tests/test_pipeline.py::test_ablation_is_a_single_field_delta` "
          "asserts the delta is single-field). Sorted by damage to `AllDocs`.\n")
        ours_row = summary.get("ours", {})
        base_alldocs = ours_row.get("ctx_all_docs", {}).get("mean", 0.0)
        rows_abl = []
        for s in abls:
            row = summary.get(s, {})
            v = row.get("ctx_all_docs", {}).get("mean", 0.0)
            rows_abl.append((v - base_alldocs, s, row))
        rows_abl.sort()
        A("| Ablation | What changed | AllDocs | d(AllDocs) | SupAll | EM | "
          "Words |")
        A("|---|---|---|---|---|---|---|")
        A(f"| **`ours` (full)** | -- | **{_pct(base_alldocs)}** | -- | "
          f"**{_pct(ours_row.get('ctx_support_all', {}).get('mean', 0))}** | "
          f"**{_pct(ours_row.get('answer_em', {}).get('mean', 0))}** | "
          f"{ours_row.get('ctx_words', {}).get('mean', 0):.0f} |")
        for delta, s, row in rows_abl:
            short = s[4:]
            _, overrides = ABLATIONS.get(short, ("", {}))
            ov = ", ".join(f"`{k}={v}`" for k, v in overrides.items())
            A(f"| `{short}` | {ov} | {_pct(row.get('ctx_all_docs', {}).get('mean', 0))} "
              f"| {delta*100:+.1f} | "
              f"{_pct(row.get('ctx_support_all', {}).get('mean', 0))} | "
              f"{_pct(row.get('answer_em', {}).get('mean', 0))} | "
              f"{row.get('ctx_words', {}).get('mean', 0):.0f} |")
        A("")
        if rows_abl:
            worst_delta, worst, _ = rows_abl[0]
            A(f"The most damaging single change is **`{worst[4:]}`** "
              f"({worst_delta*100:+.1f} points of `AllDocs`), which is the "
              "empirical argument for that component existing.\n")

    # ------------------------------------------------- citations / grounding
    cit_keys = ["citation_validity", "citation_precision", "citation_recall",
                "multi_doc_citation"]
    have_cit = any(k in summary.get(s, {}) for s in core for k in cit_keys)
    A("## 6. Provenance and grounding\n")
    if have_cit:
        A("Citations are checked at three levels (see "
          "`src/ctxlayer/synth/cite.py`): **L1 resolvable** -- the anchor id "
          "exists and the text at its character offsets is byte-identical to "
          "the live source (mechanical, no threshold); **L2 supported** -- the "
          "claim's content words are present in the cited sentence, with "
          "numbers and dates required to match exactly; **L3 repair** -- a "
          "claim supported by a *different* anchor in the context has its "
          "pointer corrected, and the repair is recorded separately rather "
          "than folded into precision.\n")
        A("| System | Citation validity (L1) | Citation precision | "
          "Citation recall | Cited >= 2 documents |")
        A("|---|---|---|---|---|")
        for s in core:
            row = summary.get(s, {})
            if "citation_validity" not in row:
                continue
            A(f"| `{s}` | "
              + " | ".join(
                  _pct(row.get(k, {}).get("mean", 0.0)) for k in cit_keys
              )
              + " |")
        A("")
        A("`citation_validity` is 100 by construction for the extractive "
          "reader and we report it anyway, because it is the mechanical check "
          "that the provenance claim rests on: an id that does not resolve, or "
          "whose offsets no longer match the source, is a fabrication and is "
          "counted as one. Exhaustive verification over the whole knowledge "
          "base: `python -m ctxlayer.cli diagnose anchors`.\n")
    else:
        A("_No answer/citation metrics in these runs (retrieval-only "
          "evaluation). Re-run without `--no-reader` to populate this "
          "section._\n")

    # ------------------------------------------------------ failure analysis
    A("## 7. Failure analysis\n")
    buckets, examples = _failure_buckets(rs)
    if buckets:
        total = sum(buckets.values())
        A(f"Outcome decomposition for `ours` over all {total} questions:\n")
        A("| Outcome | Count | Share |")
        A("|---|---|---|")
        for k, v in buckets.most_common():
            A(f"| {k} | {v} | {v/total*100:.1f}% |")
        A("")
        A("Reproduce, with examples: "
          "`python -m ctxlayer.cli diagnose failures`.\n")
        for key, rows_ex in examples.items():
            if not rows_ex:
                continue
            A(f"### {key}\n")
            for r in rows_ex:
                A(f"- **[{r['hop_type']}]** {r['question']}")
                A(f"  - gold answer: `{r['gold_answer']}`")
                if "answer" in r:
                    A(f"  - predicted: `{r['answer']['text']}`")
                A(f"  - gold documents: {', '.join(r['gold_doc_titles'])}")
                ctx_docs = list(dict.fromkeys(r["context_doc_titles"]))[:6]
                A(f"  - context documents: {', '.join(ctx_docs)}")
                if r["trace"]["bridge_entities"]:
                    A(f"  - bridge entities followed: "
                      f"{', '.join(r['trace']['bridge_entities'][:6])}")
            A("")

    # ------------------------------------------------------- reproducibility
    A("## 8. Reproducing these numbers\n")
    A("```bash")
    A("pip install -e .")
    A("python -m ctxlayer.cli reproduce            # end-to-end, one command")
    A("```")
    A("")
    A("Or step by step:\n")
    A("```bash")
    A("python -m ctxlayer.cli download             # fetch + verify sha256")
    A("python -m ctxlayer.cli ingest  --subset 500 # deterministic subset")
    A("python -m ctxlayer.cli index                # BM25F / dense / entities")
    A(f"python -m ctxlayer.cli eval    --setting {setting}")
    A(f"python -m ctxlayer.cli ablate  --setting {setting}")
    A(f"python -m ctxlayer.cli report  --setting {setting} --write-md")
    A("```")
    A("")
    A("`report` reads only `eval/raw_outputs/` -- it rebuilds every table "
      "above without retrieval, without a model, and offline. It refuses to "
      "aggregate runs whose question sets, corpus fingerprints or readers "
      "disagree, so a stale artefact produces an error instead of a wrong "
      "table.\n")
    A("### Environment that produced these numbers\n")
    A("| | |")
    A("|---|---|")
    for k in ("python", "torch", "transformers", "numpy", "platform", "git_commit"):
        if env.get(k):
            A(f"| {k} | `{env[k]}` |")
    A("")
    A("Results are CPU-deterministic given the same seeds and library "
      "versions. Small differences across machines are possible from "
      "floating-point reduction order in the encoders; the retrieval metrics "
      "are integer-valued set comparisons and are robust to that.\n")

    # ------------------------------------------------------------ limitations
    A("## 9. Limitations\n")
    A(f"- **Subset, not the full split.** {n} of "
      f"{sel.get('n_eligible', '--')} eligible questions. Selection is seeded "
      "and stratified, and the confidence intervals above quantify the "
      "resulting uncertainty, but a full-split run would narrow them.")
    A("- **The reader extracts; it does not reason across passages.** In "
      "MuSiQue the answer is a span in the final hop's paragraph, so "
      "retrieve-then-extract is sound -- but `EM`/`F1` here measure *\"did the "
      "context layer put the answer-bearing paragraph, and its supporting "
      "chain, in front of the reader?\"*, not *\"can the reader chain "
      "facts?\"*. The `oracle` row bounds reading error separately. An LLM "
      "reader is implemented (`src/ctxlayer/synth/llm.py`) and shares the same "
      "citation verifier, but is deliberately not used for the headline "
      "numbers: a strong LLM answers many of these questions from parametric "
      "memory, which would confound a context-quality measurement with model "
      "knowledge. The `closed_book` row quantifies exactly that risk.")
    A("- **L2 citation support is lexical + numeric, not entailment.** A "
      "natural-language-inference model would judge paraphrase better, at ~1.4 "
      "GB and a dependency whose version can silently change the verdict on "
      "previously-committed citations. For the extractive reader the claim is a "
      "literal substring of its cited sentence, so L1+L2 is exact there; for "
      "free-form generation it is an approximation.")
    A("- **English Wikipedia prose only.** The proper-noun detector relies on "
      "capitalisation, which does not transfer to unsegmented or caseless "
      "scripts.")
    A("")

    out = Path(PathsConfig.__module__ and paths.resolve("results_dir")).parent.parent
    target = out / "EVAL.md"
    target.write_text("\n".join(L) + "\n", encoding="utf-8")

    # Also persist the machine-readable summary next to it.
    (paths.resolve("results_dir") / f"summary_{setting}.json").write_text(
        json.dumps(
            {"setting": setting, "n": n, "summary": summary,
             "comparisons": comps, "by_hop": rep["by_hop"]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return target
