"""Scripted 5-10 minute walkthrough.

Run after `eval`, so the discriminating examples can be chosen from real raw
outputs rather than cherry-picked by hand:

    python scripts/walkthrough.py

Sections
  1  the knowledge base
  2  the measurement that chose the bridge mechanism
  3  a successful multi-document answer, with verified citations
  4  the same question under the strongest single-shot baseline -- showing the
     hard ceiling that motivates the whole design
  5  a real failure surfaced by the evaluation
  6  provenance, checked mechanically
  7  the headline results
"""

from __future__ import annotations

import json
import sys
import time

import ctxlayer
from ctxlayer.config import PRESETS, EvalConfig, PathsConfig
from ctxlayer.eval.metrics import best_over_aliases
from ctxlayer.index.store import KnowledgeBase
from ctxlayer.ingest import load_processed
from ctxlayer.pipeline import ContextLayer
from ctxlayer.retrieve.decompose import analyze_question
from ctxlayer.synth.cite import verify_citations
from ctxlayer.synth.reader import ExtractiveReader

W = 96
SETTING = sys.argv[1] if len(sys.argv) > 1 else "unified"


def rule(title: str = "") -> None:
    print("\n" + "=" * W)
    if title:
        print(title)
        print("=" * W)


def show_context(ctx, gold=None, limit=6) -> None:
    print(f"\n  {len(ctx.passages)} passages | {len(ctx.doc_ids)} distinct documents "
          f"| {ctx.n_words} words")
    for i, (p, s) in enumerate(zip(ctx.passages, ctx.scores), 1):
        if i > limit:
            print(f"      ... and {len(ctx.passages) - limit} more")
            break
        flag = ""
        if gold is not None:
            flag = "   <-- GOLD" if p.passage_id in gold.gold_passage_ids else ""
        print(f"   [{i}] {p.doc_title}  ({s:.3f}){flag}")
    if gold is not None:
        gp = len(set(ctx.passage_ids) & set(gold.gold_passage_ids))
        gd = len(set(ctx.doc_ids) & set(gold.gold_doc_ids))
        print(f"\n  gold passages in context : {gp}/{len(gold.gold_passage_ids)}")
        print(f"  gold DOCUMENTS in context: {gd}/{len(set(gold.gold_doc_ids))}")


def answer_and_cite(kb, cfg, question, ctx, gold=None):
    plan = analyze_question(question)
    ans = ExtractiveReader(cfg.reader).read(question, ctx, focus=plan.focus)
    verify_citations(kb, ans)
    print(f"\n  ANSWER : {ans.answer or '(abstained)'}")
    if gold is not None:
        em, f1 = best_over_aliases(ans.answer, gold.all_answers())
        print(f"  GOLD   : {gold.answer}    ->  EM={em:.0f}  F1={f1:.2f}")
    for c in ans.claims:
        print(f"\n  CLAIM  : \"{c.text}\"   supported={c.supported}")
        for aid in c.anchor_ids:
            a = kb.resolve_anchor(aid)
            ok = kb.verify_anchor(aid)
            print(f"    cites {aid}")
            print(f"      source   : {a.doc_title}, sentence {a.sent_idx}, "
                  f"chars {a.char_start}-{a.char_end}")
            print(f"      verified : {ok}  (text byte-identical to source)")
            print(f"      quote    : \"{a.text[:150]}\"")
    return ans


def main() -> int:
    paths = PathsConfig()
    ecfg = EvalConfig(setting=SETTING)
    res = load_processed(paths)
    cfg = PRESETS["ours"]
    kb = KnowledgeBase.load(res, cfg, paths)
    kb.enable_score_cache(paths, cfg)
    by_qid = {q.qid: q for q in res.questions}

    # ---------------------------------------------------------------- 1
    rule("1. THE KNOWLEDGE BASE")
    st = res.stats
    print(f"  benchmark      : MuSiQue-Answerable v1.0 dev")
    print(f"  sha256         : {st['source_sha256'][:48]}...")
    print(f"  dev questions  : {st['selection']['n_source']}")
    print(f"  excluded       : {st['selection']['n_dropped_single_doc']} questions "
          f"whose gold evidence is in ONE document")
    print(f"                   (Hard Req #2: several chunks of one source are not")
    print(f"                    multi-document evidence, so they cannot be scored)")
    print(f"  evaluated      : {st['n_questions']} questions "
          f"(seeded, hop-stratified)")
    print(f"  knowledge base : {st['n_passages']} passages / "
          f"{st['n_documents']} documents / {st['n_anchors']} citable anchors")
    print(f"  hop mix        : {st['hop_type_hist']}")
    print(f"  gold docs/q    : {st['gold_docs_per_question_hist']}")

    # ---------------------------------------------------------------- 2
    rule("2. WHY A MENTION INDEX, NOT A LINK GRAPH  (measured, not assumed)")
    diag = paths.resolve("results_dir") / "diagnostic_reachability.json"
    if diag.exists():
        d = json.loads(diag.read_text(encoding="utf-8"))
        print(f"  Can the candidate generator even PROPOSE the rest of the gold")
        print(f"  evidence chain? A retriever cannot rank what it never sees.\n")
        print(f"    title-keyed document link graph : "
              f"{d['title_link_graph']['overall']*100:5.1f}%   <- built first, rejected")
        print(f"    surface-form mention index      : "
              f"{d['mention_index']['overall']*100:5.1f}%   <- what we use")
        b = d["bridge_answers_that_are_titles"]
        print(f"\n  Root cause: only {b['rate']*100:.1f}% of gold bridge answers "
              f"({b['hit']}/{b['total']}) are")
        print(f"  themselves article titles, so a title-keyed graph has no node")
        print(f"  to traverse to. The bridge entity is a MENTION inside the next")
        print(f"  gold paragraph.")
    else:
        print("  run: python -m ctxlayer.cli diagnose reachability")

    # ---------------------------------------------------------------- 3/4
    # Pick a question where `ours` got every gold document and the strongest
    # single-shot baseline did not -- chosen from raw outputs, not by hand.
    demo_qid = None
    try:
        from ctxlayer.eval.report import load_runs

        rs = load_runs(paths, ecfg.label)
        if "ours" in rs.runs and "hybrid_rerank" in rs.runs:
            base = {r["qid"]: r for r in rs.runs["hybrid_rerank"].records}
            for r in rs.runs["ours"].records:
                b = base.get(r["qid"])
                if b is None:
                    continue
                if (
                    r["metrics"]["ctx_all_docs"] >= 1.0
                    and b["metrics"]["ctx_all_docs"] < 1.0
                    and r["metrics"].get("answer_em", 1.0) >= 1.0
                    and r["n_hops"] >= 2
                ):
                    demo_qid = r["qid"]
                    break
    except Exception as e:  # pragma: no cover
        print(f"  (raw outputs unavailable: {e})")

    if demo_qid is None:
        demo_qid = next(
            (q.qid for q in res.questions if q.qid.startswith("2hop")), res.questions[0].qid
        )
    gold = by_qid[demo_qid]

    rule(f"3. A SUCCESSFUL MULTI-DOCUMENT ANSWER   [{gold.hop_type}]")
    print(f"  Q: {gold.question}")
    plan = analyze_question(gold.question)
    print(f"\n  query plan")
    print(f"    anchor entities (what the question gives us) : {plan.anchor_entities}")
    print(f"    relation terms  (what it asks)               : {plan.relation_terms[:6]}")
    print(f"    expected answer type                         : {plan.focus}")
    print(f"\n  gold evidence spans {len(set(gold.gold_doc_ids))} documents: "
          f"{gold.gold_doc_titles}")

    layer = ContextLayer(kb, cfg)
    t = time.time()
    ctx = layer.build_context(gold.question)
    dt = (time.time() - t) * 1000
    print(f"\n  OURS: iterative entity-bridged retrieval ({dt:.0f} ms)")
    print(f"    queries issued   : {len(ctx.trace.queries_issued)}")
    for q in ctx.trace.queries_issued[:6]:
        print(f"      - {q}")
    print(f"    bridge entities  : {ctx.trace.bridge_entities[:6]}")
    show_context(ctx, gold)
    answer_and_cite(kb, cfg, gold.question, ctx, gold)

    rule("4. THE SAME QUESTION, STRONGEST SINGLE-SHOT BASELINE")
    print("  `hybrid_rerank` = BM25F + dense + RRF + cross-encoder reranking.")
    print("  This is the best a single shot can do -- and it still cannot see the")
    print("  second document, because the question never names it. Reranking only")
    print("  reorders what recall proposed.\n")
    bcfg = PRESETS["hybrid_rerank"]
    bctx = ContextLayer(kb, bcfg).build_context(gold.question)
    show_context(bctx, gold)
    answer_and_cite(kb, bcfg, gold.question, bctx, gold)

    # ---------------------------------------------------------------- 5
    rule("5. A REAL FAILURE SURFACED BY THE EVALUATION")
    fail = None
    try:
        from ctxlayer.eval.report import load_runs

        rs = load_runs(paths, ecfg.label)
        for r in rs.runs["ours"].records:
            if r["metrics"]["ctx_all_docs"] >= 1.0 and r["metrics"].get("answer_em", 0) < 1.0:
                fail = r
                break
    except Exception:
        pass
    if fail is None:
        print("  (run `python -m ctxlayer.cli eval` first)")
    else:
        print("  The largest failure bucket is NOT a retrieval failure: the context")
        print("  contains every gold document and the fixed extractive reader still")
        print("  answers wrongly. This is a reader limitation, and it is why the")
        print("  `oracle` row is reported -- so reading error is never silently")
        print("  counted as context-layer success.\n")
        print(f"  Q            : {fail['question']}   [{fail['hop_type']}]")
        print(f"  gold answer  : {fail['gold_answer']}")
        print(f"  predicted    : {fail.get('answer', {}).get('text', '')!r}")
        print(f"  gold docs    : {fail['gold_doc_titles']}")
        print(f"  context docs : {list(dict.fromkeys(fail['context_doc_titles']))[:6]}")
        print(f"  all gold documents retrieved : "
              f"{fail['metrics']['ctx_all_docs'] >= 1.0}")
        print(f"  answer correct               : "
              f"{fail['metrics'].get('answer_em', 0) >= 1.0}")

    # ---------------------------------------------------------------- 6
    rule("6. PROVENANCE, CHECKED MECHANICALLY")
    bad = sum(1 for a in kb.anchors if not kb.verify_anchor(a))
    print(f"  anchors in knowledge base           : {len(kb.anchors)}")
    print(f"  failing byte-exact re-resolution    : {bad}")
    print(f"\n  A fabricated anchor id is rejected, not accepted:")
    print(f"    kb.verify_anchor('deadbeef::p0::s0') = "
          f"{kb.verify_anchor('deadbeef::p0::s0')}")
    print(f"\n  Citations are checked L1 resolvable (byte-exact) -> L2 supported")
    print(f"  (content words present; numbers must match EXACTLY, because '1987'")
    print(f"  and '1978' are semantically near-identical and factually opposite)")
    print(f"  -> L3 repair, with repairs recorded separately rather than folded")
    print(f"  into precision.")

    # ---------------------------------------------------------------- 7
    rule("7. HEADLINE RESULTS")
    rep = paths.resolve("results_dir") / f"report_{ecfg.label}.json"
    if rep.exists():
        d = json.loads(rep.read_text(encoding="utf-8"))
        print(f"  setting: {d['setting']}   n = {d['n_questions']}\n")
        print(d["headline_table"])
        print()
        prim = [c for c in d["comparisons"]
                if c["baseline"] == "hybrid_rerank"
                and c["metric"] in ("ctx_all_docs", "ctx_support_all")]
        for c in prim:
            print(f"  {c['metric']:20s} {c['baseline_mean']*100:5.1f} -> "
                  f"{c['system_mean']*100:5.1f}  ({c['delta']*100:+.1f} pts, "
                  f"95% CI [{c['delta_ci95'][0]*100:+.1f}, "
                  f"{c['delta_ci95'][1]*100:+.1f}], "
                  f"adj p = {c['p_adjusted']:.4f})")
    else:
        print("  run: python -m ctxlayer.cli report --setting " + ecfg.label)
    rule()
    print("Full detail: DESIGN.md (architecture + reasoning), EVAL.md (results).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
