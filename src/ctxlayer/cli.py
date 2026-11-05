"""Command-line interface.

``argparse`` rather than a CLI framework: the dependency list is part of the
deliverable, and a reviewer should be able to run this with torch +
transformers + numpy and nothing else.

Commands
--------
``download``   fetch the benchmark, verify its sha256
``ingest``     build the processed corpus + questions (deterministic subset)
``index``      build BM25F / dense / entity indexes
``ask``        answer one question, printing the assembled context and the
               claim-level citations -- the walkthrough command
``eval``       run systems, writing per-question raw outputs
``ablate``     run the ablation suite
``report``     rebuild every published number from the raw outputs (offline)
``diagnose``   reproduce the design measurements (e.g. graph vs mention index)
``reproduce``  the one command behind Hard Requirement #4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from . import _env  # noqa: F401  (environment hardening must happen first)
from .config import (
    PRESETS,
    EvalConfig,
    IngestConfig,
    PathsConfig,
    SystemConfig,
    load_config,
)
from .index.store import KnowledgeBase
from .ingest.musique import ingest_musique, load_processed, save_processed

# --------------------------------------------------------------------------
# data source (pinned)
# --------------------------------------------------------------------------

MUSIQUE_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/"
    "musique_ans_v1.0_dev.jsonl"
)
MUSIQUE_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"


def cmd_download(args: argparse.Namespace) -> int:
    paths = PathsConfig()
    dest_dir = paths.resolve("raw_dir")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / MUSIQUE_FILE
    if dest.exists() and not args.force:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == MUSIQUE_SHA256:
            print(f"already present and verified: {dest}")
            return 0
        print(f"checksum mismatch, re-downloading (got {digest[:16]}...)")
    print(f"downloading {MUSIQUE_URL}")
    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(MUSIQUE_URL) as r, tmp.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != MUSIQUE_SHA256:
        tmp.unlink(missing_ok=True)
        print(
            f"ERROR: sha256 mismatch.\n  expected {MUSIQUE_SHA256}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1
    tmp.replace(dest)
    print(f"verified sha256 {digest}\nsaved to {dest}")
    return 0


# --------------------------------------------------------------------------
# ingest / index
# --------------------------------------------------------------------------


def _ingest_cfg(args: argparse.Namespace) -> IngestConfig:
    return IngestConfig(
        subset_size=args.subset,
        seed=args.seed,
        min_gold_docs=args.min_gold_docs,
        stratify_by_hops=not args.no_stratify,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = PathsConfig()
    res = ingest_musique(_ingest_cfg(args), paths)
    out = save_processed(res, paths)
    print(json.dumps(res.stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    paths = PathsConfig()
    try:
        res = load_processed(paths)
    except FileNotFoundError:
        print("no processed corpus; ingesting first")
        res = ingest_musique(_ingest_cfg(args), paths)
        save_processed(res, paths)
    cfg = PRESETS["ours"]
    t = time.time()
    kb = KnowledgeBase.build(res, cfg, paths, verbose=True)
    kb.save(paths, cfg)
    print(f"\nbuilt in {time.time()-t:.1f}s")
    print(json.dumps(kb.stats["index"], indent=2))
    return 0


def _load_kb(paths: PathsConfig, cfg: SystemConfig, rebuild: bool = False):
    res = load_processed(paths)
    if rebuild:
        return res, KnowledgeBase.build(res, cfg, paths, verbose=True)
    try:
        return res, KnowledgeBase.load(res, cfg, paths)
    except FileNotFoundError:
        print("index not found; building")
        kb = KnowledgeBase.build(res, cfg, paths, verbose=True)
        kb.save(paths, cfg)
        return res, kb


# --------------------------------------------------------------------------
# ask (walkthrough)
# --------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import ContextLayer
    from .retrieve.decompose import analyze_question
    from .synth.cite import verify_citations

    paths = PathsConfig()
    cfg = load_config(args.config) if args.config else PRESETS[args.system]
    res, kb = _load_kb(paths, cfg)
    kb.enable_score_cache(paths, cfg)

    question = args.question
    gold = None
    if args.qid:
        match = [q for q in res.questions if q.qid == args.qid]
        if not match:
            print(f"qid {args.qid!r} not in the processed subset", file=sys.stderr)
            return 1
        question = match[0].question
        gold = match[0]

    plan = analyze_question(question)
    layer = ContextLayer(kb, cfg)
    t0 = time.time()
    ctx = layer.build_context(
        question, gold_passage_ids=gold.gold_passage_ids if (gold and cfg.mode == "oracle") else None
    )
    dt = (time.time() - t0) * 1000

    print("=" * 100)
    print(f"QUESTION : {question}")
    print(f"SYSTEM   : {cfg.name}  ({cfg.description})")
    print(f"PLAN     : anchors={plan.anchor_entities} relation={plan.relation_terms[:6]} focus={plan.focus}")
    if ctx.trace.bridge_entities:
        print(f"BRIDGES  : {ctx.trace.bridge_entities}")
    print(f"QUERIES  : {len(ctx.trace.queries_issued)} issued")
    for q in ctx.trace.queries_issued[:8]:
        print(f"           - {q}")
    print("=" * 100)
    print(f"\nASSEMBLED CONTEXT  ({len(ctx.passages)} passages, "
          f"{len(ctx.doc_ids)} distinct documents, {ctx.n_words} words, {dt:.0f} ms)\n")
    for i, (p, s) in enumerate(zip(ctx.passages, ctx.scores), 1):
        flag = ""
        if gold:
            flag = "  <-- GOLD" if p.passage_id in gold.gold_passage_ids else ""
        print(f"  [{i}] {p.doc_title}  (score {s:.4f}, {p.n_words}w){flag}")
        print(f"      {p.text[:200]}{'...' if len(p.text) > 200 else ''}")
    if gold:
        found = set(ctx.passage_ids) & set(gold.gold_passage_ids)
        docs = set(ctx.doc_ids) & set(gold.gold_doc_ids)
        print(f"\n  gold passages in context: {len(found)}/{len(gold.gold_passage_ids)}"
              f"   gold documents: {len(docs)}/{len(set(gold.gold_doc_ids))}")

    if args.no_answer:
        return 0

    from .synth.reader import ExtractiveReader

    reader = ExtractiveReader(cfg.reader)
    ans = reader.read(question, ctx, focus=plan.focus)
    verify_citations(kb, ans)

    print("\n" + "=" * 100)
    print(f"ANSWER   : {ans.answer or '(abstained)'}")
    if gold:
        from .eval.metrics import best_over_aliases

        em, f1 = best_over_aliases(ans.answer, gold.all_answers())
        print(f"GOLD     : {gold.answer}   -> EM={em:.0f} F1={f1:.2f}")
    print("\nCLAIMS AND CITATIONS")
    for c in ans.claims:
        mark = "OK " if c.supported else "!! "
        print(f"  {mark} \"{c.text}\"")
        for aid in c.anchor_ids:
            a = kb.resolve_anchor(aid)
            verified = kb.verify_anchor(aid)
            print(f"       -> {aid}")
            print(f"          {a.doc_title} (sentence {a.sent_idx}, chars "
                  f"{a.char_start}-{a.char_end}) verified={verified}")
            print(f"          \"{a.text[:180]}\"")
    print("=" * 100)
    return 0


# --------------------------------------------------------------------------
# eval / ablate / report
# --------------------------------------------------------------------------


def _reader_for(cfg: SystemConfig):
    if cfg.reader.kind == "none":
        return None
    if cfg.reader.kind == "llm":
        from .synth.llm import LLMReader

        r = LLMReader(cfg.reader)
        if not r.available:
            print(
                f"WARNING: no API key for {cfg.reader.llm_provider}; "
                "falling back to the local extractive reader",
                file=sys.stderr,
            )
        else:
            return r
    from .synth.reader import ExtractiveReader

    return ExtractiveReader(cfg.reader)


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import run_system

    paths = PathsConfig()
    ecfg = EvalConfig(setting=args.setting, seed=args.seed,
                      out_label=args.out_label or "")
    systems = args.systems.split(",") if args.systems else list(PRESETS)
    base_cfg = PRESETS["ours"]
    res, kb = _load_kb(paths, base_cfg)
    kb.enable_score_cache(paths, base_cfg)

    questions = res.questions[: args.limit] if args.limit else res.questions
    # One reader instance shared by every system: answer-level deltas cannot
    # come from a different generator.
    reader = None if args.no_reader else _reader_for(base_cfg)

    print(f"\ncorpus: {kb.n_passages} passages / {kb.n_documents} documents")
    print(f"questions: {len(questions)}   setting: {args.setting}")
    print(f"reader: {base_cfg.reader.model if reader else 'none (retrieval only)'}\n")

    for name in systems:
        if name not in PRESETS:
            print(f"unknown system {name!r}; known: {list(PRESETS)}", file=sys.stderr)
            return 1
        cfg = PRESETS[name]
        run_system(kb, cfg, questions, ecfg, paths, reader=reader, setting=args.setting)
    print("\nraw outputs ->", paths.resolve("raw_outputs_dir") / args.setting)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from .eval.ablations import ABLATIONS, run_ablations

    paths = PathsConfig()
    ecfg = EvalConfig(setting=args.setting, seed=args.seed,
                      out_label=args.out_label or f"{args.setting}_ablation")
    if args.list:
        for name, (desc, overrides) in ABLATIONS.items():
            print(f"{name:28s} {desc}")
            print(f"{'':28s}   {overrides}")
        return 0
    return run_ablations(
        paths, ecfg, limit=args.limit, only=args.only.split(",") if args.only else None,
        no_reader=args.no_reader,
    )


def cmd_report(args: argparse.Namespace) -> int:
    from .eval.report import build_report
    from .eval.writeup import write_eval_md

    paths = PathsConfig()
    ecfg = EvalConfig(setting=args.setting, seed=args.seed,
                      out_label=args.out_label or "")
    rep = build_report(paths, ecfg.label, ecfg=ecfg)
    print(f"\n=== {args.setting.upper()} SETTING | n={rep['n_questions']} ===\n")
    print(rep["headline_table"])
    if rep["comparisons"]:
        print("\nSIGNIFICANCE vs baselines (paired permutation, Holm-adjusted)\n")
        print(rep["comparison_table"])
    out = paths.resolve("results_dir") / f"report_{ecfg.label}.json"
    print(f"\nreport -> {out}")
    if args.write_md:
        p = write_eval_md(paths, ecfg)
        print(f"EVAL.md -> {p}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .eval.diagnose import run_diagnostic

    return run_diagnostic(args.what, PathsConfig(), limit=args.limit)


def cmd_reproduce(args: argparse.Namespace) -> int:
    """The single command behind Hard Requirement #4."""
    from .eval.ablations import run_ablations
    from .eval.report import build_report
    from .eval.runner import run_system
    from .eval.writeup import write_eval_md

    paths = PathsConfig()
    t_all = time.time()

    print("[1/6] download + verify benchmark")
    if cmd_download(argparse.Namespace(force=False)) != 0:
        return 1

    print("\n[2/6] ingest (deterministic subset selection)")
    icfg = IngestConfig(subset_size=args.subset, seed=args.seed)
    res = ingest_musique(icfg, paths)
    save_processed(res, paths)
    print(f"  {res.stats['n_questions']} questions, {res.stats['n_passages']} passages, "
          f"{res.stats['n_documents']} documents "
          f"({res.stats['selection']['n_dropped_single_doc']} single-document "
          f"questions excluded)")

    print("\n[3/6] build indexes")
    base_cfg = PRESETS["ours"]
    kb = KnowledgeBase.build(res, base_cfg, paths, verbose=True)
    kb.save(paths, base_cfg)
    kb.enable_score_cache(paths, base_cfg)

    print("\n[4/6] run systems")
    ecfg = EvalConfig(setting=args.setting, seed=args.seed)
    reader = None if args.no_reader else _reader_for(base_cfg)
    questions = res.questions[: args.limit] if args.limit else res.questions
    for name in ["closed_book", "bm25", "dense", "hybrid", "hybrid_rerank", "ours", "oracle"]:
        run_system(kb, PRESETS[name], questions, ecfg, paths, reader=reader,
                   setting=args.setting)

    if not args.skip_ablations:
        print("\n[5/6] ablations")
        # Same output directory and same questions as the runs above, so the
        # `ours`/`hybrid_rerank` references already exist -- do not re-run them.
        run_ablations(paths, ecfg, limit=args.limit, kb=kb, reader=reader,
                      reference=False)
    else:
        print("\n[5/6] ablations skipped (--skip-ablations)")

    print("\n[6/6] rebuild results from raw outputs")
    rep = build_report(paths, args.setting, ecfg=ecfg)
    print()
    print(rep["headline_table"])
    print()
    print(rep["comparison_table"])
    path = write_eval_md(paths, ecfg)
    print(f"\nEVAL.md -> {path}")
    print(f"\ntotal wall clock: {(time.time()-t_all)/60:.1f} min")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ctxlayer.cli",
        description="Knowledge base as a context layer (MuSiQue multi-hop).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch + verify the benchmark")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    for name, fn, helptext in [
        ("ingest", cmd_ingest, "build the processed corpus"),
        ("index", cmd_index, "build the indexes"),
    ]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--subset", type=int, default=500)
        s.add_argument("--seed", type=int, default=20251105)
        s.add_argument("--min-gold-docs", type=int, default=2, dest="min_gold_docs")
        s.add_argument("--no-stratify", action="store_true")
        s.set_defaults(func=fn)

    a = sub.add_parser("ask", help="answer one question with citations")
    a.add_argument("question", nargs="?", default="")
    a.add_argument("--qid", default=None, help="use a benchmark question by id")
    a.add_argument("--system", default="ours", choices=list(PRESETS))
    a.add_argument("--config", default=None, help="config file overriding --system")
    a.add_argument("--no-answer", action="store_true", help="context only")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="run systems -> raw outputs")
    e.add_argument("--systems", default=None, help="comma-separated (default: all)")
    e.add_argument("--setting", default="unified", choices=["unified", "pool"])
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--seed", type=int, default=20251105)
    e.add_argument("--no-reader", action="store_true", help="retrieval metrics only")
    e.add_argument("--out-label", default=None, dest="out_label",
                   help="raw-output directory label (default: the setting)")
    e.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablate", help="run the ablation suite")
    ab.add_argument("--setting", default="unified", choices=["unified", "pool"])
    ab.add_argument("--limit", type=int, default=None)
    ab.add_argument("--seed", type=int, default=20251105)
    ab.add_argument("--only", default=None)
    ab.add_argument("--list", action="store_true")
    ab.add_argument("--no-reader", action="store_true")
    ab.add_argument("--out-label", default=None, dest="out_label")
    ab.set_defaults(func=cmd_ablate)

    r = sub.add_parser("report", help="rebuild results from raw outputs")
    r.add_argument("--setting", default="unified", choices=["unified", "pool"])
    r.add_argument("--seed", type=int, default=20251105)
    r.add_argument("--write-md", action="store_true", help="also regenerate EVAL.md")
    r.add_argument("--out-label", default=None, dest="out_label")
    r.set_defaults(func=cmd_report)

    dg = sub.add_parser("diagnose", help="reproduce the design measurements")
    dg.add_argument("what", choices=["reachability", "corpus", "failures", "anchors"])
    dg.add_argument("--limit", type=int, default=None)
    dg.set_defaults(func=cmd_diagnose)

    rp = sub.add_parser("reproduce", help="full pipeline + report (one command)")
    rp.add_argument("--subset", type=int, default=500)
    rp.add_argument("--setting", default="unified", choices=["unified", "pool"])
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--seed", type=int, default=20251105)
    rp.add_argument("--no-reader", action="store_true")
    rp.add_argument("--skip-ablations", action="store_true")
    rp.set_defaults(func=cmd_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
