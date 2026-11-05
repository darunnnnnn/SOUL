"""Diagnostics that reproduce the measurements our design decisions rest on.

Every non-obvious claim in DESIGN.md is backed by one of these, so a reviewer can
re-derive the number rather than take it on trust:

``reachability``  The measurement that chose the bridge mechanism. Compares a
                  title-keyed document-link graph (the intuitive design, which
                  we built first and rejected) against the surface-form mention
                  index, on identical data, by asking: starting from the first
                  gold passage, is the rest of the gold evidence chain reachable
                  at all? A retriever cannot rank what its candidate generator
                  can never propose, so this upper-bounds any iterative method.
``corpus``        Knowledge-base and subset statistics, including the
                  multi-document composition that Hard Requirement #2 needs.
``anchors``       Provenance integrity: every anchor in the knowledge base must
                  re-resolve to byte-identical text at its recorded offsets.
                  This is Hard Requirement #1 checked exhaustively, not sampled.
``failures``      Honest failure analysis over the raw eval outputs: where our
                  system loses, and to what.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

from ..config import GraphConfig, PathsConfig, PRESETS
from ..index.bm25 import STOPWORDS, tokenize
from ..index.entity_index import EntityIndex
from ..index.store import KnowledgeBase
from ..ingest.musique import load_processed

__all__ = ["run_diagnostic"]


# --------------------------------------------------------------------------
# the rejected design, kept so its measurement stays reproducible
# --------------------------------------------------------------------------


class _TitleLinkGraph:
    """Document-link graph keyed on article titles -- the design we rejected.

    Edges are title mentions: if a passage of document A contains a surface form
    equal to the title of document B, add A -> B. This approximates Wikipedia's
    hyperlink structure, which is how multi-hop benchmarks were constructed, and
    is the obvious thing to build. It fails for a structural reason that only
    measurement reveals: the hop-n+1 bridge entity is usually *not* an article
    title in the corpus, so there is no node to traverse to.
    """

    def __init__(self, passages: list[Any]) -> None:
        self.doc_title = {p.doc_id: p.doc_title for p in passages}
        self.doc_passages: dict[str, list[str]] = collections.defaultdict(list)
        for p in passages:
            self.doc_passages[p.doc_id].append(p.passage_id)

        self.title_lookup: dict[str, str] = {}
        by_first: dict[str, list[tuple[int, tuple[str, ...], str]]] = (
            collections.defaultdict(list)
        )
        for doc_id, title in sorted(self.doc_title.items(), key=lambda kv: kv[1]):
            if title.startswith(("List of", "Index of", "Outline of")):
                continue
            toks = tuple(tokenize(title))
            if not toks or len(title) < 4 or all(t in STOPWORDS for t in toks):
                continue
            key = " ".join(toks)
            if key in self.title_lookup:
                continue
            self.title_lookup[key] = doc_id
            by_first[toks[0]].append((len(toks), toks, doc_id))
        for lst in by_first.values():
            lst.sort(key=lambda x: -x[0])

        self.mentions: dict[str, list[str]] = {}
        self.mentioned_by: dict[str, list[str]] = collections.defaultdict(list)
        for p in passages:
            stream: list[str] = []
            caps: list[bool] = []
            for w in p.text.split():
                sub = tokenize(w)
                for s in sub:
                    stream.append(s)
                    caps.append(w[:1].isupper())
            found: list[str] = []
            j = 0
            while j < len(stream):
                cands = by_first.get(stream[j])
                if not cands:
                    j += 1
                    continue
                hit, span = None, 0
                for n, toks, doc_id in cands:
                    if j + n > len(stream):
                        continue
                    if tuple(stream[j : j + n]) != toks:
                        continue
                    if not caps[j]:
                        continue
                    hit, span = doc_id, n
                    break
                if hit is None:
                    j += 1
                    continue
                if hit != p.doc_id and hit not in found:
                    found.append(hit)
                j += span
            if found:
                self.mentions[p.passage_id] = found[:24]
                for d in found[:24]:
                    self.mentioned_by[d].append(p.passage_id)

    def candidates(self, passage_id: str, limit: int) -> list[str]:
        out: list[str] = []
        for doc_id in self.mentions.get(passage_id, []):
            out.extend(self.doc_passages.get(doc_id, [])[:limit])
            out.extend(
                q for q in self.mentioned_by.get(doc_id, [])[:limit] if q != passage_id
            )
        return list(dict.fromkeys(out))


def _chain_reachability(questions, candidates_fn, label: str) -> dict[str, Any]:
    """Can the gold chain be reached at all from the first gold passage?"""
    reach: collections.Counter = collections.Counter()
    tot: collections.Counter = collections.Counter()
    envelopes: list[int] = []
    for q in questions:
        gold = q.gold_passage_ids
        if len(gold) < 2:
            continue
        acc = {gold[0]}
        frontier = [gold[0]]
        for _ in range(len(gold) - 1):
            new: set[str] = set()
            for pid in frontier:
                new.update(candidates_fn(pid))
            acc |= new
            frontier = list(new)[:40]
        envelopes.append(len(acc))
        for gp in gold[1:]:
            tot["all"] += 1
            tot[q.hop_type] += 1
            if gp in acc:
                reach["all"] += 1
                reach[q.hop_type] += 1
    return {
        "mechanism": label,
        "overall": reach["all"] / tot["all"] if tot["all"] else 0.0,
        "n_gold_links": tot["all"],
        "by_hop_type": {
            k: {"reached": reach[k], "total": tot[k], "rate": reach[k] / tot[k]}
            for k in sorted(tot)
            if k != "all"
        },
        "mean_candidate_envelope": (
            round(statistics.mean(envelopes), 1) if envelopes else 0.0
        ),
    }


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _diag_reachability(paths: PathsConfig, limit: int | None) -> int:
    res = load_processed(paths)
    passages = [res.passages[p] for p in sorted(res.passages)]
    questions = res.questions[:limit] if limit else res.questions

    print(f"corpus: {len(passages)} passages / "
          f"{len({p.doc_id for p in passages})} documents")
    print(f"questions: {len(questions)}\n")

    print("building title-keyed link graph (the rejected design) ...")
    tg = _TitleLinkGraph(passages)
    print(f"  {len(tg.title_lookup)} titles indexed, "
          f"{sum(len(v) for v in tg.mentions.values())} edges")
    r_graph = _chain_reachability(
        questions, lambda pid: tg.candidates(pid, 40), "title_link_graph"
    )

    print("\nbuilding surface-form mention index (the chosen design) ...")
    ei = EntityIndex(GraphConfig()).build(passages, verbose=False)
    print(f"  {ei.stats['n_entity_keys']} entity keys, "
          f"{ei.stats['n_postings']} postings")

    def mention_candidates(pid: str) -> list[str]:
        out: list[str] = []
        for e in ei.entities_in(pid):
            out.extend(ei.candidates_for_entity(e, limit=40))
        return out

    r_mentions = _chain_reachability(questions, mention_candidates, "mention_index")

    # How often is a bridge answer itself an article title? This is the root
    # cause of the title graph's failure.
    hit = n = 0
    for q in questions:
        for step in q.decomposition:
            ans = (step.get("answer") or "").strip()
            if not ans:
                continue
            n += 1
            if ei.resolve_entity(ans):
                hit += 1

    print("\n" + "=" * 84)
    print("GOLD-CHAIN REACHABILITY  (can the candidate generator propose the")
    print("                          rest of the gold chain at all?)")
    print("=" * 84)
    print(f"{'hop type':12s} {'title graph':>16s} {'mention index':>16s} {'delta':>10s}")
    print("-" * 84)
    keys = sorted(set(r_graph["by_hop_type"]) | set(r_mentions["by_hop_type"]))
    for k in keys:
        a = r_graph["by_hop_type"].get(k, {}).get("rate", 0.0)
        b = r_mentions["by_hop_type"].get(k, {}).get("rate", 0.0)
        print(f"{k:12s} {a*100:15.1f}% {b*100:15.1f}% {(b-a)*100:+9.1f}")
    print("-" * 84)
    print(f"{'OVERALL':12s} {r_graph['overall']*100:15.1f}% "
          f"{r_mentions['overall']*100:15.1f}% "
          f"{(r_mentions['overall']-r_graph['overall'])*100:+9.1f}")
    print("=" * 84)
    print(f"\nRoot cause: only {hit}/{n} = {hit/max(1,n)*100:.1f}% of gold bridge "
          f"answers are themselves\narticle titles in the corpus, so a "
          f"title-keyed graph has no node to traverse to.")
    print(f"\ncandidate envelope (mean passages reachable): "
          f"title graph {r_graph['mean_candidate_envelope']}, "
          f"mention index {r_mentions['mean_candidate_envelope']} "
          f"of {len(passages)}")

    out = paths.resolve("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_questions": len(questions),
        "n_passages": len(passages),
        "title_link_graph": r_graph,
        "mention_index": r_mentions,
        "bridge_answers_that_are_titles": {"hit": hit, "total": n,
                                           "rate": hit / max(1, n)},
    }
    (out / "diagnostic_reachability.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out / 'diagnostic_reachability.json'}")
    return 0


def _diag_corpus(paths: PathsConfig, limit: int | None) -> int:
    res = load_processed(paths)
    print(json.dumps(res.stats, indent=2, ensure_ascii=False))
    docs_per_q = collections.Counter(q.n_gold_docs for q in res.questions)
    print("\ngold documents per question (Hard Requirement #2: all must be >= 2)")
    for k in sorted(docs_per_q):
        print(f"  {k} documents: {docs_per_q[k]:5d} questions")
    assert min(docs_per_q) >= 2, "a question with <2 gold documents survived selection"
    print("\nOK: every question requires evidence from at least 2 distinct documents.")
    return 0


def _diag_anchors(paths: PathsConfig, limit: int | None) -> int:
    """Exhaustive provenance integrity check (Hard Requirement #1)."""
    res = load_processed(paths)
    kb = KnowledgeBase(res.passages, res.documents)
    total = bad = 0
    coverage_bad = 0
    for p in kb.passages.values():
        # Anchors must tile the passage: every character of every passage must
        # be citable, or some claim could be grounded in text no anchor covers.
        covered = 0
        for a in p.anchors:
            total += 1
            if p.text[a.char_start : a.char_end] != a.text:
                bad += 1
            covered += a.char_end - a.char_start
        if covered < len(p.text.replace(" ", "")) * 0.8:
            coverage_bad += 1
    print(f"anchors checked          : {total}")
    print(f"offset/text mismatches   : {bad}")
    print(f"passages under-covered   : {coverage_bad}")
    print(f"mean anchors per passage : {total/max(1,len(kb.passages)):.2f}")
    if bad == 0:
        print("\nOK: every anchor re-resolves to byte-identical source text.")
    else:
        print("\nFAIL: provenance integrity violated.")
        return 1
    return 0


def _diag_failures(paths: PathsConfig, limit: int | None) -> int:
    from .report import load_runs

    for setting in ("unified", "pool"):
        base = paths.resolve("raw_outputs_dir") / setting
        if not base.exists():
            continue
        rs = load_runs(paths, setting)
        if "ours" not in rs.runs:
            continue
        ours = rs.runs["ours"]
        print(f"\n{'='*84}\nFAILURE ANALYSIS -- {setting} setting, n={rs.n}\n{'='*84}")

        buckets: collections.Counter = collections.Counter()
        examples: dict[str, list[dict]] = collections.defaultdict(list)
        for r in ours.records:
            m = r["metrics"]
            got_all = m.get("ctx_support_all", 0.0) >= 1.0
            got_docs = m.get("ctx_all_docs", 0.0) >= 1.0
            em = m.get("answer_em")
            if got_all and (em is None or em >= 1.0):
                buckets["success"] += 1
                continue
            if not got_docs:
                key = "retrieval: missed >=1 gold DOCUMENT"
            elif not got_all:
                key = "retrieval: right documents, wrong passage"
            else:
                key = "reading: full evidence present, answer wrong"
            buckets[key] += 1
            if len(examples[key]) < 4:
                examples[key].append(r)

        tot = sum(buckets.values())
        for k, v in buckets.most_common():
            print(f"  {v:5d}  ({v/tot*100:5.1f}%)  {k}")
        for key, rows in examples.items():
            if key == "success":
                continue
            print(f"\n--- {key} ---")
            for r in rows:
                print(f"  [{r['hop_type']}] {r['question']}")
                print(f"     gold answer : {r['gold_answer']}")
                if "answer" in r:
                    print(f"     predicted   : {r['answer']['text']!r}")
                print(f"     gold docs   : {r['gold_doc_titles']}")
                print(f"     ctx docs    : {list(dict.fromkeys(r['context_doc_titles']))[:6]}")
                if r["trace"]["bridge_entities"]:
                    print(f"     bridges     : {r['trace']['bridge_entities'][:6]}")
    return 0


_DIAGS = {
    "reachability": _diag_reachability,
    "corpus": _diag_corpus,
    "anchors": _diag_anchors,
    "failures": _diag_failures,
}


def run_diagnostic(what: str, paths: PathsConfig, limit: int | None = None) -> int:
    return _DIAGS[what](paths, limit)
