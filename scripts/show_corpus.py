"""Show the knowledge base: where the documents came from, how they were split.

Usage:
    python scripts/show_corpus.py                       # the whole knowledge base
    python scripts/show_corpus.py --titles                    # a random sample of titles
    python scripts/show_corpus.py --qid 2hop__130668_126089   # one question's documents
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def load(name: str) -> list[dict]:
    with (PROC / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def show_provenance(man: dict) -> None:
    rule("1. WHERE THE DOCUMENTS COME FROM")
    src = RAW / man["source_file"]
    size_mb = src.stat().st_size / 1e6 if src.exists() else float("nan")
    print(f"  dataset        : {man['dataset']}  ({man['split']})")
    print(f"  file on disk   : data/raw/{man['source_file']}  ({size_mb:.1f} MB)")
    print(f"  sha256         : {man['source_sha256']}")
    print()
    print("  The corpus is NOT hand-picked. Every document is a Wikipedia")
    print("  paragraph shipped inside MuSiQue, a published multi-hop QA")
    print("  benchmark. Pinning the sha256 means anyone re-running this")
    print("  gets byte-identical documents or a loud failure.")

    sel = man["selection"]
    print()
    print("  How 500 questions were chosen from the file:")
    print(f"    {sel['n_source']:>5}  questions in the dev split")
    print(f"  - {sel['n_dropped_single_doc']:>5}  dropped: all gold evidence sits in ONE document")
    print("           (several chunks of one source is not multi-document evidence)")
    print(f"    {sel['n_eligible']:>5}  eligible")
    print(f"    {sel['n_selected']:>5}  selected, seed {man['selection_params']['seed']}, stratified by hop count")


def show_split(man: dict, corpus: list[dict]) -> None:
    rule("2. HOW THE CONTENT IS SPLIT")
    per_doc = Counter(p["doc_id"] for p in corpus)
    multi = sum(1 for n in per_doc.values() if n > 1)
    print("  The documents those 500 questions reference, after ingestion:")
    print()
    print(f"    {man['n_documents']:>6}  documents   (one Wikipedia article each, by title)")
    print(f"    {man['n_passages']:>6}  passages    (a document split into retrievable chunks)")
    print(f"    {man['n_anchors']:>6}  anchors     (one sentence each, with byte offsets)")
    print()
    print(f"  mean passage length : {man['mean_passage_words']:.1f} words")
    print(f"  documents split into more than one passage : {multi}")
    print()
    print("  Three levels, and each one has a job:")
    print("    document  ->  what 'multi-document evidence' is counted over")
    print("    passage   ->  what retrieval scores and ranks")
    print("    anchor    ->  what a citation points at, verified byte-exact")
    print()
    print(f"  corpus_sha256    : {man['corpus_sha256']}")
    print(f"  questions_sha256 : {man['questions_sha256']}")
    print("  (the evaluator refuses to run if either changes)")


def show_titles(corpus: list[dict], k: int = 12, seed: int = 2026) -> None:
    rule("A RANDOM SAMPLE OF THE DOCUMENTS")
    docs = {}
    for p in corpus:
        docs.setdefault(p["doc_id"], p["doc_title"])
    titles = sorted(docs.values())
    print(f"  {len(titles)} documents, {len(set(titles))} distinct titles -- every document")
    print("  is a separate Wikipedia article. They are not chapters of one")
    print("  source and they share no common subject:")
    print()
    for t in random.Random(seed).sample(titles, k):
        print(f"    - {t}")
    print()
    print("  Nothing in the corpus says which of these belong together.")
    print("  There is no link structure to follow and no table of contents.")
    print("  Connections have to be discovered from the text itself.")


def show_examples(corpus: list[dict], n: int = 2) -> None:
    rule("3. WHAT ONE DOCUMENT ACTUALLY LOOKS LIKE")
    by_doc: dict[str, list[dict]] = {}
    for p in corpus:
        by_doc.setdefault(p["doc_id"], []).append(p)
    shown = 0
    for doc_id, passages in by_doc.items():
        if len(passages) < 2:
            continue
        passages.sort(key=lambda p: p["passage_ord"])
        print()
        print(f"  DOCUMENT  {passages[0]['doc_title']!r}   doc_id={doc_id}")
        print(f"            split into {len(passages)} passages")
        for p in passages:
            print(f"    passage ord={p['passage_ord']}  id={p['passage_id']}  "
                  f"{len(p['text'].split())} words  {len(p['anchors'])} anchors")
            for a in p["anchors"][:2]:
                print(f"        anchor {a['anchor_id']}  chars {a['char_start']}-{a['char_end']}")
                print(f"               {a['text'][:92]!r}")
        shown += 1
        if shown >= n:
            break


def show_question(corpus: list[dict], questions: list[dict], qid: str) -> None:
    q = next((x for x in questions if x["qid"] == qid), None)
    if q is None:
        print(f"  no question with qid {qid!r}")
        return
    by_doc = {p["doc_id"]: p for p in corpus}
    rule("THE DOCUMENTS BEHIND ONE QUESTION")
    print(f"  qid      : {q['qid']}")
    print(f"  question : {q['question']}")
    print(f"  answer   : {q['answer']}")
    print(f"  hops     : {q['n_hops']}  ({q['hop_type']})")
    print()
    print(f"  Retrieval searches all {len(corpus)} passages. Of those,")
    print(f"  {len(q['gold_doc_ids'])} documents are required to answer, and they are:")
    for doc_id, title in zip(q["gold_doc_ids"], q["gold_doc_titles"]):
        p = by_doc.get(doc_id)
        print()
        print(f"    [GOLD] {title!r}  doc_id={doc_id}")
        if p is not None:
            print(f"           {p['text'][:300]}")
    print()
    print("  The question names only one of these two titles. The second")
    print("  document has to be found from something written inside the first.")
    print("  That is the whole problem this system exists to solve.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="show the documents for one question")
    ap.add_argument("--titles", action="store_true", help="print a random sample of document titles")
    args = ap.parse_args()

    man = json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))
    corpus = load("corpus.jsonl")

    if args.qid:
        show_question(corpus, load("questions.jsonl"), args.qid)
        return 0

    if args.titles:
        show_titles(corpus)
        print()
        return 0

    show_provenance(man)
    show_split(man, corpus)
    show_titles(corpus)
    show_examples(corpus)
    print()
    print("  Verify every anchor re-resolves to byte-identical source text:")
    print("      python -m ctxlayer.cli diagnose anchors")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
