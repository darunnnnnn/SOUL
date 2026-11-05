"""Ingestion: MuSiQue-Answerable -> a unified, citable knowledge base.

Why this source material
------------------------
The assignment leaves the knowledge base scope to us. We chose MuSiQue-Ans
v1.0 (Trivedi et al., TACL 2022) because it is the open multi-hop benchmark
whose construction most directly stresses a *context layer* rather than a
reader:

* It is composed by **bottom-up assembly from single-hop sources**, and the
  authors explicitly filtered the result against single-hop shortcut models.
  HotpotQA, by contrast, is answerable from one paragraph far more often than
  its labels suggest; MuSiQue's reported "disconnected reasoning" headroom is
  dramatically lower. If the point is to prove a multi-document context layer
  works, the benchmark has to actually require multiple documents.
* Every question carries **paragraph-level supporting labels**, so support
  recall and multi-document coverage are directly measurable -- no proxy.
* It includes 3-hop and 4-hop questions, so the hop budget of an iterative
  retriever can be stress-tested rather than assumed.

What a "document" is
--------------------
Each MuSiQue paragraph carries a Wikipedia article ``title``. We treat **one
title = one document** and each paragraph as a passage of that document. This
matters for Hard Requirement #2: multi-document coverage is always counted over
distinct titles, so several paragraphs of the same article can never be mistaken
for multi-document evidence.

Two retrieval settings are built from the same ingest
-----------------------------------------------------
``pool``     Each question keeps its own 20-paragraph candidate pool. This is
             the standard setting every published MuSiQue number uses, so it
             keeps our comparisons legible against the literature.
``unified``  Every passage from every selected question is pooled into **one
             shared index** (~20k passages / ~17k documents). Distractors from
             other questions are now live, retrieval is genuinely open-domain
             over the knowledge base, and the task is much harder. This is the
             realistic deployment of a knowledge base, and it is our headline
             setting.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from ..config import IngestConfig, PathsConfig
from ..schema import Anchor, Document, Passage, Question, split_sentences, stable_id

__all__ = ["ingest_musique", "IngestResult", "load_processed", "save_processed"]

_HOP_COUNT = {"2hop": 2, "3hop1": 3, "3hop2": 3, "4hop1": 4, "4hop2": 4, "4hop3": 4}


class IngestResult:
    """Container for an ingested knowledge base + question set."""

    def __init__(
        self,
        passages: dict[str, Passage],
        documents: dict[str, Document],
        questions: list[Question],
        stats: dict[str, Any],
    ) -> None:
        self.passages = passages
        self.documents = documents
        self.questions = questions
        self.stats = stats

    @property
    def anchors(self) -> dict[str, Anchor]:
        return {a.anchor_id: a for p in self.passages.values() for a in p.anchors}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IngestResult(passages={len(self.passages)}, "
            f"documents={len(self.documents)}, questions={len(self.questions)})"
        )


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Conservative whitespace normalisation only.

    We deliberately do NOT strip punctuation, casing or the quoting style that
    MuSiQue inherits from Wikipedia: citations are verified by re-resolving
    character offsets into this exact string, so every transformation here is a
    transformation a citation must survive.
    """
    return " ".join(text.replace(" ", " ").split()).strip()


def _mint_passage(title: str, text: str, ord_: int, src: dict[str, Any]) -> Passage:
    doc_id = stable_id("doc", title)
    passage_id = stable_id("psg", title, text)
    passage = Passage(
        passage_id=passage_id,
        doc_id=doc_id,
        doc_title=title,
        passage_ord=ord_,
        text=text,
        source_ref=src,
    )
    for sent_idx, (cs, ce) in enumerate(split_sentences(text)):
        passage.anchors.append(
            Anchor(
                anchor_id=f"{doc_id}::p{ord_}::s{sent_idx}",
                passage_id=passage_id,
                doc_id=doc_id,
                doc_title=title,
                passage_ord=ord_,
                sent_idx=sent_idx,
                char_start=cs,
                char_end=ce,
                text=text[cs:ce],
            )
        )
    return passage


# --------------------------------------------------------------------------
# subset selection  (deterministic + documented, per Hard Requirement #4)
# --------------------------------------------------------------------------


def select_subset(records: list[dict[str, Any]], cfg: IngestConfig) -> list[dict[str, Any]]:
    """Deterministically select the evaluation subset.

    Selection is a pure function of (source file, ``subset_size``, ``seed``,
    ``min_gold_docs``, ``stratify_by_hops``) and is re-derived from scratch on
    every run -- nothing is cached, so the subset cannot silently drift.

    Two filters are applied, both in service of the hard requirements:

    1. ``min_gold_docs`` (default 2) drops questions whose gold supporting
       paragraphs all come from a *single* document. MuSiQue-Ans dev contains
       54 such questions. They are legitimate multi-hop questions, but the
       evidence lives in one source, so counting them would let a
       single-document answer score as a multi-document success. Hard
       Requirement #2 forbids exactly that.
    2. ``stratify_by_hops`` allocates the budget proportionally across hop
       types, then takes a seeded sample within each stratum sorted by ``id``.
       Without it the 2-hop majority (52% of dev) dominates and the hop budget
       of an iterative retriever is never tested.
    """
    eligible = []
    dropped_single_doc = 0
    for rec in records:
        gold_titles = {p["title"] for p in rec["paragraphs"] if p["is_supporting"]}
        if len(gold_titles) < cfg.min_gold_docs:
            dropped_single_doc += 1
            continue
        eligible.append(rec)

    if cfg.subset_size <= 0 or cfg.subset_size >= len(eligible):
        chosen = sorted(eligible, key=lambda r: r["id"])
    elif not cfg.stratify_by_hops:
        rng = random.Random(cfg.seed)
        chosen = sorted(rng.sample(sorted(eligible, key=lambda r: r["id"]), cfg.subset_size),
                        key=lambda r: r["id"])
    else:
        strata: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for rec in eligible:
            strata[rec["id"].split("__")[0]].append(rec)
        total = len(eligible)
        # Largest-remainder apportionment keeps the stratum proportions exact
        # and the total exactly equal to subset_size.
        quotas: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        for key, items in strata.items():
            exact = cfg.subset_size * len(items) / total
            quotas[key] = int(exact)
            remainders.append((exact - int(exact), key))
        short = cfg.subset_size - sum(quotas.values())
        for _, key in sorted(remainders, reverse=True)[:short]:
            quotas[key] += 1
        chosen = []
        for key in sorted(strata):
            items = sorted(strata[key], key=lambda r: r["id"])
            rng = random.Random(f"{cfg.seed}:{key}")
            take = min(quotas[key], len(items))
            chosen.extend(rng.sample(items, take))
        chosen.sort(key=lambda r: r["id"])

    select_subset.last_stats = {  # type: ignore[attr-defined]
        "n_source": len(records),
        "n_dropped_single_doc": dropped_single_doc,
        "n_eligible": len(eligible),
        "n_selected": len(chosen),
    }
    return chosen


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def ingest_musique(cfg: IngestConfig, paths: PathsConfig) -> IngestResult:
    src = paths.resolve("raw_dir") / cfg.source_file
    if not src.exists():
        raise FileNotFoundError(
            f"Raw benchmark file not found: {src}\n"
            "Run:  python -m ctxlayer.cli download"
        )

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    records = list(_iter_jsonl(src))
    chosen = select_subset(records, cfg)
    sel_stats = select_subset.last_stats  # type: ignore[attr-defined]

    # ---- build the unified passage store -------------------------------
    # Passages are content-addressed, so a paragraph shared by two questions
    # collapses to one knowledge-base entry (as it should -- it is one real
    # source location). passage_ord within a document is assigned by sorted
    # text hash: MuSiQue does not preserve in-article paragraph order, and a
    # deterministic surrogate keeps anchor ids stable across rebuilds.
    by_doc: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for rec in chosen:
        for para in rec["paragraphs"]:
            title, text = _clean(para["title"]), _clean(para["paragraph_text"])
            if not text or (title, text) in seen:
                continue
            seen.add((title, text))
            by_doc[title].append((hashlib.sha1(text.encode()).hexdigest(), text))

    passages: dict[str, Passage] = {}
    documents: dict[str, Document] = {}
    for title in sorted(by_doc):
        doc_id = stable_id("doc", title)
        doc = Document(doc_id=doc_id, title=title)
        for ord_, (_, text) in enumerate(sorted(by_doc[title])):
            psg = _mint_passage(title, text, ord_, {"dataset": cfg.dataset})
            passages[psg.passage_id] = psg
            doc.passage_ids.append(psg.passage_id)
        documents[doc_id] = doc

    # ---- build questions ------------------------------------------------
    questions: list[Question] = []
    for rec in chosen:
        pool_ids: list[str] = []
        gold_ids: list[str] = []
        gold_titles: list[str] = []
        for para in rec["paragraphs"]:
            title, text = _clean(para["title"]), _clean(para["paragraph_text"])
            if not text:
                continue
            pid = stable_id("psg", title, text)
            pool_ids.append(pid)
            if para["is_supporting"]:
                gold_ids.append(pid)
                gold_titles.append(title)
        hop_type = rec["id"].split("__")[0]
        questions.append(
            Question(
                qid=rec["id"],
                question=_clean(rec["question"]),
                answer=_clean(rec["answer"]),
                answer_aliases=[_clean(a) for a in rec.get("answer_aliases") or []],
                n_hops=_HOP_COUNT.get(hop_type, 2),
                hop_type=hop_type,
                gold_passage_ids=list(dict.fromkeys(gold_ids)),
                gold_doc_ids=list(dict.fromkeys(stable_id("doc", t) for t in gold_titles)),
                gold_doc_titles=list(dict.fromkeys(gold_titles)),
                pool_passage_ids=list(dict.fromkeys(pool_ids)),
                decomposition=rec.get("question_decomposition") or [],
            )
        )

    # ---- integrity checks (fail loud; these are correctness invariants) --
    for q in questions:
        missing = [p for p in q.gold_passage_ids if p not in passages]
        if missing:
            raise AssertionError(f"{q.qid}: gold passages not in corpus: {missing}")
        if q.n_gold_docs < cfg.min_gold_docs:
            raise AssertionError(
                f"{q.qid}: only {q.n_gold_docs} distinct gold docs "
                f"(min_gold_docs={cfg.min_gold_docs})"
            )

    hop_hist = collections.Counter(q.hop_type for q in questions)
    gold_doc_hist = collections.Counter(q.n_gold_docs for q in questions)
    stats: dict[str, Any] = {
        "dataset": "musique-ans-v1.0",
        "split": cfg.split,
        "source_file": cfg.source_file,
        "source_sha256": sha,
        "selection": sel_stats,
        "selection_params": {
            "subset_size": cfg.subset_size,
            "seed": cfg.seed,
            "min_gold_docs": cfg.min_gold_docs,
            "stratify_by_hops": cfg.stratify_by_hops,
        },
        "n_questions": len(questions),
        "n_passages": len(passages),
        "n_documents": len(documents),
        "n_anchors": sum(len(p.anchors) for p in passages.values()),
        "hop_type_hist": dict(sorted(hop_hist.items())),
        "gold_docs_per_question_hist": dict(sorted(gold_doc_hist.items())),
        "mean_passage_words": round(
            sum(p.n_words for p in passages.values()) / max(1, len(passages)), 2
        ),
        "mean_gold_passages": round(
            sum(len(q.gold_passage_ids) for q in questions) / max(1, len(questions)), 3
        ),
        "mean_gold_docs": round(
            sum(q.n_gold_docs for q in questions) / max(1, len(questions)), 3
        ),
    }
    return IngestResult(passages, documents, questions, stats)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_processed(res: IngestResult, paths: PathsConfig) -> Path:
    out = paths.resolve("processed_dir")
    out.mkdir(parents=True, exist_ok=True)

    with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for pid in sorted(res.passages):
            p = res.passages[pid]
            fh.write(
                json.dumps(
                    {
                        "passage_id": p.passage_id,
                        "doc_id": p.doc_id,
                        "doc_title": p.doc_title,
                        "passage_ord": p.passage_ord,
                        "text": p.text,
                        "anchors": [asdict(a) for a in p.anchors],
                        "source_ref": p.source_ref,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (out / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in sorted(res.questions, key=lambda x: x.qid):
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    manifest = dict(res.stats)
    manifest["corpus_sha256"] = hashlib.sha256(
        (out / "corpus.jsonl").read_bytes()
    ).hexdigest()
    manifest["questions_sha256"] = hashlib.sha256(
        (out / "questions.jsonl").read_bytes()
    ).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def load_processed(paths: PathsConfig) -> IngestResult:
    out = paths.resolve("processed_dir")
    man_path = out / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(
            f"No processed corpus at {out}. Run:  python -m ctxlayer.cli ingest"
        )
    passages: dict[str, Passage] = {}
    documents: dict[str, Document] = {}
    for row in _iter_jsonl(out / "corpus.jsonl"):
        anchors = [Anchor(**a) for a in row.pop("anchors")]
        p = Passage(**row, anchors=anchors)
        passages[p.passage_id] = p
        documents.setdefault(p.doc_id, Document(p.doc_id, p.doc_title)).passage_ids.append(
            p.passage_id
        )
    questions = [Question(**row) for row in _iter_jsonl(out / "questions.jsonl")]
    stats = json.loads(man_path.read_text(encoding="utf-8"))
    return IngestResult(passages, documents, questions, stats)
