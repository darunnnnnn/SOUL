"""The reader cache must be a pure speed optimisation.

It memoises the dominant cost of the evaluation, so the one thing that must be
beyond doubt is that it cannot change a result. These tests pin that:

* a cached answer equals the uncached answer field for field;
* the key covers every input the reader is a function of, so changing any of
  them MISSES rather than returning a stale answer -- in particular the passage
  *scores*, which drive the passage-selection prior and are easy to forget;
* a corrupt cache row behaves as a miss, never as a wrong answer.

The model is loaded once here (roberta-base-squad2 is already required by the
rest of the pipeline), so these are real end-to-end checks rather than mocks.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from ctxlayer.config import PRESETS
from ctxlayer.index.score_cache import BlobCache
from ctxlayer.schema import Anchor, AssembledContext, Passage, split_sentences, stable_id
from ctxlayer.synth.reader import ExtractiveReader

QUESTION = "Where was James Watt educated?"


def _mk(title: str, text: str) -> Passage:
    doc_id = stable_id("doc", title)
    p = Passage(
        passage_id=stable_id("psg", title, text),
        doc_id=doc_id, doc_title=title, passage_ord=0, text=text,
    )
    for i, (cs, ce) in enumerate(split_sentences(text)):
        p.anchors.append(
            Anchor(f"{doc_id}::p0::s{i}", p.passage_id, doc_id, title, 0, i,
                   cs, ce, text[cs:ce])
        )
    return p


@pytest.fixture(scope="module")
def ctx() -> AssembledContext:
    a = _mk(
        "Lap Engine",
        "The Lap Engine is a beam engine designed by James Watt, built by "
        "Boulton and Watt in 1788. It is preserved at the Science Museum.",
    )
    b = _mk(
        "James Watt",
        "While working as an instrument maker at the University of Glasgow, "
        "Watt became interested in the technology of steam engines.",
    )
    return AssembledContext(passages=[a, b], scores=[1.0, 0.62])


@pytest.fixture(scope="module")
def cfg():
    return PRESETS["ours"].reader


def _fields(ans):
    return (
        ans.answer,
        ans.abstained,
        [(c.text, tuple(c.anchor_ids)) for c in ans.claims],
        round(float(ans.meta.get("margin", 0.0)), 4),
        ans.meta.get("passage_id"),
    )


def test_cached_answer_is_identical_to_uncached(tmp_path, cfg, ctx):
    plain = ExtractiveReader(cfg)
    first = plain.read(QUESTION, ctx)

    cached = ExtractiveReader(
        cfg, cache=BlobCache(tmp_path / "r.sqlite", "reader-v1")
    )
    miss = cached.read(QUESTION, ctx)      # populates
    hit = cached.read(QUESTION, ctx)       # serves from cache

    assert _fields(first) == _fields(miss) == _fields(hit)
    assert miss.meta.get("cached") is None
    assert hit.meta.get("cached") is True
    # Only the first call reached the model.
    assert cached.n_read == 1
    assert cached.cache.hits == 1 and cached.cache.misses == 1


def test_answer_is_nonempty_and_cites_a_real_anchor(cfg, ctx):
    """Sanity: the fixture must actually exercise a successful read."""
    ans = ExtractiveReader(cfg).read(QUESTION, ctx)
    assert ans.answer.strip()
    assert ans.claims and ans.claims[0].anchor_ids
    aid = ans.claims[0].anchor_ids[0]
    known = {a.anchor_id for p in ctx.passages for a in p.anchors}
    assert aid in known


def test_key_depends_on_scores(tmp_path, cfg, ctx):
    """Scores drive the passage prior, so they must be part of the key."""
    cache = BlobCache(tmp_path / "r.sqlite", "reader-v1")
    r = ExtractiveReader(cfg, cache=cache)
    k1 = r._cache_key(QUESTION, ctx, "unknown")
    flipped = AssembledContext(passages=ctx.passages, scores=[0.62, 1.0])
    k2 = r._cache_key(QUESTION, flipped, "unknown")
    assert k1 != k2


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "per_passage"),
        ("max_length", 256),
        ("stride", 64),
        ("max_answer_words", 4),
        ("passage_selection_weight", 0.0),
        ("model", "some/other-model"),
    ],
)
def test_key_depends_on_every_reader_input(tmp_path, cfg, ctx, field, value):
    cache = BlobCache(tmp_path / "r.sqlite", "reader-v1")
    base = ExtractiveReader(cfg, cache=cache)
    k1 = base._cache_key(QUESTION, ctx, "unknown")
    import dataclasses

    other = ExtractiveReader(dataclasses.replace(cfg, **{field: value}), cache=cache)
    assert other._cache_key(QUESTION, ctx, "unknown") != k1


def test_key_depends_on_question_and_focus(tmp_path, cfg, ctx):
    cache = BlobCache(tmp_path / "r.sqlite", "reader-v1")
    r = ExtractiveReader(cfg, cache=cache)
    k = r._cache_key(QUESTION, ctx, "unknown")
    assert r._cache_key("A different question?", ctx, "unknown") != k
    assert r._cache_key(QUESTION, ctx, "date") != k


def test_no_cache_means_no_key(cfg, ctx):
    r = ExtractiveReader(cfg)
    assert r._cache_key(QUESTION, ctx, "unknown") is None
    disabled = ExtractiveReader(cfg, cache=BlobCache(None, "reader-v1"))
    assert disabled._cache_key(QUESTION, ctx, "unknown") is None


def test_corrupt_row_behaves_as_a_miss(tmp_path, cfg, ctx):
    path = tmp_path / "r.sqlite"
    cache = BlobCache(path, "reader-v1")
    r = ExtractiveReader(cfg, cache=cache)
    key = r._cache_key(QUESTION, ctx, "unknown")
    cache.put(key, {"answer": "x", "claims": []})
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE blobs SET v = ? WHERE k = ?", ("{not json", key))
    conn.commit()
    conn.close()
    fresh = BlobCache(path, "reader-v1")
    assert fresh.get(key) is None           # miss, not a crash
    assert fresh.misses == 1


def test_empty_context_short_circuits_before_the_cache(tmp_path, cfg):
    cache = BlobCache(tmp_path / "r.sqlite", "reader-v1")
    r = ExtractiveReader(cfg, cache=cache)
    ans = r.read(QUESTION, AssembledContext())
    assert ans.abstained and ans.answer == ""
    assert r.n_read == 0
    assert cache.hits == 0 and cache.misses == 0
