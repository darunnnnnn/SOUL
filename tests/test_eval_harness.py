"""Statistics, score cache, and the report's refusal to build bad tables.

The significance machinery decides which claims we are allowed to make, so it
is tested against cases with known answers: identical systems must not look
different, pure noise must not look significant, and a real effect must be
detected. The report's validation guards are tested too, because a results
table silently assembled from mismatched runs is worse than no table.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ctxlayer.config import PathsConfig, RerankConfig
from ctxlayer.eval.report import RunSet, Run, load_runs
from ctxlayer.eval.stats import (
    bootstrap_ci,
    cohens_d_paired,
    compare,
    holm_bonferroni,
    paired_bootstrap_delta,
    paired_permutation_test,
)
from ctxlayer.index.score_cache import ScoreCache


# --------------------------------------------------------------------------
# significance testing
# --------------------------------------------------------------------------


def test_identical_systems_are_not_significant():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 300).astype(float)
    assert paired_permutation_test(a, a.copy()) == 1.0
    d, lo, hi = paired_bootstrap_delta(a, a.copy())
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(7)
    x, y = rng.random(400), rng.random(400)
    assert paired_permutation_test(x, y) > 0.05


def test_real_effect_is_detected_with_ci_excluding_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 400).astype(float)
    b = a.copy()
    b[rng.random(400) < 0.25] = 0.0          # degrade b only
    p = paired_permutation_test(a, b)
    d, lo, hi = paired_bootstrap_delta(a, b)
    assert p < 0.01
    assert d > 0 and lo > 0                   # CI excludes zero


def test_permutation_p_value_is_never_zero():
    """Finite resampling cannot prove a difference; p must stay > 0."""
    a = np.ones(200)
    b = np.zeros(200)
    p = paired_permutation_test(a, b, n_samples=1000)
    assert 0.0 < p <= 1.0
    assert p == pytest.approx(1 / 1001)


def test_permutation_is_symmetric():
    rng = np.random.default_rng(3)
    a, b = rng.random(200), rng.random(200)
    assert paired_permutation_test(a, b, seed=5) == paired_permutation_test(
        b, a, seed=5
    )


def test_permutation_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1.0, 2.0], [1.0])


def test_all_ties_give_p_one():
    assert paired_permutation_test([1.0] * 50, [1.0] * 50) == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(2)
    v = rng.normal(0.5, 0.2, 500)
    m, lo, hi = bootstrap_ci(v, seed=11)
    assert lo < m < hi
    assert (m, lo, hi) == bootstrap_ci(v, seed=11)        # same seed, same answer


def test_bootstrap_ci_degenerate_inputs():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)


def test_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(4)
    small = bootstrap_ci(rng.normal(0, 1, 30), seed=1)
    large = bootstrap_ci(rng.normal(0, 1, 3000), seed=1)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_holm_is_monotone_and_at_least_bonferroni_powerful():
    raw = [0.001, 0.02, 0.04, 0.3]
    adj = holm_bonferroni(raw)
    # monotone non-decreasing in rank order
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    vals = [adj[i] for i in order]
    assert vals == sorted(vals)
    # never smaller than the raw value, never larger than Bonferroni
    for r, a in zip(raw, adj):
        assert a >= r - 1e-12
        assert a <= min(1.0, len(raw) * r) + 1e-12
    assert holm_bonferroni([]) == []


def test_holm_clips_at_one():
    assert all(p <= 1.0 for p in holm_bonferroni([0.6, 0.7, 0.9]))


def test_effect_size_sign_and_zero_variance():
    assert cohens_d_paired([1.0] * 10, [0.0] * 10) == 0.0   # no variance in diff
    rng = np.random.default_rng(9)
    a = rng.random(100) + 0.3
    b = rng.random(100)
    assert cohens_d_paired(a, b) > 0
    assert cohens_d_paired(b, a) < 0


def test_compare_packages_everything():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2, 200).astype(float)
    b = a.copy()
    b[rng.random(200) < 0.3] = 0.0
    c = compare("ctx_all_docs", "ours", "bm25", a, b)
    assert c.n == 200
    assert c.delta > 0
    assert c.significant is True
    d = c.to_dict()
    assert d["metric"] == "ctx_all_docs" and "delta_ci95" in d


# --------------------------------------------------------------------------
# score cache
# --------------------------------------------------------------------------


def test_score_cache_roundtrip_and_isolation(tmp_path):
    c = ScoreCache(tmp_path / "s.sqlite", "model-a", 320)
    k = c.key("query one", "passage text")
    assert c.get_many([k]) == {}
    c.put_many({k: 1.25})
    assert c.get_many([k])[k] == pytest.approx(1.25)

    # A different model or truncation length must MISS, not silently reuse.
    other_model = ScoreCache(tmp_path / "s.sqlite", "model-b", 320)
    assert other_model.get_many([other_model.key("query one", "passage text")]) == {}
    other_len = ScoreCache(tmp_path / "s.sqlite", "model-a", 128)
    assert other_len.get_many([other_len.key("query one", "passage text")]) == {}
    c.close()


def test_score_cache_key_depends_on_query_and_passage(tmp_path):
    c = ScoreCache(tmp_path / "s.sqlite", "m", 320)
    assert c.key("a", "x") != c.key("b", "x")
    assert c.key("a", "x") != c.key("a", "y")
    assert c.key("a", "x") == c.key("a", "x")
    c.close()


def test_score_cache_handles_more_keys_than_sqlite_variable_limit(tmp_path):
    """Chunking must work: SQLite caps bound variables at 999 by default."""
    c = ScoreCache(tmp_path / "s.sqlite", "m", 320)
    items = {c.key(f"q{i}", "p"): float(i) for i in range(2500)}
    c.put_many(items)
    got = c.get_many(list(items))
    assert len(got) == 2500
    assert got[c.key("q1234", "p")] == pytest.approx(1234.0)
    c.close()


def test_score_cache_disabled_is_a_noop():
    c = ScoreCache(None, "m", 320)
    assert c.enabled is False
    assert c.get_many(["anything"]) == {}
    c.put_many({"a": 1.0})          # must not raise
    assert c.stats()["cache_hit_rate"] == 0.0


def test_score_cache_tracks_hit_rate(tmp_path):
    c = ScoreCache(tmp_path / "s.sqlite", "m", 320)
    c.put_many({c.key("a", "p"): 1.0})
    c.get_many([c.key("a", "p"), c.key("b", "p")])
    s = c.stats()
    assert s["cache_hits"] == 1.0 and s["cache_misses"] == 1.0
    assert s["cache_hit_rate"] == pytest.approx(0.5)
    c.close()


# --------------------------------------------------------------------------
# report validation guards
# --------------------------------------------------------------------------


def _run(name, qids, reader="deepset/roberta-base-squad2", fp="abc", em=1.0):
    return Run(
        system=name,
        description="",
        header={
            "corpus": {"questions_sha256": fp},
            "reader": {"model": reader},
        },
        records=[
            {
                "qid": q,
                "hop_type": "2hop",
                "n_hops": 2,
                "context_ms": 1.0,
                "trace": {"n_queries": 1},
                "metrics": {"ctx_all_docs": 1.0, "answer_em": em},
            }
            for q in qids
        ],
    )


def test_runset_accepts_aligned_runs():
    rs = RunSet("unified", {"ours": _run("ours", ["a", "b"]),
                            "bm25": _run("bm25", ["a", "b"])})
    assert rs.n == 2
    assert rs.runs["ours"].series("ctx_all_docs") == [1.0, 1.0]


def test_runset_rejects_mismatched_question_sets():
    """Paired tests are invalid across different question sets."""
    with pytest.raises(ValueError, match="question-set mismatch"):
        RunSet("unified", {"ours": _run("ours", ["a", "b"]),
                           "bm25": _run("bm25", ["a", "c"])})


def test_runset_rejects_mismatched_corpus_fingerprint():
    with pytest.raises(ValueError, match="corpus fingerprint"):
        RunSet("unified", {"ours": _run("ours", ["a"], fp="one"),
                           "bm25": _run("bm25", ["a"], fp="two")})


def test_runset_rejects_answer_comparison_across_readers():
    """Answer-level deltas are only meaningful with a fixed reader."""
    with pytest.raises(ValueError, match="different readers"):
        RunSet("unified", {"ours": _run("ours", ["a"], reader="model-x"),
                           "bm25": _run("bm25", ["a"], reader="model-y")})


def test_runset_rejects_duplicate_qids():
    with pytest.raises(ValueError, match="duplicate qids"):
        RunSet("unified", {"ours": _run("ours", ["a", "a"])})


def test_runset_rejects_empty():
    with pytest.raises(ValueError, match="no runs found"):
        RunSet("unified", {})


def test_comparisons_apply_holm_across_the_whole_family():
    rs = RunSet("unified", {"ours": _run("ours", list("abcdefghij"), em=1.0),
                            "bm25": _run("bm25", list("abcdefghij"), em=0.0)})
    comps = rs.comparisons("ours", ["bm25"], metrics=("answer_em",))
    assert len(comps) == 1
    assert not math.isnan(comps[0].p_adjusted)


def test_comparisons_raise_for_unknown_system():
    rs = RunSet("unified", {"ours": _run("ours", ["a"])})
    with pytest.raises(KeyError):
        rs.comparisons("nope", ["ours"])
