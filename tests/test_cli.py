"""CLI surface: every subcommand must parse, and defaults must not drift.

Cheap but worth having -- a broken argument wiring makes the repository
unrunnable, which is the one failure mode that invalidates everything else.
These tests never execute a command, only parse it.
"""

from __future__ import annotations

import re

import pytest

from ctxlayer.cli import MUSIQUE_SHA256, build_parser, reader_identity
from ctxlayer.config import PRESETS


@pytest.mark.parametrize(
    "argv",
    [
        ["download"],
        ["download", "--force"],
        ["ingest"],
        ["ingest", "--subset", "150", "--seed", "7", "--min-gold-docs", "3"],
        ["ingest", "--no-stratify"],
        ["index"],
        ["ask", "who is x?"],
        ["ask", "--qid", "2hop__1_2", "--system", "hybrid_rerank"],
        ["ask", "q", "--no-answer"],
        ["eval"],
        ["eval", "--systems", "bm25,ours", "--setting", "pool", "--limit", "10"],
        ["eval", "--no-reader", "--out-label", "custom"],
        ["ablate"],
        ["ablate", "--list"],
        ["ablate", "--only", "no_iterative,no_assembly", "--limit", "50"],
        ["report"],
        ["report", "--setting", "pool", "--write-md"],
        ["diagnose", "reachability"],
        ["diagnose", "corpus"],
        ["diagnose", "anchors"],
        ["diagnose", "failures"],
        ["reproduce"],
        ["reproduce", "--subset", "100", "--skip-ablations", "--no-reader"],
    ],
)
def test_subcommands_parse(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_unknown_subcommand_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["frobnicate"])


def test_invalid_setting_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--setting", "nonsense"])


def test_invalid_diagnostic_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["diagnose", "nonsense"])


def test_ask_rejects_unknown_system():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "q", "--system", "not_a_system"])


def test_defaults_are_the_documented_ones():
    a = build_parser().parse_args(["eval"])
    assert a.setting == "unified"       # the harder, headline setting
    assert a.seed == 20251105
    assert a.limit is None
    assert a.no_reader is False
    b = build_parser().parse_args(["ingest"])
    assert b.subset == 500
    assert b.min_gold_docs == 2         # Hard Requirement #2


def test_benchmark_checksum_is_pinned():
    assert len(MUSIQUE_SHA256) == 64
    assert MUSIQUE_SHA256 == (
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
    )


def test_every_ask_system_choice_is_a_real_preset():
    for action in build_parser()._subparsers._group_actions[0].choices["ask"]._actions:
        if action.dest == "system":
            assert set(action.choices) == set(PRESETS)


# --------------------------------------------------------------------------
# Reader identity
#
# Regression guard for a silent-mislabelling bug: `cmd_eval` built one reader
# from PRESETS["ours"] and handed it to every system, while each run header
# recorded that system's *configured* reader. The llm-reader presets were
# therefore evaluated by the extractive reader and written to disk with
# `"kind": "llm"` in the header -- artefacts that named a generator they had
# not used. Nothing failed; the numbers were simply wrong about themselves.
# --------------------------------------------------------------------------


def test_llm_and_extractive_presets_do_not_share_a_reader():
    """Different generators must not collapse onto one instance."""
    assert reader_identity(PRESETS["ours"]) != reader_identity(PRESETS["ours_llm"])


@pytest.mark.parametrize(
    "a,b",
    [
        ("ours", "hybrid_rerank"),
        ("ours", "oracle"),
        ("ours_llm", "hybrid_rerank_llm"),
        ("ours_llm", "oracle_llm"),
        ("ours_llm", "closed_book_llm"),
    ],
)
def test_systems_compared_in_one_table_share_a_reader(a, b):
    """Systems in the same table must be read by the same generator.

    This is what makes an answer-level delta attributable to retrieval.
    """
    assert reader_identity(PRESETS[a]) == reader_identity(PRESETS[b])


def test_llm_presets_pin_an_exact_model():
    """An alias silently changes; a published number must name its model."""
    for name in ("closed_book_llm", "hybrid_rerank_llm", "ours_llm", "oracle_llm"):
        cfg = PRESETS[name]
        assert cfg.reader.kind == "llm"
        assert cfg.reader.llm_temperature == 0.0
        # A dated snapshot, e.g. gpt-4.1-mini-2025-04-14.
        assert re.search(r"-\d{4}-\d{2}-\d{2}$", cfg.reader.llm_model), (
            f"{name} must pin a dated model snapshot, got "
            f"{cfg.reader.llm_model!r}"
        )


def test_llm_presets_differ_from_their_base_only_in_the_reader():
    """The LLM table must vary the reader, not the retrieval being measured."""
    for llm_name, base_name in [
        ("ours_llm", "ours"),
        ("hybrid_rerank_llm", "hybrid_rerank"),
        ("closed_book_llm", "closed_book"),
        ("oracle_llm", "oracle"),
    ]:
        llm, base = PRESETS[llm_name], PRESETS[base_name]
        assert llm.mode == base.mode
        assert llm.iterative.enabled == base.iterative.enabled
        assert llm.rerank.enabled == base.rerank.enabled
        assert llm.assemble.enabled == base.assemble.enabled
