"""CLI surface: every subcommand must parse, and defaults must not drift.

Cheap but worth having -- a broken argument wiring makes the repository
unrunnable, which is the one failure mode that invalidates everything else.
These tests never execute a command, only parse it.
"""

from __future__ import annotations

import pytest

from ctxlayer.cli import MUSIQUE_SHA256, build_parser
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
