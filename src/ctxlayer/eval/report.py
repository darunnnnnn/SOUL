"""Report builder: raw per-question outputs -> tables, intervals, p-values.

This module never retrieves, never calls a model, and never touches the corpus.
It reads the committed JSONL artefacts in ``eval/raw_outputs/`` and recomputes
every published number. That is the whole point: ``ctxlayer report`` satisfies
"one command must rebuild the results from raw outputs", and it does so in
seconds, offline, on any machine.

It also *validates* the artefacts before trusting them, because a results table
silently built from mismatched runs is worse than no table:

* every system must cover the identical question set (paired tests are invalid
  otherwise, and the runner's progress could have been interrupted);
* every system must report the same corpus fingerprint;
* reader identity is compared across systems, since answer-level comparisons
  are only meaningful with a fixed reader.

Any violation is raised, not warned about.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import EvalConfig, PathsConfig
from .stats import Comparison, bootstrap_ci, compare, holm_bonferroni

__all__ = ["RunSet", "load_runs", "build_report", "markdown_table"]


# Metrics promoted to the headline table, in display order.
HEADLINE = [
    ("ctx_support_all", "SupAll"),
    ("ctx_all_docs", "AllDocs"),
    ("ctx_support_recall", "SupRec"),
    ("ctx_doc_coverage", "DocCov"),
    ("answer_em", "EM"),
    ("answer_f1", "F1"),
    ("ctx_words", "Words"),
    ("ctx_n_docs", "#Docs"),
]

# Metrics for which we run significance tests against each baseline.
TESTED = [
    "ctx_support_all",
    "ctx_all_docs",
    "ctx_support_recall",
    "ctx_doc_coverage",
    "answer_em",
    "answer_f1",
]

LOWER_IS_BETTER = {"ctx_words", "context_ms", "n_citations"}


@dataclass
class Run:
    system: str
    description: str
    header: dict[str, Any]
    records: list[dict[str, Any]]

    @property
    def qids(self) -> list[str]:
        return [r["qid"] for r in self.records]

    def series(self, metric: str) -> list[float]:
        return [float(r["metrics"].get(metric, 0.0)) for r in self.records]

    def has(self, metric: str) -> bool:
        return bool(self.records) and metric in self.records[0]["metrics"]


class RunSet:
    """All systems for one retrieval setting, aligned on question id."""

    def __init__(
        self,
        setting: str,
        runs: dict[str, Run],
        allow_mixed_commits: bool = False,
        holdout_skip: int = 0,
    ) -> None:
        """``holdout_skip`` drops the first N questions (by sorted qid).

        Hyperparameters were chosen with ``scripts/ab_assembly.py`` on
        ``questions[:150]``, which is the first 150 qids in sorted order. Those
        questions are therefore *development* data, and reporting a headline on
        them is mild overfitting. Dropping them yields a split that was never
        used for any tuning decision, at no extra compute: the raw outputs
        already cover every question, so the held-out table is a filter over
        committed artefacts rather than another run.

        EVAL.md reports both, labelled, so a reader can see the size of the
        effect rather than take our word that it is small.
        """
        self.setting = setting
        self.allow_mixed_commits = allow_mixed_commits
        self.holdout_skip = holdout_skip
        if holdout_skip > 0:
            skipped = sorted({q for r in runs.values() for q in r.qids})[:holdout_skip]
            drop = set(skipped)
            runs = {
                n: Run(
                    system=r.system,
                    description=r.description,
                    header=r.header,
                    records=[x for x in r.records if x["qid"] not in drop],
                )
                for n, r in runs.items()
            }
        self.runs = runs
        self._validate()

    def _validate(self) -> None:
        if not self.runs:
            raise ValueError(f"no runs found for setting {self.setting!r}")
        names = list(self.runs)
        ref = self.runs[names[0]]
        ref_qids = ref.qids
        if len(set(ref_qids)) != len(ref_qids):
            raise ValueError(f"{ref.system}: duplicate qids in raw output")
        for name in names[1:]:
            other = self.runs[name]
            if other.qids != ref_qids:
                missing = set(ref_qids) ^ set(other.qids)
                raise ValueError(
                    f"question-set mismatch between {ref.system!r} and {name!r} "
                    f"({len(missing)} differing qids). Paired comparisons require "
                    f"identical question sets -- re-run both systems."
                )
        fps = {
            n: r.header.get("corpus", {}).get("questions_sha256")
            for n, r in self.runs.items()
        }
        distinct = {v for v in fps.values() if v}
        if len(distinct) > 1:
            raise ValueError(f"runs disagree on corpus fingerprint: {fps}")
        # Code version. Runs produced by different commits are not comparable:
        # a change to the reader or the assembler silently shifts one row of
        # the table relative to the others. This caught exactly that mistake
        # once -- baselines run before a reader fix, compared against a system
        # run after it -- so it is an error, not a warning.
        commits = {
            n: (r.header.get("environment") or {}).get("git_commit")
            for n, r in self.runs.items()
        }
        distinct_commits = {c for c in commits.values() if c and c != "unknown"}
        if len(distinct_commits) > 1 and not self.allow_mixed_commits:
            lines = "\n".join(
                f"    {n}: {c}" for n, c in sorted(commits.items())
            )
            raise ValueError(
                "runs were produced by different code versions, so they are "
                "not comparable:\n" + lines + "\n"
                "Re-run every system on one commit (`ctxlayer eval`), or pass "
                "allow_mixed_commits=True if the difference is known to be "
                "irrelevant."
            )
        readers = {
            n: (r.header.get("reader") or {}).get("model")
            for n, r in self.runs.items()
            if r.has("answer_em")
        }
        if len({v for v in readers.values() if v}) > 1:
            raise ValueError(
                f"answer metrics compared across different readers: {readers}. "
                "The reader must be held fixed."
            )
        self.qids = ref_qids
        self.n = len(ref_qids)

    # ------------------------------------------------------------ aggregates
    def summary(self, metrics: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, run in self.runs.items():
            keys = metrics or sorted(run.records[0]["metrics"]) if run.records else []
            row: dict[str, Any] = {"_description": run.description}
            for m in keys:
                if not run.has(m):
                    continue
                vals = run.series(m)
                mean, lo, hi = bootstrap_ci(vals, n_samples=1000)
                row[m] = {"mean": mean, "ci95": [lo, hi]}
            row["_latency_ms_median"] = (
                statistics.median([r["context_ms"] for r in run.records])
                if run.records
                else 0.0
            )
            row["_n_queries_mean"] = (
                statistics.mean([r["trace"]["n_queries"] for r in run.records])
                if run.records
                else 0.0
            )
            out[name] = row
        return out

    def by_hop(self, metric: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, run in self.runs.items():
            if not run.has(metric):
                continue
            groups: dict[str, list[float]] = defaultdict(list)
            for r in run.records:
                groups[f"{r['n_hops']}hop"].append(float(r["metrics"].get(metric, 0.0)))
            out[name] = {k: statistics.mean(v) for k, v in sorted(groups.items())}
            out[name]["all"] = statistics.mean(run.series(metric))
        return out

    def comparisons(
        self,
        system: str,
        baselines: Sequence[str],
        metrics: Sequence[str] = tuple(TESTED),
        ecfg: EvalConfig | None = None,
    ) -> list[Comparison]:
        ecfg = ecfg or EvalConfig()
        if system not in self.runs:
            raise KeyError(f"system {system!r} not in runs: {list(self.runs)}")
        sys_run = self.runs[system]
        out: list[Comparison] = []
        for metric in metrics:
            if not sys_run.has(metric):
                continue
            for b in baselines:
                if b not in self.runs or not self.runs[b].has(metric):
                    continue
                out.append(
                    compare(
                        metric,
                        system,
                        b,
                        sys_run.series(metric),
                        self.runs[b].series(metric),
                        bootstrap_samples=ecfg.bootstrap_samples,
                        permutation_samples=ecfg.permutation_samples,
                        seed=ecfg.seed,
                    )
                )
        # Family-wise correction over the whole set of comparisons reported
        # together -- not per metric, which would under-correct.
        adj = holm_bonferroni([c.p_value for c in out])
        for c, p in zip(out, adj):
            c.p_adjusted = p
        return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_runs(
    paths: PathsConfig,
    setting: str,
    systems: Iterable[str] | None = None,
    holdout_skip: int = 0,
) -> RunSet:
    base = paths.resolve("raw_outputs_dir") / setting
    if not base.exists():
        raise FileNotFoundError(
            f"No raw outputs at {base}. Run:  python -m ctxlayer.cli eval"
        )
    runs: dict[str, Run] = {}
    wanted = set(systems) if systems else None
    for f in sorted(base.glob("*.jsonl")):
        name = f.stem
        if wanted is not None and name not in wanted:
            continue
        header: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("_record") == "header":
                    header = row
                else:
                    records.append(row)
        if not records:
            continue
        runs[name] = Run(name, header.get("description", ""), header, records)
    return RunSet(setting, runs, holdout_skip=holdout_skip)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _fmt(metric: str, value: float) -> str:
    if metric in {"ctx_words", "ctx_n_docs", "ctx_n_passages", "n_citations"}:
        return f"{value:.1f}"
    return f"{value * 100:.1f}"


def markdown_table(
    rs: RunSet,
    order: Sequence[str],
    columns: Sequence[tuple[str, str]] = tuple(HEADLINE),
    with_ci: bool = False,
) -> str:
    summary = rs.summary([m for m, _ in columns])
    present = [(m, label) for m, label in columns if any(m in summary[s] for s in order if s in summary)]
    head = "| System | " + " | ".join(label for _, label in present) + " |"
    sep = "|---" * (len(present) + 1) + "|"
    lines = [head, sep]
    for name in order:
        if name not in summary:
            continue
        cells = []
        for m, _ in present:
            cell = summary[name].get(m)
            if cell is None:
                cells.append("--")
            elif with_ci:
                lo, hi = cell["ci95"]
                cells.append(f"{_fmt(m, cell['mean'])} [{_fmt(m, lo)}, {_fmt(m, hi)}]")
            else:
                cells.append(_fmt(m, cell["mean"]))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_table(comps: Sequence[Comparison]) -> str:
    lines = [
        "| Metric | vs baseline | Baseline | Ours | Delta | 95% CI | p (raw) | p (Holm) | d |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comps:
        star = " **" if c.significant else " "
        lines.append(
            f"| {c.metric} | `{c.baseline}` | {c.baseline_mean*100:.1f} | "
            f"{c.system_mean*100:.1f} | {c.delta*100:+.1f}{star} | "
            f"[{c.delta_lo*100:+.1f}, {c.delta_hi*100:+.1f}] | "
            f"{c.p_value:.4f} | {c.p_adjusted:.4f} | {c.effect_size:.2f} |"
        )
    return "\n".join(lines)


def build_report(
    paths: PathsConfig,
    setting: str,
    system: str = "ours",
    # closed_book is included so the table shows, with the same statistics as
    # every other row, that the benchmark is not answerable without retrieval.
    baselines: Sequence[str] = (
        "closed_book", "bm25", "dense", "hybrid", "hybrid_rerank",
    ),
    order: Sequence[str] | None = None,
    ecfg: EvalConfig | None = None,
    holdout_skip: int = 0,
) -> dict[str, Any]:
    """Recompute every aggregate for one setting and persist it."""
    ecfg = ecfg or EvalConfig()
    rs = load_runs(paths, setting, holdout_skip=holdout_skip)
    default_order = [
        "closed_book", "bm25", "dense", "hybrid", "hybrid_rerank", "ours", "oracle",
    ]
    order = order or [s for s in default_order if s in rs.runs] + [
        s for s in sorted(rs.runs) if s not in default_order
    ]
    present_baselines = [b for b in baselines if b in rs.runs]
    comps = (
        rs.comparisons(system, present_baselines, ecfg=ecfg)
        if system in rs.runs
        else []
    )
    report = {
        "setting": setting,
        "holdout_skip": holdout_skip,
        "n_questions": rs.n,
        "systems": list(rs.runs),
        "order": order,
        "summary": rs.summary(),
        "by_hop": {m: rs.by_hop(m) for m in TESTED if any(r.has(m) for r in rs.runs.values())},
        "comparisons": [c.to_dict() for c in comps],
        "headline_table": markdown_table(rs, order),
        "headline_table_ci": markdown_table(rs, order, with_ci=True),
        "comparison_table": comparison_table(comps),
        "corpus": next(iter(rs.runs.values())).header.get("corpus", {}),
        "environment": next(iter(rs.runs.values())).header.get("environment", {}),
    }
    out_dir = paths.resolve("results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_holdout{holdout_skip}" if holdout_skip else ""
    (out_dir / f"report_{setting}{suffix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
