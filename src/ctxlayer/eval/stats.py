"""Significance testing for the results table.

A bare point estimate on a 500-question subset is not evidence. Two systems
differing by 1.5 points of recall may be indistinguishable, and reporting that
as an improvement would be exactly the kind of unsupported claim this
assignment warns against. So every headline comparison in EVAL.md ships with a
confidence interval and a p-value.

Choices, and why
----------------
**Paired, not unpaired.** Every system is evaluated on the *same* questions, so
the per-question difference ``a_i - b_i`` is the right unit of analysis. Pairing
removes question difficulty -- by far the largest source of variance -- and is
dramatically more sensitive than comparing two independent means.

**Permutation, not a t-test.** Our metrics are bounded and mostly binary
(``support_all@k``, ``all_docs@k`` are 0/1; EM is 0/1). Their differences are
trinary {-1, 0, +1} and nowhere near normal, so a t-test's assumptions fail.
An exact-in-the-limit paired permutation test assumes only exchangeability of
the labels under the null, which holds by construction: if the two systems were
equivalent, which of the pair produced which score is arbitrary. This is the
standard test in IR evaluation (Smucker et al., CIKM 2007, who show it agrees
with the randomisation ideal where the t-test does not).

**Bootstrap for intervals.** Percentile bootstrap over questions, resampled
with replacement, gives the interval for a single system's mean; the paired
bootstrap over per-question differences gives the interval for a delta. No
distributional assumption either way.

**Multiple comparisons.** We compare our system against several baselines on
several metrics, so ``holm_bonferroni`` is provided and EVAL.md reports
adjusted p-values alongside raw ones. Without it, testing 6 baselines x 5
metrics would be expected to produce a "significant" result by chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "bootstrap_ci",
    "paired_bootstrap_delta",
    "paired_permutation_test",
    "holm_bonferroni",
    "Comparison",
    "cohens_d_paired",
]


def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 20251105,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns ``(mean, lo, hi)``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 20251105,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for ``mean(a) - mean(b)``.

    Questions (not scores) are resampled, keeping each question's pair of
    scores together -- that is what makes the interval reflect the paired
    design.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"paired test needs equal lengths: {x.size} vs {y.size}")
    if x.size == 0:
        return 0.0, 0.0, 0.0
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_samples, d.size))
    means = d[idx].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_samples: int = 10000,
    seed: int = 20251105,
) -> float:
    """Two-sided paired permutation (randomisation) test p-value.

    Under the null the two systems are equivalent, so for each question the
    sign of ``a_i - b_i`` is arbitrary; we resample signs and count how often
    the permuted mean difference is at least as extreme as observed. The ``+1``
    in numerator and denominator is the standard Monte-Carlo correction, which
    keeps the p-value from ever being exactly 0 (we cannot prove a difference
    with finite resampling).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"paired test needs equal lengths: {x.size} vs {y.size}")
    d = x - y
    nz = d[d != 0.0]
    if nz.size == 0:
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    # Only non-zero differences can change the statistic; zeros contribute a
    # constant 0 regardless of sign, so permuting them wastes work. The mean is
    # taken over the FULL sample size to keep the statistic comparable.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, nz.size))
    perm = np.abs((signs * nz).sum(axis=1) / d.size)
    return float((np.sum(perm >= observed - 1e-12) + 1) / (n_samples + 1))


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size, for reporting alongside p-values."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 1e-12 else 0.0


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control).

    Uniformly more powerful than plain Bonferroni at the same guarantee.
    Adjusted values are made monotone non-decreasing in rank order, as the
    procedure requires, and clipped at 1.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p_values[i]
        running = max(running, val)
        adjusted[i] = min(1.0, running)
    return adjusted


@dataclass
class Comparison:
    """One system-vs-baseline comparison on one metric."""

    metric: str
    system: str
    baseline: str
    system_mean: float
    baseline_mean: float
    delta: float
    delta_lo: float
    delta_hi: float
    p_value: float
    p_adjusted: float = float("nan")
    effect_size: float = 0.0
    n: int = 0

    @property
    def significant(self) -> bool:
        p = self.p_adjusted if not math.isnan(self.p_adjusted) else self.p_value
        return p < 0.05

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "system": self.system,
            "baseline": self.baseline,
            "system_mean": round(self.system_mean, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "delta": round(self.delta, 4),
            "delta_ci95": [round(self.delta_lo, 4), round(self.delta_hi, 4)],
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size_d": round(self.effect_size, 3),
            "n": self.n,
            "significant": self.significant,
        }


def compare(
    metric: str,
    system: str,
    baseline: str,
    sys_vals: Sequence[float],
    base_vals: Sequence[float],
    bootstrap_samples: int = 2000,
    permutation_samples: int = 10000,
    seed: int = 20251105,
) -> Comparison:
    delta, lo, hi = paired_bootstrap_delta(
        sys_vals, base_vals, n_samples=bootstrap_samples, seed=seed
    )
    p = paired_permutation_test(
        sys_vals, base_vals, n_samples=permutation_samples, seed=seed
    )
    return Comparison(
        metric=metric,
        system=system,
        baseline=baseline,
        system_mean=float(np.mean(sys_vals)) if len(sys_vals) else 0.0,
        baseline_mean=float(np.mean(base_vals)) if len(base_vals) else 0.0,
        delta=delta,
        delta_lo=lo,
        delta_hi=hi,
        p_value=p,
        effect_size=cohens_d_paired(sys_vals, base_vals),
        n=len(sys_vals),
    )
