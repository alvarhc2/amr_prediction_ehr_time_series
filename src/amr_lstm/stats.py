"""Paired bootstrap confidence interval used to compare AUC vectors.

The published comparison draws, with replacement, 70% of the paired
observations, repeats the draw 2,000 times, and builds a 95% interval for the
mean difference. A difference is treated as significant when the interval does
not contain zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    low: float
    high: float
    mean_difference: float

    @property
    def excludes_zero(self) -> bool:
        return self.high < 0.0 or self.low > 0.0


def paired_bootstrap_interval(
    reference: np.ndarray,
    other: np.ndarray,
    *,
    fraction: float = 0.7,
    repetitions: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """Interval for ``mean(reference - other)`` under paired resampling.

    ``reference`` and ``other`` are aligned repetitions, for example the 50
    test AUC values of two feature-selection methods.
    """

    left = np.asarray(reference, dtype=float).ravel()
    right = np.asarray(other, dtype=float).ravel()
    if left.shape != right.shape:
        raise ValueError("Bootstrap vectors must be paired and have the same length.")
    if left.size == 0:
        raise ValueError("Bootstrap vectors must be non-empty.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    sample_size = max(1, int(round(left.size * fraction)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, left.size, size=(repetitions, sample_size))
    differences = left[draws].mean(axis=1) - right[draws].mean(axis=1)
    low, high = np.quantile(differences, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        low=float(low),
        high=float(high),
        mean_difference=float(differences.mean()),
    )
