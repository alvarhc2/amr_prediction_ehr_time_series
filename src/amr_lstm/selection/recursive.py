"""Recursive feature-number selection.

Given the validation accuracy, sensitivity and specificity obtained with the
first ``k`` features, for every ``k``, the selector keeps splitting the range
of ``k`` and retains the half with the higher mean performance value

``PV = mean(accuracy, sensitivity, specificity) - std(accuracy, sensitivity, specificity)``.

It stops when the surviving interval is shorter than ``threshold`` and returns
the right edge of that interval, which is the number of features to keep.

The research scripts scored an interval with ``mean([mean(PV inputs), 1 - mean(std)])``.
That expression is an increasing affine transform of the mean performance value
above, so it selects the same edge.
"""

from __future__ import annotations

import numpy as np


def recursive_feature_count(
    accuracy: np.ndarray,
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    *,
    threshold: int = 10,
) -> int:
    """Return how many leading features to keep.

    ``accuracy[k]`` is the validation accuracy obtained when the model is
    trained on the ``k + 1`` most important features.
    """

    accuracy = np.asarray(accuracy, dtype=float).ravel()
    sensitivity = np.asarray(sensitivity, dtype=float).ravel()
    specificity = np.asarray(specificity, dtype=float).ravel()
    if not (accuracy.shape == sensitivity.shape == specificity.shape):
        raise ValueError("Metric curves must have the same length.")
    if accuracy.size == 0:
        raise ValueError("Metric curves must be non-empty.")
    if threshold < 2:
        raise ValueError("threshold must be at least 2.")

    mean = np.nanmean(np.vstack([accuracy, sensitivity, specificity]), axis=0)
    std = np.nanstd(np.vstack([accuracy, sensitivity, specificity]), axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)

    def select(start: int, stop: int) -> int:
        length = stop - start
        if length < threshold:
            return stop
        mid = start + length // 2
        left = float(np.mean(mean[start:mid] - std[start:mid]))
        right = float(np.mean(mean[mid:stop] - std[mid:stop]))
        if left >= right:
            return select(start, mid)
        return select(mid, stop)

    return int(select(0, accuracy.size))
