"""Figures of merit used throughout the study.

A positive prediction is a resistant antibiogram and a negative prediction is a
susceptible one. Accuracy, sensitivity and specificity are averaged into the
figure of merit that ranks features. Their standard deviation is subtracted
when the recursive selector scores a candidate number of features.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute loss, AUC and the three rates reported in the paper.

    Rates are fractions in ``[0, 1]``. The published tables print accuracy,
    sensitivity, specificity and AUC as percentages.
    """

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length.")
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "loss": binary_cross_entropy(y_true, y_prob) if total else float("nan"),
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def figure_of_merit(metrics: dict[str, float]) -> float:
    """Mean of accuracy, sensitivity and specificity."""

    values = [metrics["accuracy"], metrics["sensitivity"], metrics["specificity"]]
    return float(np.nanmean(values))


def performance_value(accuracy: float, sensitivity: float, specificity: float) -> float:
    """``mean(acc, sen, spe) - std(acc, sen, spe)`` for one feature-set size."""

    values = np.array([accuracy, sensitivity, specificity], dtype=float)
    return float(np.mean(values) - np.std(values))
