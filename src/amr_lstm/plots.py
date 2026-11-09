"""Plots of the feature-selection curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amr_lstm.training import PerformanceCurve


def plot_selection_curve(curve: PerformanceCurve, n_features: int, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    k = np.arange(1, curve.accuracy.size + 1)
    figure, axis = plt.subplots(figsize=(7.4, 4.3), dpi=140)
    axis.plot(k, curve.accuracy, color="#0f6e56", linewidth=2.0, label="Accuracy")
    axis.plot(k, curve.sensitivity, color="#c2410c", linewidth=2.0, label="Sensitivity")
    axis.plot(k, curve.specificity, color="#1d4e89", linewidth=2.0, label="Specificity")
    axis.axvline(n_features, color="#9a3412", linestyle="--", linewidth=1.4, label=f"Selected ({n_features})")
    axis.set_xlabel("Number of features, in importance order")
    axis.set_ylabel("Validation score")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(1, max(1, k[-1]))
    axis.legend(frameon=False, ncol=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
