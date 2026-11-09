"""Gradient, random-subset and permutation importance, averaged after min-max scaling."""

from __future__ import annotations

import numpy as np

from amr_lstm.metrics import classification_metrics, figure_of_merit
from amr_lstm.model import evaluate_model, gradient_sensitivity, predict_proba, train_lstm


def _train(x_train, y_train, x_val, y_val, *, units, epochs, seed, dropout, learning_rate, decay, batch_size):
    return train_lstm(
        x_train,
        y_train,
        x_val,
        y_val,
        units=units,
        epochs=epochs,
        seed=seed,
        dropout=dropout,
        learning_rate=learning_rate,
        decay=decay,
        batch_size=batch_size,
    )


def gradient_feature_importance(
    feature_names: tuple[str, ...] | list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    units: int,
    epochs: int,
    repetitions: int,
    seed: int = 0,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Sensitivity analysis averaged over validation cultures and repetitions.

    Each repetition is weighted by its validation figure of merit. Within a
    culture the gradient is averaged over time and then squared, as in the
    published algorithm.
    """

    if len(y_val) == 0:
        raise ValueError("Gradient importance needs a non-empty validation set.")
    rng = np.random.default_rng(seed)
    names = list(feature_names)
    accumulated = np.zeros((repetitions, len(names)), dtype=float)
    weights = np.zeros(repetitions, dtype=float)
    for repetition in range(repetitions):
        model = _train(
            x_train,
            y_train,
            x_val,
            y_val,
            units=units,
            epochs=epochs,
            seed=int(rng.integers(0, 1_000_000_000)),
            dropout=dropout,
            learning_rate=learning_rate,
            decay=decay,
            batch_size=batch_size,
        )
        sensitivities = np.mean([gradient_sensitivity(model, sequence) for sequence in x_val], axis=0)
        accumulated[repetition] = sensitivities
        weights[repetition] = figure_of_merit(evaluate_model(model, x_val, y_val))
    weights = np.nan_to_num(weights, nan=0.0)
    if float(weights.sum()) <= 0.0:
        weights[:] = 1.0
    weights = weights / weights.sum()
    scores = weights @ accumulated
    ranking = list(zip(names, scores.tolist()))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def random_subset_feature_importance(
    feature_names: tuple[str, ...] | list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    units: int,
    epochs: int,
    n_subsets: int,
    subset_size: int,
    seed: int = 0,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Mean validation merit of the random subsets that contain each feature.

    Subsets are drawn without replacement. The published run uses 500 subsets
    of 40 features so that each column is scored about 50 times.
    """

    names = list(feature_names)
    n_features = len(names)
    if subset_size > n_features:
        raise ValueError(f"subset_size ({subset_size}) exceeds the number of features ({n_features}).")
    if subset_size < 1 or n_subsets < 1:
        raise ValueError("subset_size and n_subsets must be positive.")
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_features, dtype=float)
    counts = np.zeros(n_features, dtype=float)
    for _ in range(n_subsets):
        chosen = rng.choice(n_features, size=subset_size, replace=False)
        model = _train(
            x_train[:, :, chosen],
            y_train,
            x_val[:, :, chosen],
            y_val,
            units=units,
            epochs=epochs,
            seed=int(rng.integers(0, 1_000_000_000)),
            dropout=dropout,
            learning_rate=learning_rate,
            decay=decay,
            batch_size=batch_size,
        )
        merit = figure_of_merit(evaluate_model(model, x_val[:, :, chosen], y_val))
        if np.isnan(merit):
            continue
        totals[chosen] += merit
        counts[chosen] += 1.0
    if np.any(counts == 0):
        floor = float(np.min(totals[counts > 0])) if np.any(counts > 0) else 0.0
        totals[counts == 0] = floor
        counts[counts == 0] = 1.0
    scores = totals / counts
    ranking = list(zip(names, scores.tolist()))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def permutation_feature_importance(
    feature_names: tuple[str, ...] | list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    units: int,
    epochs: int,
    repetitions: int,
    seed: int = 0,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Drop in validation merit after the entries of one feature are shuffled.

    The shuffle mixes cultures and time steps of that column and leaves every
    other column unchanged. The network is trained once per repetition.
    """

    names = list(feature_names)
    rng = np.random.default_rng(seed)
    accumulated = np.zeros((repetitions, len(names)), dtype=float)
    for repetition in range(repetitions):
        model = _train(
            x_train,
            y_train,
            x_val,
            y_val,
            units=units,
            epochs=epochs,
            seed=int(rng.integers(0, 1_000_000_000)),
            dropout=dropout,
            learning_rate=learning_rate,
            decay=decay,
            batch_size=batch_size,
        )
        baseline = figure_of_merit(classification_metrics(y_val, predict_proba(model, x_val)))
        for feature in range(len(names)):
            shuffled = np.array(x_val, copy=True)
            column = shuffled[:, :, feature].reshape(-1).copy()
            rng.shuffle(column)
            shuffled[:, :, feature] = column.reshape(shuffled.shape[0], shuffled.shape[1])
            permuted = figure_of_merit(classification_metrics(y_val, predict_proba(model, shuffled)))
            accumulated[repetition, feature] = baseline - permuted
    scores = np.nanmean(accumulated, axis=0)
    ranking = list(zip(names, scores.tolist()))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def _minmax(ranking: list[tuple[str, float]]) -> dict[str, float]:
    values = np.asarray([score for _, score in ranking], dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return {name: 0.0 for name, _ in ranking}
    return {name: (score - low) / (high - low) for name, score in ranking}


def average_importance(*rankings: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max each method to ``[0, 1]`` and average the three scores."""

    if len(rankings) != 3:
        raise ValueError("The study averages the three feature-importance methods.")
    scaled = [_minmax(ranking) for ranking in rankings]
    names = [name for name, _ in rankings[0]]
    if any(set(item) != set(names) for item in scaled):
        raise ValueError("The three rankings must contain the same features.")
    averaged = [(name, float(np.mean([item[name] for item in scaled]))) for name in names]
    averaged.sort(key=lambda item: item[1], reverse=True)
    return averaged
