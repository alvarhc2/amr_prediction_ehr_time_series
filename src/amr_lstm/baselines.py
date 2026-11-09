"""Comparison selectors: bootstrap voting across time, and group lasso.

Bootstrap voting follows Martínez-Agüero et al. (2022), which this paper cites
for the temporal vote. For every feature and every day, a bootstrap confidence
interval is built for the difference between the mean of resistant cultures and
the mean of susceptible cultures. A day is eligible when both classes have at
least two observed values; don't-care entries (−1) and padding are skipped.
The day votes for the feature when zero falls outside the interval. The feature
is kept when it wins a majority of the eligible days. Resampling uses the
budget stated in this paper: 2,000 draws of 70% of each class and a 95% interval.

Group lasso fits a logistic model whose coefficients are grouped by feature:
the 30 time steps of one column are one group and are shrunk together. The
penalty weight is chosen on the validation figure of merit.
"""

from __future__ import annotations

import warnings

import numpy as np

from amr_lstm.metrics import classification_metrics, figure_of_merit


def bootstrap_voting_select(
    x: np.ndarray,
    y: np.ndarray,
    *,
    repetitions: int = 2000,
    fraction: float = 0.7,
    alpha: float = 0.05,
    vote_fraction: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Return indices of features kept by the temporal bootstrap vote."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must describe the same cultures.")
    _, timesteps, n_features = x.shape
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for feature in range(n_features):
        votes = 0
        eligible = 0
        for step in range(timesteps):
            column = x[:, step, feature]
            observed = column >= 0.0
            resistant = column[(y == 1) & observed]
            susceptible = column[(y == 0) & observed]
            if resistant.size < 2 or susceptible.size < 2:
                continue
            eligible += 1
            if _means_differ(resistant, susceptible, rng, repetitions, fraction, alpha):
                votes += 1
        if eligible and votes > vote_fraction * eligible:
            selected.append(feature)
    return np.asarray(selected, dtype=int)


def _means_differ(
    resistant: np.ndarray,
    susceptible: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
    fraction: float,
    alpha: float,
) -> bool:
    if resistant.size < 2 or susceptible.size < 2:
        return False
    n_resistant = max(1, int(round(resistant.size * fraction)))
    n_susceptible = max(1, int(round(susceptible.size * fraction)))
    draw_r = rng.integers(0, resistant.size, size=(repetitions, n_resistant))
    draw_s = rng.integers(0, susceptible.size, size=(repetitions, n_susceptible))
    delta = resistant[draw_r].mean(axis=1) - susceptible[draw_s].mean(axis=1)
    low, high = np.quantile(delta, [alpha / 2.0, 1.0 - alpha / 2.0])
    return bool(high < 0.0 or low > 0.0)


def group_lasso_select(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lambdas: tuple[float, ...] = (0.002, 0.01, 0.03, 0.1, 0.3),
    n_iter: int = 400,
    learning_rate: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Return feature indices whose time-step group survives the chosen penalty."""

    if len(y_val) == 0:
        raise ValueError("Group lasso needs a validation set to choose lambda.")
    best_indices: np.ndarray | None = None
    best_merit = -np.inf
    best_size = 10**9
    for lam in lambdas:
        weights, mean, std = _fit_group_lasso(
            x_train, y_train, lam=lam, n_iter=n_iter, learning_rate=learning_rate, seed=seed
        )
        indices = _active_groups(weights, x_train.shape[1], x_train.shape[2])
        probabilities = _predict(x_val, weights, mean, std)
        merit = figure_of_merit(classification_metrics(y_val, probabilities))
        size = int(indices.size)
        if merit > best_merit + 1e-6 or (abs(merit - best_merit) <= 1e-6 and size < best_size):
            best_merit = merit
            best_size = size
            best_indices = indices
    return np.asarray([] if best_indices is None else best_indices, dtype=int)


def _fit_group_lasso(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    n_iter: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design, mean, std = _design(x, mean=None, std=None)
    targets = np.asarray(y, dtype=float).ravel()
    weights = np.zeros(design.shape[1], dtype=float)
    rng = np.random.default_rng(seed)
    weights[-1] = float(np.clip(targets.mean(), 0.05, 0.95))
    weights[-1] = float(np.log(weights[-1] / (1.0 - weights[-1])))
    weights[:-1] = rng.normal(0.0, 0.01, size=weights.size - 1)
    timesteps = x.shape[1]
    n_features = x.shape[2]
    for _ in range(n_iter):
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - targets) / targets.size
        weights = weights - learning_rate * gradient
        weights = _group_soft_threshold(weights, timesteps, n_features, learning_rate * lam)
    return weights, mean, std


def _design(
    x: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.transpose(np.asarray(x, dtype=float), (0, 2, 1))
    flat = ordered.reshape(ordered.shape[0], -1).copy()
    # The study codes don't-care as -1, outside the normalized [0, 1] range.
    flat[np.isclose(flat, -1.0, atol=1e-5)] = np.nan
    if mean is None or std is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
    scaled = (flat - mean) / std
    scaled = np.where(np.isnan(scaled), 0.0, scaled)
    design = np.concatenate([scaled, np.ones((scaled.shape[0], 1))], axis=1)
    return design, mean, std


def _predict(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    design, _, _ = _design(x, mean, std)
    return _sigmoid(design @ weights)


def _active_groups(weights: np.ndarray, timesteps: int, n_features: int, tolerance: float = 1e-4) -> np.ndarray:
    active = []
    for feature in range(n_features):
        block = weights[feature * timesteps : (feature + 1) * timesteps]
        if float(np.linalg.norm(block)) > tolerance:
            active.append(feature)
    return np.asarray(active, dtype=int)


def _group_soft_threshold(weights: np.ndarray, timesteps: int, n_features: int, threshold: float) -> np.ndarray:
    shrunk = weights.copy()
    for feature in range(n_features):
        start = feature * timesteps
        stop = start + timesteps
        block = shrunk[start:stop]
        norm = float(np.linalg.norm(block))
        if norm <= threshold or norm == 0.0:
            shrunk[start:stop] = 0.0
        else:
            shrunk[start:stop] = block * (1.0 - threshold / norm)
    return shrunk


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
