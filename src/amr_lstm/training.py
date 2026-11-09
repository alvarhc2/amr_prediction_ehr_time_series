"""Forward selection curve and the horizontal voting ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from amr_lstm.metrics import classification_metrics
from amr_lstm.model import build_lstm, compile_model, predict_proba, set_seed, train_lstm


@dataclass
class PerformanceCurve:
    accuracy: np.ndarray
    sensitivity: np.ndarray
    specificity: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.feature_names = tuple(self.feature_names)


def selection_curve(
    ranked_features: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    *,
    units: int,
    epochs: int,
    seed: int = 0,
    max_features: int | None = None,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
) -> PerformanceCurve:
    """Train on the top-1, top-2, ... features and record validation rates.

    The published curve walks all 398 features. ``max_features`` stops early,
    which the demo uses.
    """

    lookup = {name: index for index, name in enumerate(feature_names)}
    limit = len(ranked_features) if max_features is None else min(max_features, len(ranked_features))
    chosen: list[int] = []
    accuracy = np.zeros(limit, dtype=float)
    sensitivity = np.zeros(limit, dtype=float)
    specificity = np.zeros(limit, dtype=float)
    kept_names: list[str] = []
    rng = np.random.default_rng(seed)
    for step in range(limit):
        name = ranked_features[step]
        chosen.append(lookup[name])
        kept_names.append(name)
        model = train_lstm(
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
        metrics = classification_metrics(y_val, predict_proba(model, x_val[:, :, chosen]))
        accuracy[step] = metrics["accuracy"]
        sensitivity[step] = metrics["sensitivity"]
        specificity[step] = metrics["specificity"]
    return PerformanceCurve(accuracy, sensitivity, specificity, tuple(kept_names))


def fit_horizontal_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    units: int,
    epochs: int,
    first_epoch: int,
    reference_epoch: int,
    seed: int,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
) -> tuple[list[tf.keras.Model], tf.keras.Model]:
    """Keep a checkpoint at the end of each epoch in ``[first_epoch, epochs]``.

    Returns ``(members, reference)``. ``reference`` is the single network at
    ``reference_epoch``. Members vote by averaging probabilities. Epoch numbers
    are 1-based. The published ensemble keeps epochs 11 through 30 and compares
    them with the network at epoch 20.
    """

    if not 1 <= first_epoch <= reference_epoch <= epochs:
        raise ValueError("Epoch bounds must satisfy 1 <= first <= reference <= epochs.")
    set_seed(seed)
    model = build_lstm(x_train.shape[1], x_train.shape[2], units, dropout)
    compile_model(model, learning_rate, decay)
    validation = None
    if x_val is not None and y_val is not None and len(y_val) > 0:
        validation = (x_val, y_val)
    members: list[tf.keras.Model] = []
    for epoch in range(1, epochs + 1):
        model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=validation,
            verbose=0,
        )
        if epoch >= first_epoch:
            members.append(_clone(model))
    reference = members[reference_epoch - first_epoch]
    return members, reference


def ensemble_proba(members: list[tf.keras.Model], x: np.ndarray) -> np.ndarray:
    if not members:
        raise ValueError("The ensemble has no members.")
    predictions = [predict_proba(member, x) for member in members]
    return np.mean(np.vstack(predictions), axis=0)


def _clone(model: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone
