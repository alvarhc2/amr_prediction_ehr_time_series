"""Stacked LSTM used as the classifier and as the wrapper inside feature selection.

The network is the one in the paper: two LSTM layers, dropout 0.2, a sigmoid
unit, binary cross-entropy and RMSprop. The LSTM candidate activation is ReLU.
The legacy ``decay`` argument of RMSprop is reproduced with inverse-time decay,
``learning_rate / (1 + decay * step)``.

Models that see every feature use 100 units. Models trained after feature
selection, including the ensemble, use 50 units.
"""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf

from amr_lstm.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm(timesteps: int, n_features: int, units: int, dropout: float = 0.2) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, n_features)),
            tf.keras.layers.LSTM(units, activation="relu", return_sequences=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def rmsprop(learning_rate: float, decay: float) -> tf.keras.optimizers.Optimizer:
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1,
        decay_rate=decay,
        staircase=False,
    )
    return tf.keras.optimizers.RMSprop(learning_rate=schedule)


def compile_model(model: tf.keras.Model, learning_rate: float, decay: float) -> tf.keras.Model:
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=rmsprop(learning_rate, decay))
    return model


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    units: int,
    epochs: int,
    seed: int,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    decay: float = 1e-5,
    batch_size: int = 32,
    verbose: int = 0,
) -> tf.keras.Model:
    set_seed(seed)
    model = build_lstm(x_train.shape[1], x_train.shape[2], units, dropout)
    compile_model(model, learning_rate, decay)
    validation = None
    if x_val is not None and y_val is not None and len(y_val) > 0:
        validation = (x_val, y_val)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation,
        verbose=verbose,
    )
    return model


def predict_proba(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(model.predict(x, verbose=0), dtype=np.float32).reshape(-1)


def evaluate_model(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return classification_metrics(y, predict_proba(model, x))


def gradient_sensitivity(model: tf.keras.Model, sequence: np.ndarray) -> np.ndarray:
    """Squared mean-over-time input gradient of one culture. Shape ``(n_features,)``."""

    inputs = tf.Variable(sequence[np.newaxis, ...].astype(np.float32))
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=False)
    gradients = tape.gradient(prediction, inputs)
    if gradients is None:
        raise RuntimeError("The model did not produce an input gradient.")
    averaged = tf.reduce_mean(gradients, axis=1).numpy()[0]
    return np.square(averaged)
