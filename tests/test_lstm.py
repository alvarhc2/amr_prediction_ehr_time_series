import numpy as np

from amr_lstm.model import evaluate_model, gradient_sensitivity, train_lstm


def test_lstm_trains_and_exposes_an_input_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5, 3)).astype(np.float32)
    y = (x[:, -1, 0] > 0).astype(np.float32)
    model = train_lstm(x[:8], y[:8], x[8:], y[8:], units=4, epochs=1, seed=0, batch_size=4)
    metrics = evaluate_model(model, x[8:], y[8:])
    assert set(metrics) >= {"loss", "auc", "accuracy", "sensitivity", "specificity"}
    sensitivity = gradient_sensitivity(model, x[0])
    assert sensitivity.shape == (3,)
    assert np.all(np.isfinite(sensitivity))
