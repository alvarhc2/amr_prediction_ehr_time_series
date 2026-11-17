import warnings

import numpy as np

from amr_lstm.metrics import classification_metrics


def test_auc_is_undefined_for_a_single_class_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = classification_metrics(np.zeros(4), np.linspace(0.2, 0.8, 4))
    assert np.isnan(metrics["auc"])
