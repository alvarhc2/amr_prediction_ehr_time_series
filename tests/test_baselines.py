import numpy as np

from amr_lstm.baselines import bootstrap_voting_select, group_lasso_select
from amr_lstm.importance import average_importance


def test_average_importance_minmax_then_mean():
    gradient = [("a", 0.0), ("b", 10.0)]
    subset = [("a", 5.0), ("b", 5.0)]
    permutation = [("a", 1.0), ("b", 3.0)]
    ranking = average_importance(gradient, subset, permutation)
    scores = dict(ranking)
    # a -> (0, 0, 0), b -> (1, 0, 1). Mean 0 vs 2/3.
    assert scores["b"] > scores["a"]
    assert ranking[0][0] == "b"


def test_bootstrap_voting_keeps_the_separated_feature():
    rng = np.random.default_rng(0)
    n, timesteps, features = 80, 6, 2
    x = rng.normal(0.2, 0.05, size=(n, timesteps, features))
    y = np.array([1] * 40 + [0] * 40)
    x[:40, :, 0] = 0.9
    x[40:, :, 0] = 0.1
    x[:, :, :] = np.clip(x, 0.0, 1.0)
    selected = bootstrap_voting_select(x, y, repetitions=200, fraction=0.7, seed=0)
    assert 0 in selected
    assert 1 not in selected


def test_group_lasso_prefers_the_informative_group():
    rng = np.random.default_rng(1)
    n, timesteps, features = 100, 4, 3
    x = rng.random((n, timesteps, features))
    y = np.zeros(n, dtype=int)
    y[::2] = 1
    x[y == 1, :, 1] = 0.95
    x[y == 0, :, 1] = 0.05
    x[:, 0, 2] = -1.0
    selected = group_lasso_select(
        x[:80],
        y[:80],
        x[80:],
        y[80:],
        lambdas=(0.02, 0.08),
        n_iter=250,
        seed=0,
    )
    assert 1 in selected
