"""Hyper-parameters of the published study, and a small demo budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudyConfig:
    """Settings that reproduce the experimental protocol in the paper.

    ``paper()`` is the published budget. It trains thousands of LSTM networks
    and is not the default entry point. ``demo()`` keeps the same protocol and
    shortens every repetition so the pipeline can be executed locally.
    """

    timesteps: int = 30
    units_full: int = 100
    units_reduced: int = 50
    dropout: float = 0.2
    learning_rate: float = 1e-4
    learning_rate_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 32
    year_split: int = 2012
    gradient_repetitions: int = 50
    random_subsets: int = 500
    random_subset_size: int = 40
    permutation_repetitions: int = 50
    recursive_threshold: int = 10
    ensemble_epochs: int = 30
    # 1-based epoch numbers. Members are the checkpoints from
    # ``ensemble_first_epoch`` through ``ensemble_epochs`` (20 networks when
    # these are 11 and 30). The single-network baseline is epoch 20.
    ensemble_first_epoch: int = 11
    ensemble_reference_epoch: int = 20
    bootstrap_repetitions: int = 2000
    bootstrap_fraction: float = 0.7
    bootstrap_alpha: float = 0.05
    vote_fraction: float = 0.5
    group_lasso_lambdas: tuple[float, ...] = (0.002, 0.01, 0.03, 0.1, 0.3)
    group_lasso_iterations: int = 400
    curve_max_features: int | None = None

    @classmethod
    def paper(cls) -> StudyConfig:
        return cls()

    @classmethod
    def demo(cls) -> StudyConfig:
        return cls(
            units_full=8,
            units_reduced=8,
            epochs=2,
            ensemble_epochs=4,
            ensemble_first_epoch=2,
            ensemble_reference_epoch=3,
            gradient_repetitions=2,
            random_subsets=6,
            random_subset_size=6,
            permutation_repetitions=1,
            recursive_threshold=4,
            bootstrap_repetitions=300,
            group_lasso_iterations=120,
            curve_max_features=8,
        )
