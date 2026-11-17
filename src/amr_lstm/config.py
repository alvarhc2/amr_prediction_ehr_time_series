"""Hyper-parameters of the published study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StudyConfig:
    """Settings that reproduce the experimental protocol in the paper.

    The field defaults are that published budget. ``paper()`` returns them
    unchanged. A command-line run can replace the fields it exposes, such as
    the unit counts, the epoch counts and the importance repetitions.
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


def ensemble_bounds(epochs: int) -> tuple[int, int]:
    """Checkpoint window for a horizontal ensemble trained for ``epochs`` epochs.

    The published run trains for 30 epochs, keeps epochs 11 through 30, and
    uses epoch 20 as the single network. For another length the same
    fractions are kept: the last two thirds vote, and the single network is
    the epoch two thirds of the way through training.
    """

    if epochs < 1:
        raise ValueError("The ensemble needs at least one epoch.")
    members = max(1, round(20 / 30 * epochs))
    first = max(1, epochs - members + 1)
    reference = max(first, min(epochs, round(20 / 30 * epochs)))
    return first, reference
