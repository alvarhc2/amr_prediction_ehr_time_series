"""End-to-end selection for one antibiotic family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from amr_lstm.baselines import bootstrap_voting_select, group_lasso_select
from amr_lstm.config import StudyConfig
from amr_lstm.data import Split, feature_index, slice_features
from amr_lstm.importance import (
    average_importance,
    gradient_feature_importance,
    permutation_feature_importance,
    random_subset_feature_importance,
)
from amr_lstm.selection.recursive import recursive_feature_count
from amr_lstm.training import PerformanceCurve, selection_curve


@dataclass
class MethodSelection:
    method: str
    feature_names: list[str]
    indices: list[int]


@dataclass
class FamilyResult:
    family: str
    ranked: list[tuple[str, float]]
    curve: PerformanceCurve
    n_features: int
    proposed: MethodSelection
    bootstrap_voting: MethodSelection
    group_lasso: MethodSelection
    gradient: list[tuple[str, float]] = field(repr=False)
    random_subset: list[tuple[str, float]] = field(repr=False)
    permutation: list[tuple[str, float]] = field(repr=False)


def select_features(
    split: Split,
    feature_names: tuple[str, ...],
    config: StudyConfig,
    *,
    seed: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> FamilyResult:
    """Run the three importance methods, the recursive cutoff, and both baselines.

    ``progress``, when given, is called as ``progress(stage, done, total)``
    before a stage starts and after each unit of work in that stage.
    """

    shared = dict(
        units=config.units_full,
        epochs=config.epochs,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        decay=config.learning_rate_decay,
        batch_size=config.batch_size,
    )
    gradient = gradient_feature_importance(
        feature_names,
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        repetitions=config.gradient_repetitions,
        seed=seed,
        progress=_stage(progress, "Gradient feature importance"),
        **shared,
    )
    random_subset = random_subset_feature_importance(
        feature_names,
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        n_subsets=config.random_subsets,
        subset_size=config.random_subset_size,
        seed=seed + 1,
        progress=_stage(progress, "Random-subset feature importance"),
        **shared,
    )
    permutation = permutation_feature_importance(
        feature_names,
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        repetitions=config.permutation_repetitions,
        seed=seed + 2,
        progress=_stage(progress, "Permutation feature importance"),
        **shared,
    )
    ranked = average_importance(gradient, random_subset, permutation)
    curve = selection_curve(
        [name for name, _ in ranked],
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        feature_names,
        units=config.units_full,
        epochs=config.epochs,
        seed=seed + 3,
        max_features=config.curve_max_features,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        decay=config.learning_rate_decay,
        batch_size=config.batch_size,
        progress=_stage(progress, "Selection curve"),
    )
    n_features = recursive_feature_count(
        curve.accuracy,
        curve.sensitivity,
        curve.specificity,
        threshold=min(config.recursive_threshold, max(2, curve.accuracy.size)),
    )
    proposed_names = [name for name, _ in ranked[:n_features]]
    bootstrap_idx = bootstrap_voting_select(
        split.x_train,
        split.y_train,
        repetitions=config.bootstrap_repetitions,
        fraction=config.bootstrap_fraction,
        alpha=config.bootstrap_alpha,
        vote_fraction=config.vote_fraction,
        seed=seed + 4,
        progress=_stage(progress, "Bootstrap voting"),
    )
    lasso_idx = group_lasso_select(
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        lambdas=config.group_lasso_lambdas,
        n_iter=config.group_lasso_iterations,
        seed=seed + 5,
        progress=_stage(progress, "Group lasso"),
    )
    return FamilyResult(
        family=split.family,
        ranked=ranked,
        curve=curve,
        n_features=n_features,
        proposed=_selection("proposed", proposed_names, feature_names),
        bootstrap_voting=_selection("bootstrap_voting", [feature_names[i] for i in bootstrap_idx], feature_names),
        group_lasso=_selection("group_lasso", [feature_names[i] for i in lasso_idx], feature_names),
        gradient=gradient,
        random_subset=random_subset,
        permutation=permutation,
    )


def _stage(
    progress: Callable[[str, int, int], None] | None,
    label: str,
) -> Callable[[int, int], None] | None:
    if progress is None:
        return None

    def report(done: int, total: int) -> None:
        progress(label, done, total)

    return report


def _selection(method: str, names: list[str], feature_names: tuple[str, ...]) -> MethodSelection:
    return MethodSelection(method, names, feature_index(feature_names, names) if names else [])


def subset_split(split: Split, indices: list[int]) -> Split:
    """Restrict every array of a split to the selected columns."""

    if not indices:
        raise ValueError("Refusing to train on an empty feature set.")
    return Split(
        slice_features(split.x_train, indices),
        split.y_train,
        slice_features(split.x_val, indices),
        split.y_val,
        slice_features(split.x_test, indices),
        split.y_test,
        split.family,
    )
