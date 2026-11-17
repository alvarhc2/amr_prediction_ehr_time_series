"""Command line for the synthetic cohort and for a local cohort file.

The default settings are the published budget. Pass a cohort with ``--data``,
or leave it unset to generate a synthetic one.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from amr_lstm.config import StudyConfig, ensemble_bounds
from amr_lstm.data import load_cohort, temporal_split
from amr_lstm.metrics import classification_metrics
from amr_lstm.model import evaluate_model
from amr_lstm.pipeline import FamilyResult, select_features, subset_split
from amr_lstm.plots import plot_selection_curve
from amr_lstm.schema import FAMILIES, clinical_name
from amr_lstm.synthetic import generate_cohort, short_feature_names
from amr_lstm.training import ensemble_proba, fit_horizontal_ensemble


def main() -> None:
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    args = _parser().parse_args()
    if args.data and args.synthetic:
        raise SystemExit("Pass either --data or --synthetic.")
    if not args.data and not args.synthetic:
        args.synthetic = True

    config = _config(args)
    cohort = _cohort(args)
    subset_size = min(config.random_subset_size, cohort.n_features)
    if subset_size != config.random_subset_size:
        print(
            f"subset    {config.random_subset_size} exceeds {cohort.n_features} columns; using {subset_size}",
            flush=True,
        )
        config = replace(config, random_subset_size=subset_size)
    print(
        f"budget    {config.units_full} units, {config.epochs} epochs, "
        f"importance {config.gradient_repetitions}/{config.random_subsets}/{config.permutation_repetitions}, "
        f"selected units {config.units_reduced}",
        flush=True,
    )
    published = StudyConfig.paper()
    if (
        config.units_full == published.units_full
        and config.epochs == published.epochs
        and config.gradient_repetitions == published.gradient_repetitions
        and config.random_subsets == published.random_subsets
        and config.curve_max_features is None
        and cohort.n_features > 100
    ):
        print("Published budget. The selection curve trains one network per feature.", flush=True)

    families = list(FAMILIES) if args.family == "all" else [args.family]
    output = Path(args.output)
    for family in families:
        _run_family(cohort, family, config, seed=args.seed, output=output, several=len(families) > 1)


def _cohort(args: argparse.Namespace):
    if args.synthetic:
        names = short_feature_names() if args.columns == "short" else None
        print(
            f"cohort    synthetic, {args.patients} patients, "
            f"{'short column list' if names is not None else 'study columns'}",
            flush=True,
        )
        cohort = generate_cohort(
            args.patients,
            feature_names=names,
            seed=args.seed,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    else:
        print(f"cohort    {args.data}", flush=True)
        cohort = load_cohort(args.data)
    print(
        f"          {cohort.x.shape[0]} cultures, {cohort.timesteps} days, {cohort.n_features} features",
        flush=True,
    )
    return cohort


def _run_family(cohort, family: str, config: StudyConfig, *, seed: int, output: Path, several: bool) -> None:
    split = temporal_split(cohort, family, year_split=config.year_split, seed=seed)
    print(
        f"{family.upper():<9} train {len(split.y_train)}, "
        f"validation {len(split.y_val)}, test {len(split.y_test)}",
        flush=True,
    )
    if min(len(split.y_train), len(split.y_val), len(split.y_test)) == 0:
        raise SystemExit(f"{family.upper()} has an empty train, validation or test split.")

    status = _Status()
    result = select_features(split, cohort.feature_names, config, seed=seed, progress=status)
    status.close()
    print(f"selected  {result.n_features} features: " + _format(result.proposed.feature_names), flush=True)
    print(f"bootstrap {len(result.bootstrap_voting.feature_names)} features", flush=True)
    print(f"lasso     {len(result.group_lasso.feature_names)} features", flush=True)

    stem = f"{family}_" if several else ""
    figure = output / f"{stem}selection.png"
    ranking = output / f"{stem}ranking.txt"
    plot_selection_curve(result.curve, result.n_features, figure)
    _write_ranking(ranking, result)
    print(f"Wrote {figure}", flush=True)
    print(f"Wrote {ranking}", flush=True)

    reduced = subset_split(split, result.proposed.indices)
    status = _Status()
    members, reference = fit_horizontal_ensemble(
        reduced.x_train,
        reduced.y_train,
        reduced.x_val,
        reduced.y_val,
        units=config.units_reduced,
        epochs=config.ensemble_epochs,
        first_epoch=config.ensemble_first_epoch,
        reference_epoch=config.ensemble_reference_epoch,
        seed=seed,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        decay=config.learning_rate_decay,
        batch_size=config.batch_size,
        progress=lambda done, total: status("Training horizontal ensemble", done, total),
    )
    status.close()
    single = evaluate_model(reference, reduced.x_test, reduced.y_test)
    voted = classification_metrics(reduced.y_test, ensemble_proba(members, reduced.x_test))
    print(
        f"test      single-network AUC {single['auc']:.3f}, "
        f"ensemble of {len(members)} checkpoints AUC {voted['auc']:.3f}",
        flush=True,
    )


def _config(args: argparse.Namespace) -> StudyConfig:
    first, reference = ensemble_bounds(args.ensemble_epochs)
    published = StudyConfig.paper()
    return StudyConfig(
        units_full=args.units,
        units_reduced=args.units_selected,
        epochs=args.epochs,
        year_split=args.year_split,
        gradient_repetitions=args.gradient_repetitions,
        random_subsets=args.random_subsets,
        random_subset_size=args.subset_size,
        permutation_repetitions=args.permutation_repetitions,
        ensemble_epochs=args.ensemble_epochs,
        ensemble_first_epoch=first,
        ensemble_reference_epoch=reference,
        curve_max_features=args.max_features,
        dropout=published.dropout,
        learning_rate=published.learning_rate,
        learning_rate_decay=published.learning_rate_decay,
        batch_size=published.batch_size,
        recursive_threshold=published.recursive_threshold,
        bootstrap_repetitions=published.bootstrap_repetitions,
        bootstrap_fraction=published.bootstrap_fraction,
        bootstrap_alpha=published.bootstrap_alpha,
        vote_fraction=published.vote_fraction,
        group_lasso_lambdas=published.group_lasso_lambdas,
        group_lasso_iterations=published.group_lasso_iterations,
    )


def _parser() -> argparse.ArgumentParser:
    published = StudyConfig.paper()
    parser = argparse.ArgumentParser(
        description="Predict Pseudomonas aeruginosa resistance with the LSTM study."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=str, help="NPZ cohort with x, y, year, patient_id, order and feature_names")
    source.add_argument("--synthetic", action="store_true", help="Run on a generated cohort")
    parser.add_argument("--family", default="amg", choices=[*FAMILIES, "all"])
    parser.add_argument("--year-split", type=int, default=published.year_split, help="Last training year")
    parser.add_argument("--units", type=int, default=published.units_full, help="LSTM units on the full feature set")
    parser.add_argument(
        "--units-selected",
        type=int,
        default=published.units_reduced,
        help="LSTM units after feature selection",
    )
    parser.add_argument("--epochs", type=int, default=published.epochs, help="Epochs for importance and the selection curve")
    parser.add_argument(
        "--ensemble-epochs",
        type=int,
        default=published.ensemble_epochs,
        help="Epochs for the horizontal ensemble",
    )
    parser.add_argument("--gradient-repetitions", type=int, default=published.gradient_repetitions)
    parser.add_argument("--random-subsets", type=int, default=published.random_subsets)
    parser.add_argument("--subset-size", type=int, default=published.random_subset_size)
    parser.add_argument("--permutation-repetitions", type=int, default=published.permutation_repetitions)
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Stop the selection curve after this many features. Default: every feature",
    )
    parser.add_argument("--patients", type=int, default=80, help="Patients in the synthetic cohort")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-year", type=int, default=2004, help="First admission year of the synthetic cohort")
    parser.add_argument("--end-year", type=int, default=2019, help="Last admission year of the synthetic cohort")
    parser.add_argument(
        "--columns",
        default="study",
        choices=["study", "short"],
        help="Synthetic columns: the 398 study columns, or a short list",
    )
    parser.add_argument("--output", type=str, default="outputs", help="Directory for the curve and the ranking")
    return parser


class _Status:
    """One updating line per stage. A new stage keeps the finished line."""

    def __init__(self) -> None:
        self._label: str | None = None
        self._width = 0

    def __call__(self, label: str, done: int, total: int) -> None:
        if label != self._label:
            self.close()
            self._label = label
        filled = 0 if total <= 0 else int(20 * done / total)
        bar = "#" * filled + "-" * (20 - filled)
        text = f"{label} [{bar}] {done}/{total}"
        padding = " " * max(0, self._width - len(text))
        print("\r" + text + padding, end="", flush=True)
        self._width = len(text)

    def close(self) -> None:
        if self._label is not None:
            print(flush=True)
            self._label = None
            self._width = 0


def _write_ranking(path: Path, result: FamilyResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gradient = dict(result.gradient)
    random_subset = dict(result.random_subset)
    permutation = dict(result.permutation)
    lines = ["rank\tfeature\taverage\tgradient\trandom_subset\tpermutation"]
    for rank, (name, average) in enumerate(result.ranked, start=1):
        lines.append(
            f"{rank}\t{clinical_name(name)}\t{average:.6f}\t"
            f"{gradient[name]:.6f}\t{random_subset[name]:.6f}\t{permutation[name]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(names: list[str]) -> str:
    shown = [clinical_name(name) for name in names[:8]]
    suffix = "" if len(names) <= 8 else f" … (+{len(names) - 8})"
    return ", ".join(shown) + suffix


if __name__ == "__main__":
    main()
