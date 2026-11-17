"""Command line for the synthetic cohort and for a local cohort file.

The default settings are the published budget. Pass a cohort with ``--data``,
or leave it unset to generate a synthetic one.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from amr_lstm.config import StudyConfig, ensemble_bounds
from amr_lstm.schema import FAMILIES, clinical_name


def main() -> None:
    _enable_color()
    args = _parser().parse_args()
    if args.data and args.synthetic:
        raise SystemExit("Pass either --data or --synthetic.")
    if not args.data and not args.synthetic:
        args.synthetic = True
    _bind_runtime()

    config = _config(args)
    cohort = _cohort(args)
    subset_size = min(config.random_subset_size, cohort.n_features)
    if subset_size != config.random_subset_size:
        _line(
            f"subset size {config.random_subset_size} exceeds {cohort.n_features} columns; using {subset_size}"
        )
        config = replace(config, random_subset_size=subset_size)
    _title("Budget")
    _line(
        f"{config.units_full} units, {config.epochs} epochs, "
        f"importance {config.gradient_repetitions}/{config.random_subsets}/{config.permutation_repetitions}, "
        f"{config.units_reduced} units after selection"
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
        _line("Published budget. The selection curve trains one network per feature.")

    families = list(FAMILIES) if args.family == "all" else [args.family]
    output = Path(args.output)
    for family in families:
        _run_family(cohort, family, config, seed=args.seed, output=output, several=len(families) > 1)


def _cohort(args: argparse.Namespace):
    if args.synthetic:
        names = short_feature_names() if args.columns == "short" else None
        _title("Cohort")
        _line(
            f"synthetic, {args.patients} patients, "
            f"{'short column list' if names is not None else 'study columns'}"
        )
        cohort = generate_cohort(
            args.patients,
            feature_names=names,
            seed=args.seed,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    else:
        _title("Cohort")
        _line(str(args.data))
        cohort = load_cohort(args.data)
    _line(f"{cohort.x.shape[0]} cultures, {cohort.timesteps} days, {cohort.n_features} features")
    return cohort


def _run_family(cohort, family: str, config: StudyConfig, *, seed: int, output: Path, several: bool) -> None:
    split = temporal_split(cohort, family, year_split=config.year_split, seed=seed)
    _title(family.upper())
    _line(
        f"train {_split_count(split.y_train)}, "
        f"validation {_split_count(split.y_val)}, "
        f"test {_split_count(split.y_test)}"
    )
    if min(len(split.y_train), len(split.y_val), len(split.y_test)) == 0:
        raise SystemExit(f"{family.upper()} has an empty train, validation or test split.")

    status = _Status()
    result = select_features(split, cohort.feature_names, config, seed=seed, progress=status)
    status.close()
    _line(f"kept {result.n_features}: " + _format(result.proposed.feature_names))
    _line(f"bootstrap voting kept {len(result.bootstrap_voting.feature_names)}")
    _line(f"group lasso kept {len(result.group_lasso.feature_names)}")

    stem = f"{family}_" if several else ""
    figure = output / f"{stem}selection.png"
    ranking = output / f"{stem}ranking.txt"
    plot_selection_curve(result.curve, result.n_features, figure)
    _write_ranking(ranking, result)
    _title("Files")
    _line(str(figure))
    _line(str(ranking))

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
    if np.unique(reduced.y_test.astype(int)).size < 2:
        _line("test AUC is undefined: the test split has only one class")
    else:
        _line(f"single network AUC {single['auc']:.3f}")
        _line(f"ensemble of {len(members)} checkpoints AUC {voted['auc']:.3f}")


def _bind_runtime() -> None:
    """Load TensorFlow and the training code after ``--help`` has already returned."""

    global np
    global classification_metrics, evaluate_model, select_features, subset_split
    global plot_selection_curve, generate_cohort, short_feature_names
    global load_cohort, temporal_split, ensemble_proba, fit_horizontal_ensemble
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as np
    from amr_lstm.data import load_cohort, temporal_split
    from amr_lstm.metrics import classification_metrics
    from amr_lstm.model import evaluate_model
    from amr_lstm.pipeline import select_features, subset_split
    from amr_lstm.plots import plot_selection_curve
    from amr_lstm.synthetic import generate_cohort, short_feature_names
    from amr_lstm.training import ensemble_proba, fit_horizontal_ensemble
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


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
        prog="python -m amr_lstm",
        description="Predict Pseudomonas aeruginosa resistance with the LSTM study.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=str, help="Path to an NPZ cohort with x, y, year, patient_id, order and feature_names")
    source.add_argument("--synthetic", action="store_true", help="Generate a cohort instead of reading one")
    parser.add_argument("--family", default="amg", choices=[*FAMILIES, "all"], help="Antibiotic family to predict, or all six")
    parser.add_argument("--year-split", type=int, default=published.year_split, help="Last year included in training")
    parser.add_argument("--units", type=int, default=published.units_full, help="Number of LSTM units on the full feature set")
    parser.add_argument(
        "--units-selected",
        type=int,
        default=published.units_reduced,
        help="Number of LSTM units after feature selection",
    )
    parser.add_argument("--epochs", type=int, default=published.epochs, help="Number of epochs for importance and the selection curve")
    parser.add_argument(
        "--ensemble-epochs",
        type=int,
        default=published.ensemble_epochs,
        help="Number of epochs for the horizontal ensemble",
    )
    parser.add_argument("--gradient-repetitions", type=int, default=published.gradient_repetitions, help="Number of trainings for gradient feature importance")
    parser.add_argument("--random-subsets", type=int, default=published.random_subsets, help="Number of random feature subsets")
    parser.add_argument("--subset-size", type=int, default=published.random_subset_size, help="Number of features in each random subset")
    parser.add_argument("--permutation-repetitions", type=int, default=published.permutation_repetitions, help="Number of trainings for permutation feature importance")
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Number of features at which the selection curve stops. Default: every feature",
    )
    parser.add_argument("--patients", type=int, default=80, help="Number of patients in the synthetic cohort")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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


_SECTIONS = {
    "Gradient feature importance": ("Feature importance", "Gradient"),
    "Random-subset feature importance": ("Feature importance", "Random subset"),
    "Permutation feature importance": ("Feature importance", "Permutation"),
    "Selection curve": ("Feature selection", "Curve"),
    "Bootstrap voting": ("Feature selection", "Bootstrap voting"),
    "Group lasso": ("Feature selection", "Group lasso"),
    "Training horizontal ensemble": ("Ensemble", "Training"),
}


class _Status:
    """One updating line per stage, grouped under a section title."""

    def __init__(self) -> None:
        self._label: str | None = None
        self._section: str | None = None
        self._width = 0

    def __call__(self, label: str, done: int, total: int) -> None:
        section, short = _SECTIONS.get(label, (None, label))
        if label != self._label:
            self.close()
            if section != self._section:
                _title(section or label)
                self._section = section
            self._label = label
            self._short = short
        filled = 0 if total <= 0 else int(20 * done / total)
        bar = _paint("#" * filled, "32") + _paint("-" * (20 - filled), "2")
        visible = f"  {self._short:<18} [{'#' * filled}{'-' * (20 - filled)}] {done}/{total}"
        text = f"  {self._short:<18} [{bar}] {done}/{total}"
        padding = " " * max(0, self._width - len(visible))
        print("\r" + text + padding, end="", flush=True)
        self._width = len(visible)

    def close(self) -> None:
        if self._label is not None:
            print(flush=True)
            self._label = None
            self._width = 0


def _split_count(labels: np.ndarray) -> str:
    resistant = int(np.sum(np.asarray(labels) == 1))
    return f"{len(labels)} ({resistant} resistant)"


def _enable_color() -> None:
    global _COLOR
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        _COLOR = False
        return
    if sys.platform == "win32":
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                _COLOR = False
                return
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            _COLOR = False
            return
    _COLOR = True


def _paint(text: str, code: str) -> str:
    if not _COLOR or not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _title(text: str) -> None:
    print(flush=True)
    print(_paint(text, "1;32"), flush=True)


def _line(text: str) -> None:
    print(f"  {text}", flush=True)


_COLOR = False


def _write_ranking(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gradient = dict(result.gradient)
    random_subset = dict(result.random_subset)
    permutation = dict(result.permutation)
    rows = [("rank", "feature", "average", "gradient", "random_subset", "permutation")]
    for rank, (name, average) in enumerate(result.ranked, start=1):
        rows.append(
            (
                str(rank),
                clinical_name(name),
                f"{average:.6f}",
                f"{gradient[name]:.6f}",
                f"{random_subset[name]:.6f}",
                f"{permutation[name]:.6f}",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [
            row[0].rjust(widths[0]),
            row[1].ljust(widths[1]),
            *(cell.rjust(widths[index]) for index, cell in enumerate(row[2:], start=2)),
        ]
        lines.append("  ".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(names: list[str]) -> str:
    shown = [clinical_name(name) for name in names[:8]]
    suffix = "" if len(names) <= 8 else f" … (+{len(names) - 8})"
    return ", ".join(shown) + suffix


if __name__ == "__main__":
    main()
