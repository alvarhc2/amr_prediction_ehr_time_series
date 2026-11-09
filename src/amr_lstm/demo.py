"""Run the published protocol on a small synthetic cohort.

The run is a functional check of the code, not a reproduction of the hospital
results. Paper-scale budgets live on ``StudyConfig.paper()``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from amr_lstm.config import StudyConfig
from amr_lstm.data import temporal_split
from amr_lstm.metrics import classification_metrics
from amr_lstm.model import evaluate_model
from amr_lstm.pipeline import select_features, subset_split
from amr_lstm.plots import plot_selection_curve
from amr_lstm.schema import clinical_name
from amr_lstm.synthetic import demo_feature_names, generate_cohort
from amr_lstm.training import ensemble_proba, fit_horizontal_ensemble


def main() -> None:
    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description="Run a short synthetic demo of the AMR LSTM pipeline.")
    parser.add_argument("--family", default="amg", help="Antibiotic family: amg, car, cf4, pap, pol, qui.")
    parser.add_argument("--patients", type=int, default=36, help="Number of synthetic patients.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "demo_selection.png",
        help="Where to write the selection-curve figure.",
    )
    args = parser.parse_args()
    config = StudyConfig.demo()
    names = demo_feature_names()
    cohort = generate_cohort(args.patients, feature_names=names, seed=args.seed)
    split = temporal_split(cohort, args.family, year_split=config.year_split, seed=args.seed)
    print(f"Synthetic cohort: {cohort.x.shape[0]} cultures, {cohort.timesteps} days, {cohort.n_features} features")
    print(
        f"Split ({args.family}): train {len(split.y_train)}, "
        f"validation {len(split.y_val)}, test {len(split.y_test)}"
    )
    result = select_features(split, cohort.feature_names, config, seed=args.seed)
    print(f"Recursive selector keeps {result.n_features} features: " + _format(result.proposed.feature_names))
    print(f"Bootstrap voting keeps {len(result.bootstrap_voting.feature_names)} features")
    print(f"Group lasso keeps {len(result.group_lasso.feature_names)} features")
    plot_selection_curve(result.curve, result.n_features, args.output)
    print(f"Wrote {args.output}")

    reduced = subset_split(split, result.proposed.indices)
    members, reference = fit_horizontal_ensemble(
        reduced.x_train,
        reduced.y_train,
        reduced.x_val,
        reduced.y_val,
        units=config.units_reduced,
        epochs=config.ensemble_epochs,
        first_epoch=config.ensemble_first_epoch,
        reference_epoch=config.ensemble_reference_epoch,
        seed=args.seed,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        decay=config.learning_rate_decay,
        batch_size=config.batch_size,
    )
    single = evaluate_model(reference, reduced.x_test, reduced.y_test)
    voted = classification_metrics(reduced.y_test, ensemble_proba(members, reduced.x_test))
    print(
        "Test scores on this synthetic draw — "
        f"single network AUC {single['auc']:.3f}, "
        f"ensemble of {len(members)} checkpoints AUC {voted['auc']:.3f}. "
        "These are not the hospital results."
    )


def _format(names: list[str]) -> str:
    shown = [clinical_name(name) for name in names[:8]]
    suffix = "" if len(names) <= 8 else f" … (+{len(names) - 8})"
    return ", ".join(shown) + suffix


if __name__ == "__main__":
    main()
