"""Cohort container and the temporal, patient-independent split.

A cohort is a tensor ``x`` of shape ``(n_cultures, 30, n_features)`` plus one
binary target per antibiotic family. Missing targets (a family that was not
tested on that culture) are ``NaN``. Cultures are ordered in time. Training
uses cultures from ``year_split`` and earlier. Later cultures whose patient
already appears in training are dropped. The remaining cultures are cut in
half: the earlier half is validation and the later half is test. Each split is
then shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amr_lstm.schema import FAMILIES


@dataclass
class Cohort:
    x: np.ndarray
    y: np.ndarray
    year: np.ndarray
    patient_id: np.ndarray
    order: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        self.year = np.asarray(self.year, dtype=np.int32)
        self.patient_id = np.asarray(self.patient_id).astype(str)
        self.order = np.asarray(self.order, dtype=np.int64)
        self.feature_names = tuple(self.feature_names)
        n = self.x.shape[0]
        if self.x.ndim != 3:
            raise ValueError("x must have shape (n_cultures, timesteps, n_features).")
        if self.y.shape != (n, len(FAMILIES)):
            raise ValueError(f"y must have shape (n_cultures, {len(FAMILIES)}).")
        if not (self.year.shape == self.patient_id.shape == self.order.shape == (n,)):
            raise ValueError("year, patient_id and order must have one entry per culture.")
        if self.x.shape[2] != len(self.feature_names):
            raise ValueError("x and feature_names disagree on the number of features.")

    @property
    def n_features(self) -> int:
        return int(self.x.shape[2])

    @property
    def timesteps(self) -> int:
        return int(self.x.shape[1])


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    family: str

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[2]) if self.x_train.size else 0


def family_index(family: str) -> int:
    try:
        return FAMILIES.index(family)
    except ValueError as exc:
        raise ValueError(f"Unknown family {family!r}. Expected one of {FAMILIES}.") from exc


def feature_index(feature_names: tuple[str, ...] | list[str], names: list[str] | tuple[str, ...]) -> list[int]:
    lookup = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise KeyError(f"Features not in the cohort: {missing[:5]}")
    return [lookup[name] for name in names]


def slice_features(x: np.ndarray, indices: list[int] | np.ndarray) -> np.ndarray:
    return np.asarray(x)[:, :, list(indices)]


def temporal_split(cohort: Cohort, family: str, *, year_split: int = 2012, seed: int = 0) -> Split:
    """Build the train, validation and test tensors for one antibiotic family."""

    labels = cohort.y[:, family_index(family)]
    observed = np.flatnonzero(~np.isnan(labels))
    observed = observed[np.argsort(cohort.order[observed], kind="mergesort")]

    train_ids: list[int] = []
    later_ids: list[int] = []
    train_patients: set[str] = set()
    for index in observed.tolist():
        patient = cohort.patient_id[index]
        if int(cohort.year[index]) <= year_split:
            train_ids.append(index)
            train_patients.add(patient)
        elif patient not in train_patients:
            later_ids.append(index)

    half = len(later_ids) // 2
    val_ids = later_ids[:half]
    test_ids = later_ids[half:]
    rng = np.random.default_rng(seed)

    def _pack(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not ids:
            empty_x = cohort.x[:0]
            empty_y = labels[:0]
            return empty_x, empty_y
        chosen = np.asarray(ids, dtype=int)
        if chosen.size > 1:
            chosen = rng.permutation(chosen)
        return cohort.x[chosen], labels[chosen].astype(np.float32)

    x_train, y_train = _pack(train_ids)
    x_val, y_val = _pack(val_ids)
    x_test, y_test = _pack(test_ids)
    return Split(x_train, y_train, x_val, y_val, x_test, y_test, family)


def save_cohort(path: str | Path, cohort: Cohort) -> None:
    np.savez_compressed(
        path,
        x=cohort.x,
        y=cohort.y,
        year=cohort.year,
        patient_id=cohort.patient_id,
        order=cohort.order,
        feature_names=np.asarray(cohort.feature_names),
    )


def load_cohort(path: str | Path) -> Cohort:
    with np.load(path, allow_pickle=False) as payload:
        return Cohort(
            x=payload["x"],
            y=payload["y"],
            year=payload["year"],
            patient_id=payload["patient_id"],
            order=payload["order"],
            feature_names=tuple(str(name) for name in payload["feature_names"].tolist()),
        )
