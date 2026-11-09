"""Synthetic ICU cohort with the published tensor layout.

The generator does not imitate hospital records. It builds cultures whose
columns, 30-day window, don't-care code (``-1``) and six family targets follow
the study, and it plants a simple resistance mechanism so the selectors have a
signal to recover: prior resistance, a few administered antibiotics, sex,
respiratory admission and mechanical ventilation, plus a drift toward
resistance in later years. Polymyxin labels are absent before 2007, as in the
study cohort.
"""

from __future__ import annotations

import numpy as np

from amr_lstm.data import Cohort
from amr_lstm.schema import FAMILIES, FEATURES, feature_group

_SIGNAL = (
    "resistance_amg",
    "resistance_car",
    "resistance_cf4",
    "resistance_pap",
    "resistance_pol",
    "resistance_qui",
    "antibiotics-ciprofloxacina_j01ma",
    "antibiotics-colistina_j01xb",
    "antibiotics-amikacina_j01gb",
    "antibiotics-meropenem_j01dh",
    "gender",
    "age",
    "reason_admission-acute_respiratory_failure",
    "reason_admission-neuromuscular",
    "reason_admission-hypovolaemia",
    "patient_category-medical_patient",
    "mechanic_ventilation",
    "admission",
    "cultures",
    "pseudomonas_aeruginosa_detected",
    "day_week",
    "day_month",
    "month",
    "related_patients",
)


def demo_feature_names(n_noise: int = 16) -> tuple[str, ...]:
    """A short column list for the runnable demo, in modeling-matrix order."""

    noise = [
        name
        for name in FEATURES
        if name not in _SIGNAL and feature_group(name) in {"culture", "icu_treatment", "icu_admission"}
    ][:n_noise]
    chosen = set(_SIGNAL).union(noise)
    return tuple(name for name in FEATURES if name in chosen)


def generate_cohort(
    n_patients: int = 80,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
    seed: int = 0,
    start_year: int = 2004,
    end_year: int = 2019,
) -> Cohort:
    """Return a synthetic cohort. ``end_year`` is the last admission year."""

    if n_patients < 4:
        raise ValueError("n_patients must be at least 4.")
    names = tuple(feature_names) if feature_names is not None else FEATURES
    index = {name: position for position, name in enumerate(names)}
    rng = np.random.default_rng(seed)
    reasons = [name for name in names if name.startswith("reason_admission-")]
    origins = [name for name in names if name.startswith("origin-")]
    categories = [name for name in names if name.startswith("patient_category-")]
    destinations = [name for name in names if name.startswith("destination-")]
    culture_types = [name for name in names if name.startswith("culture_type-")]
    antibiotic_names = [name for name in names if name.startswith("antibiotics-")]
    culture_columns = [name for name in names if feature_group(name) == "culture"]
    resistance_columns = [name for name in names if feature_group(name) == "resistance"]
    icu_columns = [name for name in names if name.startswith("icu_")]

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    years: list[int] = []
    patients: list[str] = []
    orders: list[int] = []

    for patient_number in range(n_patients):
        year = int(rng.integers(start_year, end_year + 1))
        length_of_stay = int(rng.integers(6, 36))
        age = float(rng.integers(30, 90))
        gender = int(rng.integers(0, 2))
        reason = str(rng.choice(reasons)) if reasons else ""
        origin = str(rng.choice(origins)) if origins else ""
        category = str(rng.choice(categories)) if categories else ""
        destination = str(rng.choice(destinations)) if destinations else ""
        respiratory = int(reason == "reason_admission-acute_respiratory_failure")
        culture_days = sorted(int(day) for day in rng.choice(np.arange(2, length_of_stay), size=int(rng.integers(1, 4)), replace=False))
        courses = []
        for _ in range(int(rng.integers(1, 4))):
            if not antibiotic_names:
                break
            drug = str(rng.choice(antibiotic_names))
            start = int(rng.integers(0, length_of_stay))
            stop = int(rng.integers(start + 1, length_of_stay + 1))
            courses.append((drug, start, stop))
        vent_start = int(rng.integers(0, max(1, length_of_stay // 2)))
        vent_stop = int(rng.integers(vent_start, length_of_stay))
        sample = str(rng.choice(culture_types)) if culture_types else ""

        labels_by_day: dict[int, np.ndarray] = {}
        for culture_day in culture_days:
            prior = _prior_label(labels_by_day, culture_day)
            cipro = _drug_on(courses, "antibiotics-ciprofloxacina_j01ma", culture_day)
            colistin = _drug_on(courses, "antibiotics-colistina_j01xb", culture_day)
            amikacin = _drug_on(courses, "antibiotics-amikacina_j01gb", culture_day)
            ventilated = int(vent_start <= culture_day < vent_stop)
            label = np.full(len(FAMILIES), np.nan, dtype=np.float32)
            for family_id, family in enumerate(FAMILIES):
                if family == "pol" and year < 2007:
                    continue
                prior_value = 0.0 if prior is None else float(prior[family_id])
                label[family_id] = _draw_label(
                    rng,
                    family=family,
                    year=year,
                    prior=prior_value,
                    cipro=cipro,
                    colistin=colistin,
                    amikacin=amikacin,
                    gender=gender,
                    respiratory=respiratory,
                    ventilated=ventilated,
                )
            labels_by_day[culture_day] = label

        for culture_day, label in labels_by_day.items():
            window = np.full((30, len(names)), -1.0, dtype=np.float32)
            for step, day in enumerate(range(culture_day - 29, culture_day + 1)):
                if day < 0:
                    continue
                window[step] = _day_vector(
                    rng,
                    names=names,
                    index=index,
                    day=day,
                    length_of_stay=length_of_stay,
                    year=year,
                    age=age,
                    gender=gender,
                    reason=reason,
                    origin=origin,
                    category=category,
                    destination=destination,
                    culture_days=set(culture_days),
                    sample=sample,
                    labels_by_day=labels_by_day,
                    courses=courses,
                    vent_start=vent_start,
                    vent_stop=vent_stop,
                    culture_columns=culture_columns,
                    resistance_columns=resistance_columns,
                    icu_columns=icu_columns,
                )
            sequences.append(window)
            targets.append(label)
            years.append(year)
            patients.append(f"S{patient_number:04d}")
            orders.append(year * 100_000 + culture_day)

    return Cohort(
        x=np.stack(sequences),
        y=np.stack(targets),
        year=np.asarray(years),
        patient_id=np.asarray(patients),
        order=np.asarray(orders),
        feature_names=names,
    )


def _prior_label(labels_by_day: dict[int, np.ndarray], culture_day: int) -> np.ndarray | None:
    eligible = [day for day in labels_by_day if day <= culture_day - 2]
    if not eligible:
        return None
    return labels_by_day[max(eligible)]


def _drug_on(courses: list[tuple[str, int, int]], drug: str, day: int) -> int:
    return int(any(name == drug and start <= day < stop for name, start, stop in courses))


def _draw_label(rng: np.random.Generator, **factors: float | int | str) -> float:
    family = str(factors["family"])
    base = {"amg": -0.35, "car": 0.15, "cf4": -0.55, "pap": -0.45, "pol": -2.7, "qui": 0.35}[family]
    eta = (
        base
        + 0.09 * (int(factors["year"]) - 2004)
        + 1.7 * float(factors["prior"])
        + 1.15 * float(factors["cipro"])
        + 0.95 * float(factors["colistin"])
        + 0.65 * float(factors["amikacin"])
        + 0.35 * float(factors["gender"])
        + 0.6 * float(factors["respiratory"])
        + 0.45 * float(factors["ventilated"])
        + float(rng.normal(0.0, 0.35))
    )
    probability = 1.0 / (1.0 + np.exp(-eta))
    return float(rng.random() < probability)


def _day_vector(
    rng: np.random.Generator,
    *,
    names: tuple[str, ...],
    index: dict[str, int],
    day: int,
    length_of_stay: int,
    year: int,
    age: float,
    gender: int,
    reason: str,
    origin: str,
    category: str,
    destination: str,
    culture_days: set[int],
    sample: str,
    labels_by_day: dict[int, np.ndarray],
    courses: list[tuple[str, int, int]],
    vent_start: int,
    vent_stop: int,
    culture_columns: list[str],
    resistance_columns: list[str],
    icu_columns: list[str],
) -> np.ndarray:
    vector = np.zeros(len(names), dtype=np.float32)
    for name in culture_columns:
        vector[index[name]] = -1.0
    for name in resistance_columns:
        vector[index[name]] = -1.0

    def put(name: str, value: float) -> None:
        if name in index:
            vector[index[name]] = value

    put("admission", 1.0)
    put("age", age / 100.0)
    put("gender", float(gender))
    put("day_week", (day % 7) / 6.0)
    put("day_month", ((day % 28) + 1) / 31.0)
    put("month", ((year + day // 28) % 12) / 11.0)
    put("related_patients", float(rng.random() * 0.4))
    if reason:
        put(reason, 1.0)
    if category:
        put(category, 1.0)
    if day == 0 and origin:
        put(origin, 1.0)
    if day == length_of_stay - 1 and destination:
        put(destination, 1.0)

    if day in culture_days:
        for name in culture_columns:
            vector[index[name]] = 0.0
        put("cultures", 1.0)
        if sample:
            put(sample, 1.0)
        put("pseudomonas_aeruginosa_detected", 1.0)
    else:
        put("cultures", 0.0)

    prior_days = [item for item in labels_by_day if item <= day - 2]
    if prior_days:
        prior = labels_by_day[max(prior_days)]
        for family_id, family in enumerate(FAMILIES):
            column = f"resistance_{family}"
            if column in index and not np.isnan(prior[family_id]):
                vector[index[column]] = prior[family_id]
        put("pseudomonas_aeruginosa_detected", 1.0)

    for drug, start, stop in courses:
        if start <= day < stop:
            put(drug, 1.0)
    put("mechanic_ventilation", float(vent_start <= day < vent_stop))

    for name in icu_columns:
        if vector[index[name]] == 0.0:
            vector[index[name]] = float(rng.random() * 0.05)
    put("icu_admission", 1.0)
    return vector
