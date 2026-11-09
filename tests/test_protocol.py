import numpy as np

from amr_lstm.data import Cohort, load_cohort, save_cohort, temporal_split
from amr_lstm.schema import FAMILIES
from amr_lstm.selection.recursive import recursive_feature_count
from amr_lstm.stats import paired_bootstrap_interval
from amr_lstm.synthetic import generate_cohort


def test_a_training_patient_is_removed_from_later_splits():
    x = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
    y = np.full((4, 6), np.nan, dtype=np.float32)
    y[:, 0] = [1, 0, 1, 0]
    cohort = Cohort(
        x=x,
        y=y,
        year=np.array([2010, 2016, 2014, 2015]),
        patient_id=np.array(["A", "A", "B", "C"]),
        order=np.array([2010, 2016, 2014, 2015]),
        feature_names=("age",),
    )
    split = temporal_split(cohort, "amg", year_split=2012, seed=0)
    assert split.y_train.tolist() == [1.0]
    assert split.y_val.tolist() == [1.0]
    assert split.y_test.tolist() == [0.0]


def test_synthetic_encoding_stays_in_the_published_range():
    cohort = generate_cohort(12, seed=3)
    observed = cohort.x[cohort.x >= 0]
    assert cohort.timesteps == 30
    assert np.all((cohort.x == -1) | ((cohort.x >= 0) & (cohort.x <= 1)))
    assert observed.size > 0
    assert len(cohort.feature_names) == 398


def test_polymyxin_is_untested_before_2007():
    cohort = generate_cohort(60, seed=4)
    pol = FAMILIES.index("pol")
    early = cohort.year < 2007
    assert np.all(np.isnan(cohort.y[early, pol]))
    assert np.any(~np.isnan(cohort.y[~early, pol]))


def test_roundtrip_npz(tmp_path):
    cohort = generate_cohort(8, seed=5)
    path = tmp_path / "cohort.npz"
    save_cohort(path, cohort)
    loaded = load_cohort(path)
    assert loaded.feature_names == cohort.feature_names
    assert np.allclose(loaded.x, cohort.x)


def test_recursive_selector_returns_the_right_edge_of_the_better_half():
    accuracy = np.array([0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    sensitivity = accuracy.copy()
    specificity = accuracy.copy()
    # The left half is better. With a threshold of 3 the search stops on the
    # two leading positions and returns their right edge.
    assert recursive_feature_count(accuracy, sensitivity, specificity, threshold=3) == 2


def test_bootstrap_interval_excludes_zero_for_a_real_gap():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.70, 0.02, size=50)
    other = rng.normal(0.55, 0.02, size=50)
    interval = paired_bootstrap_interval(reference, other, repetitions=400, seed=1)
    assert interval.excludes_zero
    assert interval.mean_difference > 0

    identical = paired_bootstrap_interval(reference, reference, repetitions=200, seed=1)
    assert not identical.excludes_zero
