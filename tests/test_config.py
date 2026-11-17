from amr_lstm.config import StudyConfig, ensemble_bounds


def test_published_ensemble_window():
    assert ensemble_bounds(30) == (11, 20)
    published = StudyConfig.paper()
    assert published.ensemble_epochs == 30
    assert (published.ensemble_first_epoch, published.ensemble_reference_epoch) == (11, 20)


def test_shorter_ensemble_keeps_the_same_fractions():
    assert ensemble_bounds(4) == (2, 3)
    assert ensemble_bounds(1) == (1, 1)
