from amr_lstm.published import FAMILIES, SELECTED_FEATURES, TEST_AUC
from amr_lstm.schema import FEATURES, clinical_name, feature_group, group_counts


def test_vocabulary_has_the_published_width():
    assert len(FEATURES) == 398
    assert len(set(FEATURES)) == 398
    counts = group_counts()
    assert sum(counts.values()) == 398
    assert counts["temporal"] == 3
    assert counts["resistance"] == 7


def test_clinical_alias_used_in_the_paper_table():
    assert clinical_name("antibiotics-ciprofloxacina_j01ma") == "antibiotics-ciprofloxacin"
    assert feature_group("resistance_qui") == "resistance"
    assert feature_group("icu_antibiotics-colistina_j01xb") == "icu_treatment"
    assert feature_group("related_patients") == "coadmitted"


def test_published_tables_cover_every_family():
    assert set(SELECTED_FEATURES) == set(FAMILIES)
    assert set(TEST_AUC) == set(FAMILIES)
    assert SELECTED_FEATURES["CAR"] == 18
    assert TEST_AUC["QUI"]["ensemble"] == 89.37
