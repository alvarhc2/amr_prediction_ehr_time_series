"""Reported cohort sizes and test-set AUC from the published experiments.

Fractions in the paper are stored here as percentages, matching the tables.
These constants describe the hospital study. They are not computed from the
synthetic cohort.
"""

from __future__ import annotations

FAMILIES = ("AMG", "CAR", "CF4", "PAP", "POL", "QUI")

# Susceptible / resistant counts. Train is 2004–2012. Validation is the earlier
# half of 2013–2019 and test is the later half, after removing patients who
# already appear in training.
SPLIT_COUNTS = {
    "AMG": {"train": (324, 221), "val": (30, 73), "test": (28, 76)},
    "CAR": {"train": (205, 234), "val": (10, 90), "test": (11, 90)},
    "CF4": {"train": (331, 207), "val": (49, 55), "test": (37, 67)},
    "PAP": {"train": (331, 207), "val": (42, 62), "test": (43, 61)},
    "POL": {"train": (275, 9), "val": (90, 8), "test": (67, 31)},
    "QUI": {"train": (250, 288), "val": (28, 55), "test": (7, 77)},
}

SELECTED_FEATURES = {"AMG": 36, "CAR": 18, "CF4": 67, "PAP": 24, "POL": 55, "QUI": 49}
BOOTSTRAP_VOTING_FEATURES = {"AMG": 99, "CAR": 94, "CF4": 86, "PAP": 56, "POL": 46, "QUI": 187}
GROUP_LASSO_FEATURES = {"AMG": 61, "CAR": 96, "CF4": 100, "PAP": 103, "POL": 41, "QUI": 93}

# Mean test AUC (percent) over 50 random initializations.
TEST_AUC = {
    "AMG": {"all": 60.17, "proposed": 76.60, "ensemble": 76.77},
    "CAR": {"all": 68.88, "proposed": 65.93, "ensemble": 66.53},
    "CF4": {"all": 57.96, "proposed": 62.77, "ensemble": 63.45},
    "PAP": {"all": 63.73, "proposed": 71.19, "ensemble": 71.63},
    "POL": {"all": 37.13, "proposed": 41.58, "ensemble": 42.32},
    "QUI": {"all": 75.31, "proposed": 88.10, "ensemble": 89.37},
}
