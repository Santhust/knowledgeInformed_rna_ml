"""
Shared evaluation protocol for the M0-M3 model comparison.

WHY THIS EXISTS
---------------
The M0-M3 conditions must be compared on *identical* data partitions,
otherwise a difference in performance cannot be attributed to the prior
under test. This module is the single definition of those partitions, and it
is imported by every model condition so that they cannot drift apart.

TWO PROTOCOLS, CLEARLY DISTINCT
-------------------------------
1. `make_holdout_split` -- the 75/25 "fixed holdout diagnostic split".
   This is a DIAGNOSTIC split, not an untouched final test set. Its held-out
   results were inspected while deciding how to handle an overfitting
   default configuration in the HistGradientBoosting baseline, so the split
   is no longer independent of a modelling decision. It is retained because
   it is transparent and cheap, and because it is the only place where the
   overfitting behaviour of the default configuration is visible. It is NOT
   the basis for M0-M3 comparisons. It has not been re-run with a different
   seed to obtain friendlier numbers, and must not be.

2. `make_cv` -- stratified 5-fold cross-validation over the full observable
   dataset. THIS IS THE PRIMARY PROTOCOL. It is deterministic: the same seed
   and fold count always produce the same partition, so every model condition
   can be evaluated on exactly the same folds and compared fold by fold.

   Preprocessing is fitted INSIDE each training fold. A scaler is never fit on
   the full dataset before splitting, which would leak test-fold statistics
   into training.

SCIENTIFIC STATUS
    The dataset is synthetic. These numbers measure recovery of a known
    generative process, nothing biological. No biological validity is claimed.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Shared, fixed, and never tuned. Changing either value would invalidate every
# cross-model comparison made with this protocol.
HOLDOUT_TEST_SIZE = 0.25
HOLDOUT_RANDOM_STATE = 0

CV_N_SPLITS = 5
CV_RANDOM_STATE = 0

# Primary-protocol metrics reported for every fold. Kept here so M0-M3 cannot
# report different metric sets.
PRIMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "f1",
    "brier",
]


def make_holdout_split(y: np.ndarray):
    """DIAGNOSTIC ONLY -- the fixed 75/25 holdout.

    Retained for transparency and for showing the overfitting behaviour of the
    default boosting configuration. Not the primary comparison protocol; see
    the module docstring.
    """
    indices = np.arange(len(y))
    return train_test_split(
        indices,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        stratify=y,
    )


def make_cv() -> StratifiedKFold:
    """The primary protocol: deterministic stratified 5-fold CV.

    Every model condition must use this exact object so that fold-by-fold
    comparison is valid.
    """
    return StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)


def fold_assignment(y: np.ndarray) -> list[list[int]]:
    """The concrete test indices of each fold.

    Exposed so that any later model condition can assert it received exactly
    the same partition as every other condition. This is the mechanism that
    makes fold-by-fold comparison trustworthy.
    """
    return [list(test_idx) for _, test_idx in make_cv().split(np.zeros(len(y)), y)]


def fold_fingerprint(y: np.ndarray) -> str:
    """Short stable digest of the fold assignment, for reproducibility checks."""
    import hashlib

    payload = ";".join(
        ",".join(str(i) for i in fold) for fold in fold_assignment(y)
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def assert_identical_folds(y: np.ndarray, expected: list[list[int]]) -> None:
    """Fail if a model condition was evaluated on different folds."""
    actual = fold_assignment(y)
    assert len(actual) == len(expected), (
        f"fold count differs: {len(actual)} vs expected {len(expected)}"
    )
    for k, (got, want) in enumerate(zip(actual, expected)):
        assert list(got) == list(want), f"fold {k} assignment differs from the shared protocol"
