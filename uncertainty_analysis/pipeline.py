"""Cross-validation, method selection, and weight conversion for uncertainty
estimates.

This module composes the data-building, uncertainty-estimation, and evaluation
functions from the sibling modules into a leave-one-source_file-out
cross-validation pipeline and provides helpers for selecting the best
uncertainty method and converting uncertainty scores into observation weights.
"""

import numpy as np
import pandas as pd

from uncertainty_analysis.evaluation import aurc, spearman_error_correlation
from uncertainty_analysis.methods import (
    FEATURES,
    ensemble_disagreement,
    fit_residual_calibrator,
    predict_calibrated_uncertainty,
)

# Minimum number of test samples required to evaluate a fold.
_MIN_TEST_SAMPLES = 3


def cross_validate_methods(
    table: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-one-*source_file*-out cross-validation of uncertainty methods.

    For each unique ``source_file`` in *table*:

    1. Training fold consists of all rows whose ``source_file`` differs from
       the test file.
    2. On the test fold, compute **ensemble disagreement** (unsupervised) and
       **calibrated uncertainty** by fitting a residual calibrator on the
       training fold and predicting on the test fold.
    3. Absolute error on the test fold is ``abs(ensemble_median - true_value)``.
    4. Both methods are evaluated with AURC and Spearman correlation.

    Parameters
    ----------
    table : pd.DataFrame
        Position table produced by
        :func:`uncertainty_analysis.data.build_position_table`.  Must contain
        ``source_file``, ``ensemble_median``, ``true_value``, and the columns
        listed in :const:`uncertainty_analysis.methods.FEATURES`.
    random_state : int, optional
        Random state passed to :func:`fit_residual_calibrator`.

    Returns
    -------
    pd.DataFrame
        Columns: ``source_file``, ``method`` (``"disagreement"`` |
        ``"calibrated"``), ``aurc``, ``spearman``, ``n_test_samples``.
        Folds with fewer than :data:`_MIN_TEST_SAMPLES` rows in the training
        set are omitted.
    """
    records = []
    for test_file in sorted(table["source_file"].unique()):
        train = table[table["source_file"] != test_file].reset_index(drop=True)
        test = table[table["source_file"] == test_file].reset_index(drop=True)

        if len(test) < _MIN_TEST_SAMPLES or len(train) < _MIN_TEST_SAMPLES:
            continue

        # Compute test error
        test_error = np.abs(test["ensemble_median"].to_numpy(float)
                            - test["true_value"].to_numpy(float))

        # --- Method 1: ensemble disagreement (unsupervised) ---
        disagreement = ensemble_disagreement(test)
        records.append({
            "source_file": test_file,
            "method": "disagreement",
            "aurc": aurc(test_error, disagreement),
            "spearman": spearman_error_correlation(test_error, disagreement),
            "n_test_samples": len(test),
        })

        # --- Method 2: calibrated uncertainty ---
        calibrator = fit_residual_calibrator(train, random_state=random_state)
        calibrated = predict_calibrated_uncertainty(calibrator, test)
        records.append({
            "source_file": test_file,
            "method": "calibrated",
            "aurc": aurc(test_error, calibrated),
            "spearman": spearman_error_correlation(test_error, calibrated),
            "n_test_samples": len(test),
        })

    return pd.DataFrame(records)


def select_method(metrics: pd.DataFrame) -> str:
    """Select the best uncertainty method from cross-validation metrics.

    Ranking (descending priority):

    1. Lower mean AURC (primary).
    2. Higher mean Spearman correlation (secondary).
    3. Earlier method name alphabetically (final tiebreaker).

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain columns ``method``, ``aurc``, ``spearman``.

    Returns
    -------
    str
        Name of the selected method.
    """
    grouped = metrics.groupby("method").agg(
        mean_aurc=("aurc", "mean"),
        mean_spearman=("spearman", "mean"),
    ).reset_index()

    grouped = grouped.sort_values(
        by=["mean_aurc", "mean_spearman", "method"],
        ascending=[True, False, True],
    )
    return grouped.iloc[0]["method"]


def uncertainty_to_weight(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    minimum: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert uncertainty scores to observation weights via empirical CDF.

    For each test score, compute its percentile rank within the training
    distribution.  A score that is high (at or near the maximum of the
    training distribution) maps to a low confidence and weight; a score that
    is low maps to high confidence and weight.

    Parameters
    ----------
    train_scores : np.ndarray
        1-D array of uncertainty scores from the training set (used to
        define the empirical CDF).
    test_scores : np.ndarray
        1-D array of uncertainty scores to convert.
    minimum : float, optional
        Minimum weight value (clamps the lower end).  Default 0.1.

    Returns
    -------
    confidence : np.ndarray
        Float array in ``[0, 1]``: ``1.0 - percentile``.
    weight : np.ndarray
        Float array in ``[minimum, 1.0]``.
    """
    train = np.sort(np.asarray(train_scores, float))
    test = np.asarray(test_scores, float)
    percentile = np.searchsorted(train, test, side="right") / len(train)
    confidence = np.clip(1.0 - percentile, 0.0, 1.0)
    weight = minimum + (1.0 - minimum) * confidence
    return confidence, weight
