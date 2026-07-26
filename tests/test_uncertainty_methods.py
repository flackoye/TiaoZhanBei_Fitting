"""Tests for uncertainty_analysis.methods."""

import numpy as np
import pandas as pd
import pytest
from uncertainty_analysis.methods import (
    ensemble_disagreement,
    fit_residual_calibrator,
    mc_dropout_status,
    predict_calibrated_uncertainty,
)


def test_disagreement_is_monotonic_and_preserves_missing():
    table = pd.DataFrame({"ensemble_std": [0.0, 2.0, np.nan]})
    score = ensemble_disagreement(table)
    assert score[0] < score[1]
    assert np.isnan(score[2])


def test_mc_dropout_is_explicitly_unavailable():
    status = mc_dropout_status()
    assert status == {"available": False, "reason": "unavailable_no_model_artifacts"}


def test_residual_calibrator_penalises_high_disagreement():
    """Synthetic check: rows with larger ensemble_std have larger absolute
    error, so the calibrator's predicted uncertainty should be higher for
    high-disagreement rows after fitting."""
    rng = np.random.RandomState(42)
    n = 50
    train = pd.DataFrame({
        "ensemble_std": rng.uniform(0, 5, n),
        "ensemble_range": rng.uniform(0, 10, n),
        "mean_state_confidence": rng.uniform(0, 1, n),
        "depth_cm": rng.uniform(0, 100, n),
        "segment_index": [0] * n,
        "ensemble_median": [50.0] * n,
        "true_value": 50.0 + rng.uniform(-1, 1, n),
    })
    # Inject structure: rows with high ensemble_std get bigger true errors
    high_mask = train["ensemble_std"] > 2.5
    train.loc[high_mask, "true_value"] = (
        50.0 + train.loc[high_mask, "ensemble_std"] * 2
    )

    estimator = fit_residual_calibrator(train, random_state=42)
    test = pd.DataFrame({
        "ensemble_std": [0.1, 4.9],
        "ensemble_range": [0.2, 9.8],
        "mean_state_confidence": [0.9, 0.1],
        "depth_cm": [50.0, 50.0],
        "segment_index": [0, 0],
    })
    preds = predict_calibrated_uncertainty(estimator, test)
    assert preds[0] < preds[1], (
        f"Expected low-disagreement row to have smaller calibrated uncertainty, "
        f"got {preds[0]} vs {preds[1]}"
    )
