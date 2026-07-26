import numpy as np
from uncertainty_analysis.evaluation import aurc, spearman_error_correlation


def test_perfect_ranking_beats_reversed_ranking():
    errors = np.array([0.0, 1.0, 2.0, 10.0])
    assert aurc(errors, errors) < aurc(errors, -errors)
    assert spearman_error_correlation(errors, errors) == 1.0


def test_metrics_ignore_nonfinite_pairs():
    assert np.isfinite(aurc(np.array([1., 2., np.nan]), np.array([.1, .2, .3])))
