"""Uncertainty estimation methods for model inversion fitting.

Provides:
- ``ensemble_disagreement`` – extract ensemble standard deviation as a
  disagreement score.
- ``fit_residual_calibrator`` – train a quantile regression model to
  predict the 90th percentile of absolute error given disagreement
  features.
- ``predict_calibrated_uncertainty`` – apply the fitted calibrator to
  yield non-negative uncertainty estimates.
- ``mc_dropout_status`` – returns a dict indicating whether MC Dropout
  is available (always ``False`` in this pipeline because we have no
  saved model artifacts to enable it).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["ensemble_std", "ensemble_range", "mean_state_confidence",
            "depth_cm", "segment_index"]


def ensemble_disagreement(table):
    """Return per-row ensemble standard deviation as the disagreement score.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain an ``ensemble_std`` column.

    Returns
    -------
    np.ndarray
        Float array of disagreement values.  ``NaN`` entries are preserved
        (e.g. for groups with fewer than 2 models).
    """
    return table["ensemble_std"].to_numpy(float)


def fit_residual_calibrator(train, random_state=42):
    """Fit a quantile (90th percentile) regressor that predicts absolute
    prediction error from disagreement features.

    Parameters
    ----------
    train : pd.DataFrame
        Must contain the columns listed in ``FEATURES`` plus
        ``ensemble_median`` and ``true_value`` (used to compute absolute
        error).
    random_state : int, optional
        Random state passed to the underlying gradient boosting model.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline whose ``.predict()`` method returns the estimated
        90th percentile absolute error.
    """
    numeric = ["ensemble_std", "ensemble_range", "mean_state_confidence", "depth_cm"]
    categorical = ["segment_index"]
    transform = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    model = HistGradientBoostingRegressor(loss="quantile", quantile=0.9,
                                          max_iter=100, random_state=random_state)
    estimator = Pipeline([("features", transform), ("model", model)])
    absolute_error = np.abs(train["ensemble_median"] - train["true_value"])
    return estimator.fit(train[FEATURES], absolute_error)


def predict_calibrated_uncertainty(estimator, table):
    """Predict calibrated uncertainty for each row in *table*.

    Parameters
    ----------
    estimator : sklearn.pipeline.Pipeline
        Fitted pipeline from :func:`fit_residual_calibrator`.
    table : pd.DataFrame
        Must contain the columns listed in ``FEATURES``.

    Returns
    -------
    np.ndarray
        Non-negative float array of calibrated uncertainty estimates.
    """
    return np.maximum(0.0, estimator.predict(table[FEATURES]))


def mc_dropout_status():
    """Report whether MC Dropout uncertainty is available.

    In the current pipeline no saved model artifacts exist that could be
    used to run MC Dropout, so this function always returns
    ``{"available": False, "reason": "unavailable_no_model_artifacts"}``.

    Returns
    -------
    dict
    """
    return {"available": False, "reason": "unavailable_no_model_artifacts"}
