import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def risk_coverage(errors, uncertainty):
    error = np.asarray(errors, float)
    score = np.asarray(uncertainty, float)
    valid = np.isfinite(error) & np.isfinite(score)
    error, score = error[valid], score[valid]
    order = np.argsort(score)
    sorted_error = error[order]
    count = np.arange(1, len(sorted_error) + 1)
    return pd.DataFrame({"coverage": count / len(sorted_error),
                         "risk_mae": np.cumsum(sorted_error) / count})


def aurc(errors, uncertainty):
    curve = risk_coverage(errors, uncertainty)
    return float(np.trapezoid(curve["risk_mae"], curve["coverage"]))


def spearman_error_correlation(errors, uncertainty):
    error = np.asarray(errors, float)
    score = np.asarray(uncertainty, float)
    valid = np.isfinite(error) & np.isfinite(score)
    return float(spearmanr(error[valid], score[valid]).statistic)
