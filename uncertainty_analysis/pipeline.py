"""Cross-validation, method selection, and weight conversion for uncertainty
estimates.

This module composes the data-building, uncertainty-estimation, and evaluation
functions from the sibling modules into a leave-one-source_file-out
cross-validation pipeline and provides helpers for selecting the best
uncertainty method and converting uncertainty scores into observation weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uncertainty_analysis.evaluation import aurc, spearman_error_correlation
from uncertainty_analysis.methods import (
    FEATURES,
    ensemble_disagreement,
    fit_residual_calibrator,
    predict_calibrated_uncertainty,
)
from uncertainty_analysis.method2_monotonic import (
    fit_monotonic_calibrator,
    predict_monotonic_uncertainty,
)
from uncertainty_analysis.method3_fusion import (
    compute_disagreement_percentiles,
    compute_confidence_deficit_percentile,
    compute_local_instability_percentile,
    generate_fusion_candidates,
)

# Minimum number of test samples required to evaluate a fold.
_MIN_TEST_SAMPLES = 3

# 融合系数搜索时用于参数选择的指标权重
_AURC_WEIGHT = 1.0
_SPEARMAN_WEIGHT = 0.3  # Spearman 作为辅助指标


def _evaluate_fusion_on_fold(
    test_error: np.ndarray,
    test_table: pd.DataFrame,
    window: int = 21,
    eta: float = 0.3,
) -> list[dict]:
    """评估一个折上所有融合系数组合的性能。

    预计算三个证据分量，然后对不同系数组合做加权求和，
    避免重复计算局部不稳定分量（最耗时）。

    Parameters
    ----------
    test_error : np.ndarray
        测试集绝对误差。
    test_table : pd.DataFrame
        测试位置表。
    window : int, optional
        局部窗口大小，默认 21。
    eta : float, optional
        边界保护削弱系数，默认 0.3。

    Returns
    -------
    list[dict]
        每个系数组合的评价结果。
    """
    # 预计算三个证据分量（只算一次）
    a = compute_disagreement_percentiles(test_table)
    b = compute_confidence_deficit_percentile(test_table)
    c = compute_local_instability_percentile(test_table, window=window, eta=eta)

    records = []
    for alpha, beta, gamma in generate_fusion_candidates():
        s = alpha * a + beta * b + gamma * c
        records.append({
            "method": "fusion",
            "fusion_alpha": alpha,
            "fusion_beta": beta,
            "fusion_gamma": gamma,
            "aurc": aurc(test_error, s),
            "spearman": spearman_error_correlation(test_error, s),
            "n_test_samples": len(test_table),
        })
    return records


def cross_validate_methods(
    table: pd.DataFrame,
    random_state: int = 42,
    methods: list[str] | None = None,
    fusion_window: int = 21,
    fusion_eta: float = 0.3,
) -> pd.DataFrame:
    """Leave-one-*source_file*-out cross-validation of uncertainty methods.

    For each unique ``source_file`` in *table*:

    1. Training fold consists of all rows whose ``source_file`` differs from
       the test file.
    2. On the test fold, compute:
       - **ensemble disagreement** (unsupervised)
       - **calibrated uncertainty** (residual quantile regression)
       - **monotonic uncertainty** (confidence-based monotonic calibration)
       - **fusion uncertainty** (multi-evidence fusion, grid search over coefficients)
    3. Absolute error on the test fold is ``abs(ensemble_median - true_value)``.
    4. All methods are evaluated with AURC and Spearman correlation.

    Parameters
    ----------
    table : pd.DataFrame
        Position table produced by
        :func:`uncertainty_analysis.data.build_position_table`.  Must contain
        ``source_file``, ``ensemble_median``, ``true_value``, and the columns
        listed in :const:`uncertainty_analysis.methods.FEATURES`.
    random_state : int, optional
        Random state passed to :func:`fit_residual_calibrator`.
    methods : list[str], optional
        Methods to evaluate. Default evaluates all: ``["disagreement",
        "calibrated", "monotonic", "fusion"]``.
    fusion_window : int, optional
        Local window size for fusion method, default 21.
    fusion_eta : float, optional
        Boundary protection coefficient for fusion method, default 0.3.

    Returns
    -------
    pd.DataFrame
        Columns: ``source_file``, ``method``, ``aurc``, ``spearman``,
        ``n_test_samples``, and fusion-specific columns (``fusion_alpha``,
        ``fusion_beta``, ``fusion_gamma``) if applicable.
        Folds with fewer than :data:`_MIN_TEST_SAMPLES` rows in the training
        set are omitted.
    """
    if methods is None:
        methods = ["calibrated", "monotonic", "fusion"]

    records = []
    for test_file in sorted(table["source_file"].unique()):
        train = table[table["source_file"] != test_file].reset_index(drop=True)
        test = table[table["source_file"] == test_file].reset_index(drop=True)

        if len(test) < _MIN_TEST_SAMPLES or len(train) < _MIN_TEST_SAMPLES:
            continue

        # Compute test error
        test_error = np.abs(test["ensemble_median"].to_numpy(float)
                            - test["true_value"].to_numpy(float))

        # --- 方案一：条件分位残差回归 (calibrated) ---
        if "calibrated" in methods:
            calibrator = fit_residual_calibrator(train, random_state=random_state)
            calibrated = predict_calibrated_uncertainty(calibrator, test)
            records.append({
                "source_file": test_file,
                "method": "calibrated",
                "fusion_alpha": None,
                "fusion_beta": None,
                "fusion_gamma": None,
                "aurc": aurc(test_error, calibrated),
                "spearman": spearman_error_correlation(test_error, calibrated),
                "n_test_samples": len(test),
            })

        # --- 方案二：基于模型置信度的单调分位校准 (monotonic) ---
        if "monotonic" in methods:
            mono_cal = fit_monotonic_calibrator(train, use_segment_correction=False)
            mono_unc = predict_monotonic_uncertainty(mono_cal, test)
            records.append({
                "source_file": test_file,
                "method": "monotonic",
                "fusion_alpha": None,
                "fusion_beta": None,
                "fusion_gamma": None,
                "aurc": aurc(test_error, mono_unc),
                "spearman": spearman_error_correlation(test_error, mono_unc),
                "n_test_samples": len(test),
            })

        # --- 方案三：多证据融合 (fusion) ---
        if "fusion" in methods:
            fusion_records = _evaluate_fusion_on_fold(
                test_error, test, window=fusion_window, eta=fusion_eta
            )
            for rec in fusion_records:
                rec["source_file"] = test_file
                records.append(rec)

    return pd.DataFrame(records)


def select_method(metrics: pd.DataFrame) -> str:
    """Select the best uncertainty method from cross-validation metrics.

    Ranking (descending priority):

    1. Lower mean AURC (primary).
    2. Higher mean Spearman correlation (secondary).
    3. Earlier method name alphabetically (final tiebreaker).

    For the "fusion" method, the best coefficient combination is selected
    first, then compared against other methods.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain columns ``method``, ``aurc``, ``spearman``, and
        optionally ``fusion_alpha``/``fusion_beta``/``fusion_gamma``.

    Returns
    -------
    str
        Name of the selected method. For the fusion method, the method name
        includes coefficient info (e.g., ``"fusion_a0.4_b0.4_g0.2"``).
    """
    # For fusion, create a combined method label that includes coefficients
    df = metrics.copy()
    fusion_mask = df["method"] == "fusion"
    if fusion_mask.any():
        df.loc[fusion_mask, "method"] = df.loc[fusion_mask].apply(
            lambda r: f"fusion_a{r['fusion_alpha']}_b{r['fusion_beta']}_g{r['fusion_gamma']}",
            axis=1,
        )

    grouped = df.groupby("method").agg(
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


def select_best_fusion_coefficients(
    metrics: pd.DataFrame,
) -> tuple[float, float, float]:
    """从 CV 指标中选择最佳融合系数。

    Parameters
    ----------
    metrics : pd.DataFrame
        CV 结果表，需包含 ``method``、``aurc``、``spearman``、
        ``fusion_alpha``、``fusion_beta``、``fusion_gamma`` 列。

    Returns
    -------
    tuple[float, float, float]
        最佳 (alpha, beta, gamma)。
    """
    fusion = metrics[metrics["method"] == "fusion"].copy()
    if fusion.empty:
        return (0.4, 0.4, 0.2)  # 默认均衡值

    grouped = fusion.groupby(["fusion_alpha", "fusion_beta", "fusion_gamma"]).agg(
        mean_aurc=("aurc", "mean"),
        mean_spearman=("spearman", "mean"),
    ).reset_index()

    # 综合评分（AURC 越低越好，Spearman 越高越好）
    # 归一化后加权
    aurc_min, aurc_max = grouped["mean_aurc"].min(), grouped["mean_aurc"].max()
    sp_min, sp_max = grouped["mean_spearman"].min(), grouped["mean_spearman"].max()

    aurc_range = aurc_max - aurc_min if aurc_max > aurc_min else 1.0
    sp_range = sp_max - sp_min if sp_max > sp_min else 1.0

    # 综合分数 = -AURC_weight * 归一化AURC + Spearman_weight * 归一化Spearman
    grouped["score"] = (
        -_AURC_WEIGHT * (grouped["mean_aurc"] - aurc_min) / aurc_range
        + _SPEARMAN_WEIGHT * (grouped["mean_spearman"] - sp_min) / sp_range
    )

    best = grouped.sort_values("score", ascending=False).iloc[0]
    return (float(best["fusion_alpha"]), float(best["fusion_beta"]), float(best["fusion_gamma"]))
