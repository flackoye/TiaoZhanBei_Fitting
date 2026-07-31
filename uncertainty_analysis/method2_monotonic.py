"""方案二：基于模型置信度的单调分位校准。

核心思想
--------
利用平均状态置信度（c̄_i）与真实绝对误差之间的单调关系，将置信缺失
z_i = 1 - c̄_i 分箱后统计 90% 分位误差，再用保序回归保证单调性。

参考文档
--------
docs/5.5_uncertainty_algorithm_design.md 第 5 节
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# 分箱数（文档 2.5 节：B = 15）
N_BINS = 15
# 启用区段修正的最小孔数（文档 5.6 节）
MIN_HOLES_FOR_SEGMENT_CORRECTION = 5
# 区段修正系数的上下限
SEGMENT_LAMBDA_MIN = 0.5
SEGMENT_LAMBDA_MAX = 2.0


def _compute_confidence_deficit(table: pd.DataFrame) -> np.ndarray:
    """计算置信缺失 z_i = 1 - c̄_i。

    Parameters
    ----------
    table : pd.DataFrame
        必须包含 ``mean_state_confidence`` 列。

    Returns
    -------
    np.ndarray
        置信缺失值数组。
    """
    conf = table["mean_state_confidence"].to_numpy(float, copy=True)
    # 缺失置信度按 0.5 中性值处理
    conf = np.nan_to_num(conf, nan=0.5)
    return 1.0 - conf


def _segment_adjustment(
    table: pd.DataFrame,
    train_table: pd.DataFrame,
) -> np.ndarray:
    """计算区段修正系数 λ_s。

    仅对覆盖 >= MIN_HOLES_FOR_SEGMENT_CORRECTION 个孔的区段启用修正，
    其余区段 λ = 1.0。

    Parameters
    ----------
    table : pd.DataFrame
        待修正数据，需包含 ``segment_index`` 和 ``source_file``。
    train_table : pd.DataFrame
        训练数据，用于统计每区段的孔数和估计修正系数。

    Returns
    -------
    np.ndarray
        每个样本对应的 λ_s 值。
    """
    # 统计训练数据中每区段的孔数
    seg_holes = train_table.groupby("segment_index")["source_file"].nunique()
    # 统计训练数据中每区段的中位绝对误差（相对于总体中位误差）
    train_error = np.abs(
        train_table["ensemble_median"].to_numpy(float)
        - train_table["true_value"].to_numpy(float)
    )
    overall_median_error = np.median(train_error)

    seg_medians = {}
    for seg in seg_holes.index:
        if seg_holes[seg] >= MIN_HOLES_FOR_SEGMENT_CORRECTION:
            mask = train_table["segment_index"] == seg
            seg_median = np.median(train_error[mask])
            # λ = 区段中位误差 / 总体中位误差，钳制到 [0.5, 2.0]
            lam = np.clip(
                seg_median / overall_median_error if overall_median_error > 0 else 1.0,
                SEGMENT_LAMBDA_MIN,
                SEGMENT_LAMBDA_MAX,
            )
            seg_medians[seg] = lam
        else:
            seg_medians[seg] = 1.0

    return np.array([seg_medians.get(s, 1.0) for s in table["segment_index"]])


def fit_monotonic_calibrator(
    train: pd.DataFrame,
    n_bins: int = N_BINS,
    use_segment_correction: bool = False,
) -> dict:
    """训练单调分位校准器。

    步骤（文档 5.3-5.4 节）：
    1. 计算置信缺失 z_i = 1 - c̄_i；
    2. 按 z_i 分位数分为 B 个区间；
    3. 每区间计算绝对误差的 90% 分位数；
    4. 用保序回归拟合非递减函数。

    Parameters
    ----------
    train : pd.DataFrame
        训练位置表，需包含 ``mean_state_confidence``、``ensemble_median``、
        ``true_value``、``segment_index``、``source_file``。
    n_bins : int, optional
        分箱数，默认 15。
    use_segment_correction : bool, optional
        是否启用区段修正，默认 False。

    Returns
    -------
    dict
        包含校准器参数的字典：
        - ``z_bin_edges``: 分箱边界
        - ``z_bin_centers``: 每箱 z 中位数
        - ``q_bin``: 每箱 90% 分位误差
        - ``isotonic``: 保序回归模型
        - ``use_segment_correction``: 是否启用区段修正
        - ``train_table``: 训练数据（区段修正所需）
    """
    z = _compute_confidence_deficit(train)
    error = np.abs(
        train["ensemble_median"].to_numpy(float)
        - train["true_value"].to_numpy(float)
    )

    # 按 z 分位数分箱
    bin_edges = np.percentile(z, np.linspace(0, 100, n_bins + 1))
    # 确保边界唯一
    bin_edges = np.unique(bin_edges)

    # 处理边界情况：所有 z 值相同 → 只有一个分箱
    if len(bin_edges) < 2:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        q_90 = np.percentile(error, 90)
        iso.fit([0.0, 1.0], [q_90, q_90])
        calibrator = {
            "z_bin_edges": bin_edges,
            "z_bin_centers": np.array([0.5]),
            "q_bin": np.array([q_90]),
            "isotonic": iso,
            "use_segment_correction": use_segment_correction,
            "train_table": train.copy() if use_segment_correction else None,
        }
        return calibrator

    bin_indices = np.digitize(z, bin_edges, right=True)
    # 将超出范围的归入最后一个箱子
    bin_indices = np.clip(bin_indices, 1, len(bin_edges) - 1) - 1

    n_actual_bins = len(bin_edges) - 1
    z_centers = np.empty(n_actual_bins)
    q_bin = np.empty(n_actual_bins)

    for b in range(n_actual_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            z_centers[b] = np.median(z[mask])
            q_bin[b] = np.percentile(error[mask], 90)
        else:
            # 空箱：使用相邻箱的线性插值
            z_centers[b] = (bin_edges[b] + bin_edges[b + 1]) / 2
            q_bin[b] = 0.0

    # 保序回归：保证非递减
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(z_centers, q_bin)

    calibrator = {
        "z_bin_edges": bin_edges,
        "z_bin_centers": z_centers,
        "q_bin": q_bin,
        "isotonic": iso,
        "use_segment_correction": use_segment_correction,
        "train_table": train.copy() if use_segment_correction else None,
    }
    return calibrator


def predict_monotonic_uncertainty(
    calibrator: dict,
    table: pd.DataFrame,
) -> np.ndarray:
    """用训练好的单调校准器预测不确定性。

    Parameters
    ----------
    calibrator : dict
        ``fit_monotonic_calibrator`` 返回的校准器。
    table : pd.DataFrame
        待预测数据。

    Returns
    -------
    np.ndarray
        非负不确定性估计值。
    """
    z = _compute_confidence_deficit(table)
    iso = calibrator["isotonic"]
    uncertainty = iso.predict(z)
    uncertainty = np.maximum(0.0, uncertainty)

    # 可选区段修正
    if calibrator.get("use_segment_correction") and calibrator["train_table"] is not None:
        lambdas = _segment_adjustment(table, calibrator["train_table"])
        uncertainty = uncertainty * lambdas

    return uncertainty


def monotonic_disagreement(table: pd.DataFrame) -> np.ndarray:
    """简易版本的单调不确定性：直接使用置信缺失百分位作为不确定性分数。

    此函数不需要训练数据，仅用于快速预览或作为方案三的置信缺失分量。

    Parameters
    ----------
    table : pd.DataFrame
        必须包含 ``mean_state_confidence`` 列。

    Returns
    -------
    np.ndarray
        置信缺失的经验百分位（0-1）。
    """
    z = _compute_confidence_deficit(table)
    # 使用经验百分位
    ranks = np.argsort(np.argsort(z))
    percentile = ranks.astype(float) / max(len(z) - 1, 1)
    return percentile
