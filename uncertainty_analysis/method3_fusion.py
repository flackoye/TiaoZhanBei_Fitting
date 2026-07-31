"""方案三：模型分歧—置信缺失—局部不稳定多证据融合。

核心思想
--------
融合三类互补的不确定性证据：
1. 模型分歧（标准差 + 极差的百分位均值）
2. 置信缺失（1 - 平均置信度的百分位）
3. 局部不稳定（MAD 稳健偏离度，含边界保护）

参考文献
--------
docs/5.5_uncertainty_algorithm_design.md 第 6 节
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# 局部窗口大小（文档 2.5 节：21 点，半径 10）
LOCAL_WINDOW = 21
LOCAL_HALF_WINDOW = LOCAL_WINDOW // 2  # 10
# 边界保护连续点数阈值（文档 2.5 节：≥ 3）
BOUNDARY_RUN_LENGTH = 3
# 边界保护削弱系数（文档 2.5 节：η = 0.3）
BOUNDARY_ETA = 0.3
# 防止除零
EPS = 1e-10

# 融合系数搜索网格（文档 2.5 节：{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}）
FUSION_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def compute_disagreement_percentiles(table: pd.DataFrame) -> np.ndarray:
    """计算模型分歧百分位分量 a_i。

    对每个测点，取 (标准差百分位 + 极差百分位) / 2。

    Parameters
    ----------
    table : pd.DataFrame
        必须包含 ``ensemble_std`` 和 ``ensemble_range`` 列。

    Returns
    -------
    np.ndarray
        模型分歧百分位，范围 [0, 1]。
    """
    std = table["ensemble_std"].to_numpy(float, copy=True)
    rng = table["ensemble_range"].to_numpy(float, copy=True)

    # 单模型测点用中位数填充
    median_std = np.nanmedian(std)
    median_rng = np.nanmedian(rng)
    std = np.nan_to_num(std, nan=median_std)
    rng = np.nan_to_num(rng, nan=median_rng)

    # 经验百分位（排序后插值）
    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        n = len(values)
        ranks = np.argsort(np.argsort(values))
        return ranks.astype(float) / max(n - 1, 1)

    p_std = _percentile_rank(std)
    p_rng = _percentile_rank(rng)
    return (p_std + p_rng) / 2.0


def compute_confidence_deficit_percentile(table: pd.DataFrame) -> np.ndarray:
    """计算置信缺失百分位分量 b_i。

    z_i = 1 - c̄_i 的经验百分位。

    Parameters
    ----------
    table : pd.DataFrame
        必须包含 ``mean_state_confidence`` 列。

    Returns
    -------
    np.ndarray
        置信缺失百分位，范围 [0, 1]。
    """
    conf = table["mean_state_confidence"].to_numpy(float, copy=True)
    conf = np.nan_to_num(conf, nan=0.5)
    z = 1.0 - conf
    ranks = np.argsort(np.argsort(z))
    return ranks.astype(float) / max(len(z) - 1, 1)


def _compute_local_mad(
    y: np.ndarray,
    window: int = LOCAL_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口计算局部中位数和局部 MAD。

    Parameters
    ----------
    y : np.ndarray
        输入序列。
    window : int, optional
        窗口大小，默认 21。

    Returns
    -------
    local_median : np.ndarray
        局部中位数。
    local_mad : np.ndarray
        局部中位绝对偏差。
    """
    n = len(y)
    local_median = np.full(n, np.nan)
    local_mad = np.full(n, np.nan)
    half_w = window // 2

    for i in range(n):
        left = max(0, i - half_w)
        right = min(n, i + half_w + 1)
        segment = y[left:right]
        med = np.median(segment)
        local_median[i] = med
        local_mad[i] = np.median(np.abs(segment - med))

    return local_median, local_mad


def _detect_boundary(y: np.ndarray, local_median: np.ndarray) -> np.ndarray:
    """检测受保护的真实边界。

    条件（文档 6.5 节）：
    1. ≥ BOUNDARY_RUN_LENGTH 个连续点同方向偏离局部中位数；
    2. 突变后形成新的稳定区间（不是单点尖峰后立即恢复）。

    Parameters
    ----------
    y : np.ndarray
        原始序列（ensemble_median）。
    local_median : np.ndarray
        局部中位数序列。

    Returns
    -------
    np.ndarray
        bool 数组，True 表示该点位于受保护边界区域。
    """
    n = len(y)
    boundary = np.zeros(n, dtype=bool)

    # 判断每个点是否偏离局部中位数（偏离方向）
    deviation = y - local_median
    direction = np.sign(deviation)  # -1, 0, 1

    # 找出连续 ≥ BOUNDARY_RUN_LENGTH 个相同符号的偏离
    # 但不包括方向为 0（无偏离）的点
    run_start = None
    run_dir = 0
    run_len = 0

    for i in range(n):
        if direction[i] != 0 and direction[i] == run_dir:
            run_len += 1
        elif direction[i] != 0:
            # 新方向开始
            if run_len >= BOUNDARY_RUN_LENGTH and run_dir != 0:
                # 标记之前的连续段为边界
                for j in range(run_start, run_start + run_len):
                    boundary[j] = True
            run_start = i
            run_dir = direction[i]
            run_len = 1
        else:
            # 无偏离，重置
            if run_len >= BOUNDARY_RUN_LENGTH and run_dir != 0:
                for j in range(run_start, run_start + run_len):
                    boundary[j] = True
            run_start = None
            run_dir = 0
            run_len = 0

    # 检查最后一个段
    if run_len >= BOUNDARY_RUN_LENGTH and run_dir != 0:
        for j in range(run_start, run_start + run_len):
            boundary[j] = True

    # 进一步筛选：仅保留形成"新稳定区间"的边界。
    # 规则：如果边界后的点在窗口内恢复到原值（尖峰模式），则不保护。
    for i in range(n):
        if not boundary[i]:
            continue
        # 检查是否孤立尖峰：边界后恢复原值的点
        # 取边界前后各 half_window 点的中位数
        half_w = LOCAL_HALF_WINDOW
        after_left = min(n, i + 1)
        after_right = min(n, i + half_w + 1)
        before_left = max(0, i - half_w)
        before_right = max(0, i)

        if after_right > after_left and before_right > before_left:
            after_med = np.median(y[after_left:after_right])
            before_med = np.median(y[before_left:before_right])
            # 如果前后中位数接近（变化小于 5%），则是尖峰，取消保护
            range_y = np.nanmax(y) - np.nanmin(y)
            if range_y > 0 and abs(after_med - before_med) / range_y < 0.05:
                boundary[i] = False

    return boundary


def compute_local_instability_percentile(
    table: pd.DataFrame,
    window: int = LOCAL_WINDOW,
    eta: float = BOUNDARY_ETA,
) -> np.ndarray:
    """计算局部不稳定百分位分量 c_i（含边界保护）。

    步骤（文档 6.4-6.5 节）：
    1. 滑动窗口计算局部中位数和 MAD；
    2. 计算稳健局部偏离度 r_i；
    3. 转换为经验百分位；
    4. 检测真实边界，应用保护系数 η。

    Parameters
    ----------
    table : pd.DataFrame
        必须包含 ``ensemble_median`` 列。假设已按深度排序。
    window : int, optional
        局部窗口大小，默认 21。
    eta : float, optional
        边界保护削弱系数，默认 0.3。

    Returns
    -------
    np.ndarray
        局部不稳定百分位，范围 [0, 1]。
    """
    y = table["ensemble_median"].to_numpy(float, copy=True)

    # 1. 滑动窗口局部中位数和 MAD
    local_med, local_mad = _compute_local_mad(y, window)

    # 2. 稳健局部偏离度
    deviation = np.abs(y - local_med)
    scale = 1.4826 * local_mad + EPS
    r = deviation / scale

    # 3. 转换为经验百分位
    ranks = np.argsort(np.argsort(r))
    c = ranks.astype(float) / max(len(r) - 1, 1)

    # 4. 边界保护
    boundary = _detect_boundary(y, local_med)
    c[boundary] *= eta

    return c


def compute_fusion_score(
    table: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    window: int = LOCAL_WINDOW,
    eta: float = BOUNDARY_ETA,
) -> np.ndarray:
    """计算多证据融合的综合风险分数 s_i。

    s_i = α * a_i + β * b_i + γ * c̃_i

    其中 a_i = 模型分歧，b_i = 置信缺失，c̃_i = 局部不稳定（含边界保护）。

    Parameters
    ----------
    table : pd.DataFrame
        位置表，需包含 ``ensemble_std``、``ensemble_range``、
        ``mean_state_confidence``、``ensemble_median`` 列。
    alpha : float
        模型分歧权重。
    beta : float
        置信缺失权重。
    gamma : float
        局部不稳定权重。
    window : int, optional
        局部窗口大小，默认 21。
    eta : float, optional
        边界保护削弱系数，默认 0.3。

    Returns
    -------
    np.ndarray
        综合风险分数（融合三个证据分量）。
    """
    a = compute_disagreement_percentiles(table)
    b = compute_confidence_deficit_percentile(table)
    c = compute_local_instability_percentile(table, window=window, eta=eta)
    return alpha * a + beta * b + gamma * c


def fit_fusion_calibrator(
    train: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    window: int = LOCAL_WINDOW,
    eta: float = BOUNDARY_ETA,
    n_bins: int = 15,
) -> dict:
    """训练多证据融合 + 单调分位校准器。

    先用融合系数计算综合风险分数 s_i，再用保序回归将 s_i 映射为
    绝对误差的 90% 分位估计。

    Parameters
    ----------
    train : pd.DataFrame
        训练位置表。
    alpha, beta, gamma : float
        融合系数（α + β + γ = 1）。
    window, eta : optional
        局部不稳定参数。
    n_bins : int, optional
        单调校准的分箱数，默认 15。

    Returns
    -------
    dict
        包含校准器参数的字典。
    """
    s = compute_fusion_score(train, alpha, beta, gamma, window=window, eta=eta)
    error = np.abs(
        train["ensemble_median"].to_numpy(float)
        - train["true_value"].to_numpy(float)
    )

    # 分箱统计
    bin_edges = np.percentile(s, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_indices = np.digitize(s, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, len(bin_edges) - 1) - 1

    n_actual = len(bin_edges) - 1
    s_centers = np.empty(n_actual)
    q_bin = np.empty(n_actual)

    for b in range(n_actual):
        mask = bin_indices == b
        if mask.sum() > 0:
            s_centers[b] = np.median(s[mask])
            q_bin[b] = np.percentile(error[mask], 90)
        else:
            s_centers[b] = (bin_edges[b] + bin_edges[b + 1]) / 2
            q_bin[b] = 0.0

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_centers, q_bin)

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "window": window,
        "eta": eta,
        "s_bin_edges": bin_edges,
        "s_bin_centers": s_centers,
        "q_bin": q_bin,
        "isotonic": iso,
    }


def predict_fusion_uncertainty(
    calibrator: dict,
    table: pd.DataFrame,
) -> np.ndarray:
    """用训练好的融合校准器预测不确定性。

    Parameters
    ----------
    calibrator : dict
        ``fit_fusion_calibrator`` 返回的校准器。
    table : pd.DataFrame
        待预测数据。

    Returns
    -------
    np.ndarray
        非负不确定性估计值。
    """
    s = compute_fusion_score(
        table,
        calibrator["alpha"],
        calibrator["beta"],
        calibrator["gamma"],
        window=calibrator.get("window", LOCAL_WINDOW),
        eta=calibrator.get("eta", BOUNDARY_ETA),
    )
    iso = calibrator["isotonic"]
    return np.maximum(0.0, iso.predict(s))


def generate_fusion_candidates() -> list[tuple[float, float, float]]:
    """生成融合系数搜索候选列表。

    搜索空间 {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} 中 α+β+γ=1 的所有组合。

    Returns
    -------
    list[tuple[float, float, float]]
        (α, β, γ) 候选列表。
    """
    candidates = []
    for a in FUSION_GRID:
        for b in FUSION_GRID:
            c = round(1.0 - a - b, 1)
            if c in FUSION_GRID and c >= 0:
                candidates.append((a, b, c))
    return candidates
