#!/usr/bin/env python3
"""CLI entry point for uncertainty analysis.

Reads a model predictions CSV, estimates per-point uncertainty for both
*depth* and *stress* targets via ensemble disagreement and/or a calibrated
residual regressor, converts the uncertainty scores into confidence/weight
values, and writes four contract output files.

Usage
-----
    python run_uncertainty.py --config uncertainty_config.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uncertainty_analysis.data import build_position_table
from uncertainty_analysis.evaluation import risk_coverage
from uncertainty_analysis.methods import (
    fit_residual_calibrator,
    predict_calibrated_uncertainty,
)
from uncertainty_analysis.method2_monotonic import (
    fit_monotonic_calibrator,
    predict_monotonic_uncertainty,
)
from uncertainty_analysis.method3_fusion import (
    compute_fusion_score,
    fit_fusion_calibrator,
    predict_fusion_uncertainty,
)
from uncertainty_analysis.pipeline import (
    cross_validate_methods,
    select_method,
    select_best_fusion_coefficients,
    uncertainty_to_weight,
)

LOG = logging.getLogger("uncertainty")

KEY_COLS = ["source_file", "experiment_id", "sample_index"]

# 融合方法默认参数（文档 2.5 节）
FUSION_WINDOW = 21
FUSION_ETA = 0.3


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uncertainty analysis CLI")
    parser.add_argument(
        "--config",
        default="uncertainty_config.yaml",
        help="Path to configuration YAML (default: uncertainty_config.yaml)",
    )
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_method_name(method_str: str) -> tuple[str, dict | None]:
    """解析方法名称，提取基础方法和参数。

    支持格式：
    - "calibrated", "monotonic"
    - "fusion_a0.4_b0.4_g0.2"

    Parameters
    ----------
    method_str : str
        select_method 返回的方法名称。

    Returns
    -------
    base_method : str
        基础方法名（"calibrated", "monotonic", "fusion"）。
    params : dict or None
        额外参数（融合系数等）。
    """
    if method_str.startswith("fusion_"):
        parts = method_str.split("_")
        alpha = float(parts[1].lstrip("a"))
        beta = float(parts[2].lstrip("b"))
        gamma = float(parts[3].lstrip("g"))
        return "fusion", {"alpha": alpha, "beta": beta, "gamma": gamma}
    return method_str, None


def _process_target(
    target: str,
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    """Run the full uncertainty pipeline for one target.

    Returns
    -------
    table : pd.DataFrame
        Position table augmented with ``{target}_uncertainty``,
        ``{target}_confidence``, ``{target}_weight``.
    method_name : str
        Name of the selected uncertainty method.
    risk_coverage_df : pd.DataFrame
        Risk-coverage curve for the selected method with a ``target`` column.
    eval_df : pd.DataFrame
        Cross-validation metrics with a ``target`` column.
    """
    rng = cfg.get("random_state", 42)
    minimum = cfg.get("minimum_weight", 0.1)
    methods = cfg.get("methods", ["calibrated", "monotonic", "fusion"])

    LOG.info("[%s] 构建位置表 ...", target)
    table = build_position_table(df, target)
    LOG.info("[%s] 位置表完成: %d 行, %d 个 source_files",
             target, len(table), table["source_file"].nunique())

    # (b) Leave-one-source_file-out cross-validation
    n_files = table["source_file"].nunique()
    LOG.info("[%s] 留一文件交叉验证 (%d folds) ...", target, n_files)
    t0 = time.perf_counter()
    metrics = cross_validate_methods(
        table,
        random_state=rng,
        methods=methods,
        fusion_window=FUSION_WINDOW,
        fusion_eta=FUSION_ETA,
    )
    LOG.info("[%s] 交叉验证完成: %d 个评估项, 用时 %.1f秒",
             target, len(metrics), time.perf_counter() - t0)

    # (c) Select the best method
    if metrics.empty:
        best_method = "calibrated"
        LOG.info("[%s] CV空结果，回退到 calibrated（方案一）", target)
    else:
        best_method = select_method(metrics)
        LOG.info("[%s] 选中方法: %s", target, best_method)

    # (d) 解析方法名称和参数
    base_method, fusion_params = _parse_method_name(best_method)

    # (e) 计算最终不确定性
    if base_method == "calibrated":
        calibrator = fit_residual_calibrator(table, random_state=rng)
        selected = predict_calibrated_uncertainty(calibrator, table)
        LOG.info("[%s] 使用 calibrated 方法", target)
    elif base_method == "monotonic":
        use_seg = cfg.get("monotonic_use_segment_correction", False)
        mono_cal = fit_monotonic_calibrator(table, use_segment_correction=use_seg)
        selected = predict_monotonic_uncertainty(mono_cal, table)
        LOG.info("[%s] 使用 monotonic 方法 (segment_correction=%s)",
                 target, use_seg)
    elif base_method == "fusion":
        if fusion_params:
            alpha = fusion_params["alpha"]
            beta = fusion_params["beta"]
            gamma = fusion_params["gamma"]
            LOG.info("[%s] 使用 fusion 方法 (α=%.1f, β=%.1f, γ=%.1f)",
                     target, alpha, beta, gamma)
        else:
            # 从 CV 结果中选择最佳融合系数
            alpha, beta, gamma = select_best_fusion_coefficients(metrics)
            LOG.info("[%s] 使用 fusion 方法 (CV选择: α=%.1f, β=%.1f, γ=%.1f)",
                     target, alpha, beta, gamma)

        # 拟合单调校准器
        fusion_cal = fit_fusion_calibrator(
            table, alpha, beta, gamma,
            window=FUSION_WINDOW, eta=FUSION_ETA,
        )
        selected = predict_fusion_uncertainty(fusion_cal, table)
    else:
        LOG.warning("[%s] 未知方法 %s，回退到 calibrated（方案一）", target, best_method)
        calibrator = fit_residual_calibrator(table, random_state=rng)
        selected = predict_calibrated_uncertainty(calibrator, table)

    # (f) Convert to confidence / weight
    confidence, weight = uncertainty_to_weight(selected, selected, minimum=minimum)
    LOG.info("[%s] 权重范围: [%.3f, %.3f]", target, weight.min(), weight.max())

    # (g) Augment the position table
    table[f"{target}_uncertainty"] = selected
    table[f"{target}_confidence"] = confidence
    table[f"{target}_weight"] = weight

    # Build risk-coverage curve for the selected method
    test_errors = np.abs(
        table["ensemble_median"].to_numpy(float) - table["true_value"].to_numpy(float)
    )
    rc = risk_coverage(test_errors, selected)
    rc["target"] = target

    # Add target column to cross-validation metrics
    metrics_df = metrics.copy()
    metrics_df["target"] = target

    return table, best_method, rc, metrics_df


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    cfg = load_config(args.config)

    input_file = cfg["input_file"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("加载数据: %s", input_file)
    t_start = time.perf_counter()
    df = pd.read_csv(input_file)
    LOG.info("加载完成: %d 行, %d 列 (%.1f秒)",
             len(df), len(df.columns), time.perf_counter() - t_start)

    # Process both targets
    damage_table, damage_method, damage_rc, damage_metrics = _process_target("damage", df, cfg)
    stress_table, stress_method, stress_rc, stress_metrics = _process_target("stress", df, cfg)

    method_names = {"damage": damage_method, "stress": stress_method}

    # Merge position-level results back to original rows
    damage_cols = KEY_COLS + [
        "damage_uncertainty",
        "damage_confidence",
        "damage_weight",
    ]
    stress_cols = KEY_COLS + [
        "stress_uncertainty",
        "stress_confidence",
        "stress_weight",
    ]

    merged = damage_table[damage_cols].merge(
        stress_table[stress_cols],
        on=KEY_COLS,
        how="outer",
    )

    # Join back to the original prediction rows (by KEY columns)
    point_df = df.merge(merged, on=KEY_COLS, how="left")

    # Add method-name columns
    point_df["damage_uncertainty_method"] = damage_method
    point_df["stress_uncertainty_method"] = stress_method

    # --- Write all four contract outputs ---

    # 1. point_uncertainty.csv
    LOG.info("写入 point_uncertainty.csv (%d 行) ...", len(point_df))
    point_df.to_csv(
        output_dir / "point_uncertainty.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. method_evaluation.csv
    eval_df = pd.concat([damage_metrics, stress_metrics], ignore_index=True)
    LOG.info("写入 method_evaluation.csv (%d 行) ...", len(eval_df))
    eval_df.to_csv(
        output_dir / "method_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3. selected_methods.json
    LOG.info("写入 selected_methods.json: %s", method_names)
    with open(output_dir / "selected_methods.json", "w", encoding="utf-8") as f:
        json.dump(method_names, f, indent=2)

    # 4. risk_coverage_curves.csv
    rc_df = pd.concat([damage_rc, stress_rc], ignore_index=True)
    LOG.info("写入 risk_coverage_curves.csv (%d 行) ...", len(rc_df))
    rc_df.to_csv(
        output_dir / "risk_coverage_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = time.perf_counter() - t_start
    LOG.info("全部完成! 用时 %.1f 秒", elapsed)


if __name__ == "__main__":
    main()
