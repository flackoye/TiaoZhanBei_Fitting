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
    ensemble_disagreement,
    fit_residual_calibrator,
    predict_calibrated_uncertainty,
)
from uncertainty_analysis.pipeline import (
    cross_validate_methods,
    select_method,
    uncertainty_to_weight,
)

LOG = logging.getLogger("uncertainty")

KEY_COLS = ["source_file", "experiment_id", "sample_index"]


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
        Name of the selected uncertainty method (``"disagreement"`` or
        ``"calibrated"``).
    risk_coverage_df : pd.DataFrame
        Risk-coverage curve for the selected method with a ``target`` column.
    eval_df : pd.DataFrame
        Cross-validation metrics with a ``target`` column.
    """
    rng = cfg.get("random_state", 42)
    minimum = cfg.get("minimum_weight", 0.1)

    LOG.info("[%s] 构建位置表 ...", target)
    table = build_position_table(df, target)
    LOG.info("[%s] 位置表完成: %d 行, %d 个 source_files",
             target, len(table), table["source_file"].nunique())

    # (b) Leave-one-source_file-out cross-validation
    LOG.info("[%s] 留一文件交叉验证 (%d folds) ...", target, table["source_file"].nunique())
    t0 = time.perf_counter()
    metrics = cross_validate_methods(table, random_state=rng)
    LOG.info("[%s] 交叉验证完成: %d 个评估项, 用时 %.1f秒",
             target, len(metrics), time.perf_counter() - t0)

    # (c) Select the best method
    if metrics.empty:
        best_method = "disagreement"
        LOG.info("[%s] CV空结果，回退到 disagreement 方法", target)
    else:
        best_method = select_method(metrics)
        LOG.info("[%s] 选中方法: %s (AURC: %.4f, Spearman: %.4f)",
                 target, best_method,
                 metrics[metrics["method"] == best_method]["aurc"].mean(),
                 metrics[metrics["method"] == best_method]["spearman"].mean())

    # (d) Fit final residual calibrator on ALL data
    calibrator = fit_residual_calibrator(table, random_state=rng)

    # (e) Get calibrated uncertainty on ALL data
    calibrated = predict_calibrated_uncertainty(calibrator, table)

    # (f) Get disagreement uncertainty on ALL data
    disagreement = ensemble_disagreement(table)

    # (g) Pick the selected method's uncertainty
    if best_method == "calibrated":
        selected = calibrated
    else:
        selected = disagreement

    # (h) Convert to confidence / weight
    confidence, weight = uncertainty_to_weight(selected, selected, minimum=minimum)
    LOG.info("[%s] 权重范围: [%.3f, %.3f]",
             target, weight.min(), weight.max())

    # (i) Augment the position table
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
