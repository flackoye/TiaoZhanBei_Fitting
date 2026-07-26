"""数据读取、字段适配与清洗。"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
REQUIRED = ["model", "source_file", "experiment_id", "sample_index",
            "pred_damage_level", "pred_stress_mpa"]

def load_predictions(path: str | Path, config: dict) -> tuple[pd.DataFrame, str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path.resolve()}")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}; 实际字段: {list(df.columns)}")
    depth = next((c for c in config.get("depth_preference", []) if c in df.columns), None)
    if depth is None:
        raise ValueError("未找到 depth_cm 或 cumulative_depth_cm")
    if "state_confidence" not in df:
        df["state_confidence"] = np.nan
        LOG.warning("缺少 state_confidence，已按中性置信度处理")
    groups = [c for c in config.get("group_columns", []) if c in df.columns]
    if not groups:
        groups = [c for c in ["model", "source_file", "experiment_id"] if c in df]
    for c in [depth, "pred_damage_level", "pred_stress_mpa", "state_confidence"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # --- 加载不确定性权重（可选） ---
    unc_cfg = config.get("uncertainty", {})
    unc_file = unc_cfg.get("file") if isinstance(unc_cfg, dict) else None
    if unc_file is not None:
        unc_path = Path(unc_file)
        if not unc_path.exists():
            raise FileNotFoundError(f"找不到不确定性权重文件: {unc_path.resolve()}")
        unc_df = pd.read_csv(unc_path, low_memory=False)
        join_cols = unc_cfg.get("join_columns", ["model", "source_file", "experiment_id", "sample_index"])
        missing_join = [c for c in join_cols if c not in unc_df.columns]
        if missing_join:
            raise ValueError(f"不确定性文件缺少 join 字段: {missing_join}")
        weight_cols = ["damage_weight", "stress_weight"]
        missing_weight = [c for c in weight_cols if c not in unc_df.columns]
        if missing_weight:
            raise ValueError(f"不确定性文件缺少权重字段: {missing_weight}")
        for wc in weight_cols:
            unc_df[wc] = pd.to_numeric(unc_df[wc], errors="coerce")
            if unc_df[wc].isna().any():
                raise ValueError(f"权重列 {wc} 包含非数值或缺失值")
            out_of_range = ((unc_df[wc] < 0) | (unc_df[wc] > 1)).sum()
            if out_of_range:
                raise ValueError(f"权重列 {wc} 有 {out_of_range} 个值超出 [0, 1] 范围")
        before = len(df)
        df = df.merge(unc_df[join_cols + weight_cols], on=join_cols, how="left", validate="many_to_one")
        if len(df) != before:
            LOG.warning("不确定性权重连接后行数变化: %d -> %d", before, len(df))
        LOG.info("已加载不确定性权重: %s", unc_path.name)
    else:
        LOG.debug("未配置不确定性权重文件，使用 state_confidence")

    bad_depth = int(df[depth].isna().sum())
    if bad_depth:
        LOG.warning("删除 %d 行无效孔深", bad_depth)
        df = df[df[depth].notna()].copy()
    for c in ["pred_damage_level", "pred_stress_mpa"]:
        n = int(df[c].isna().sum())
        if n:
            LOG.warning("%s 有 %d 个缺失/无穷值，将在组内插值", c, n)
            df[c] = df.groupby(groups, dropna=False)[c].transform(lambda s: s.interpolate(limit_direction="both"))
    order_bad = 0
    for _, g in df.groupby(groups, sort=False, dropna=False):
        order_bad += int((np.diff(g[depth].to_numpy()) < 0).any())
    if order_bad:
        LOG.warning("发现 %d 个孔深顺序异常分组，已排序", order_bad)
    df = df.sort_values(groups + [depth, "sample_index"], kind="stable").reset_index(drop=True)
    df["_original_row"] = np.arange(len(df))
    return df, depth, groups

def collapse_duplicate_depths(group: pd.DataFrame, depth: str) -> tuple[pd.DataFrame, int]:
    """同深度聚合用于拟合；结果随后映射回全部原始行。"""
    ndup = int(group.duplicated(depth, keep=False).sum())
    if not ndup:
        return group.copy(), 0
    numeric = group.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "median" for c in numeric if c != depth}
    for c in group.columns:
        if c not in agg and c != depth:
            agg[c] = "first"
    return group.groupby(depth, as_index=False, sort=True, dropna=False).agg(agg), ndup
