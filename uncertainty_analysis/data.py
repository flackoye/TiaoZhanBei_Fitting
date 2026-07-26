import numpy as np
import pandas as pd

KEY = ["source_file", "experiment_id", "sample_index"]
REQUIRED = KEY + ["model", "pred_damage_level", "pred_stress_mpa",
                  "true_damage_level", "true_stress_mpa"]


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")


def build_position_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    validate_input(df)
    pred = {"damage": "pred_damage_level", "stress": "pred_stress_mpa"}[target]
    truth = {"damage": "true_damage_level", "stress": "true_stress_mpa"}[target]
    depth = "cumulative_depth_cm" if "cumulative_depth_cm" in df else "depth_cm"
    grouped = df.groupby(KEY, sort=False, dropna=False)
    table = grouped.agg(
        depth_cm=(depth, "median"), segment_index=("segment_index", "first"),
        true_value=(truth, "median"), ensemble_mean=(pred, "mean"),
        ensemble_median=(pred, "median"), ensemble_std=(pred, "std"),
        ensemble_range=(pred, lambda values: values.max() - values.min()),
        ensemble_count=("model", "nunique"),
        mean_state_confidence=("state_confidence", "mean"),
    ).reset_index()
    table.loc[table["ensemble_count"] < 2, ["ensemble_std", "ensemble_range"]] = np.nan
    return table
