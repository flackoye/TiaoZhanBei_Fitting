import numpy as np
import pandas as pd
import pytest
from uncertainty_analysis.data import build_position_table, validate_input


def sample_rows():
    return pd.DataFrame({
        "source_file": ["a.csv"] * 3,
        "experiment_id": ["A"] * 3,
        "sample_index": [1] * 3,
        "cumulative_depth_cm": [1.0] * 3,
        "segment_index": [0] * 3,
        "model": ["m1", "m2", "m3"],
        "pred_damage_level": [20.0, 40.0, 60.0],
        "pred_stress_mpa": [8.0, 10.0, 12.0],
        "true_damage_level": [40.0] * 3,
        "true_stress_mpa": [10.0] * 3,
        "state_confidence": [0.8, np.nan, 0.6],
    })


def test_position_table_aligns_three_models_without_losing_truth():
    table = build_position_table(sample_rows(), "damage")
    assert len(table) == 1
    assert table.loc[0, "ensemble_count"] == 3
    assert table.loc[0, "true_value"] == 40.0
    assert table.loc[0, "ensemble_std"] == pytest.approx(20.0)


def test_validation_rejects_missing_primary_key():
    with pytest.raises(ValueError, match="sample_index"):
        validate_input(sample_rows().drop(columns="sample_index"))
