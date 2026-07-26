"""Tests for integrating calibrated uncertainty weights into the fitting pipeline.

This verifies that:
1. When uncertainty.file is null/None in config, old behavior (state_confidence) is preserved
2. When uncertainty.file is set, weights are loaded, validated, and joined correctly
3. Validation rejects missing columns / out-of-range weights
4. Continuous blending works correctly in fit_curve
5. The run_fitting process correctly uses damage_weight / stress_weight
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from data_loader import load_predictions
from fitting_methods import fit_curve


# ---------------------------------------------------------------------------
# Fixtures: small input CSVs
# ---------------------------------------------------------------------------

@pytest.fixture
def fitting_input_csv(tmp_path):
    """A minimal all_model_predictions.csv with one group."""
    rows = []
    for i in range(10):
        rows.append({
            "model": "m1",
            "source_file": "hole_a.csv",
            "experiment_id": "exp1",
            "sample_index": i,
            "cumulative_depth_cm": float(i),
            "pred_damage_level": 50.0 + i * 0.5,
            "pred_stress_mpa": 10.0 + i * 0.2,
            "state_confidence": 0.8 if i % 2 == 0 else 0.5,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "all_model_predictions.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def uncertainty_csv(tmp_path):
    """A minimal point_uncertainty.csv with damage_weight/stress_weight."""
    rows = []
    for i in range(10):
        rows.append({
            "model": "m1",
            "source_file": "hole_a.csv",
            "experiment_id": "exp1",
            "sample_index": i,
            "damage_weight": 1.0 - i * 0.05,   # descending from 1.0
            "stress_weight": 0.9 - i * 0.05,   # descending from 0.9
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "point_uncertainty.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def minimal_config(fitting_input_csv):
    """A config dict without uncertainty section (old-behavior test)."""
    return {
        "input_file": str(fitting_input_csv),
        "output_dir": str(fitting_input_csv.parent),
        "group_columns": ["model", "source_file", "experiment_id"],
        "depth_preference": ["cumulative_depth_cm", "depth_cm"],
        "outlier": {
            "method": "hampel",
            "window_size": 9,
            "mad_threshold": 3.5,
            "correction": "interpolate",
        },
        "fitting": {
            "method": "savgol",
            "savgol_window_length": 5,
            "savgol_polyorder": 2,
            "confidence_threshold": 0.7,
            "high_confidence_blend": 0.2,
            "low_confidence_blend": 0.8,
            "output_depth_step": 0.1,
        },
        "evaluation": {
            "high_confidence_threshold": 0.8,
            "peak_quantile": 0.95,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Old behavior when uncertainty.file is absent / null
# ---------------------------------------------------------------------------

def test_old_behavior_without_uncertainty_config(minimal_config):
    """When no uncertainty section at all, load_predictions works as before."""
    df, depth, groups = load_predictions(minimal_config["input_file"], minimal_config)
    assert depth == "cumulative_depth_cm"
    assert "state_confidence" in df.columns
    assert "damage_weight" not in df.columns
    assert "stress_weight" not in df.columns
    assert len(df) == 10


def test_old_behavior_with_null_uncertainty_file(minimal_config):
    """When uncertainty.file is null, old behavior is preserved."""
    minimal_config["uncertainty"] = {"file": None, "join_columns": ["model", "source_file", "experiment_id", "sample_index"]}
    df, depth, groups = load_predictions(minimal_config["input_file"], minimal_config)
    assert depth == "cumulative_depth_cm"
    assert "state_confidence" in df.columns
    assert "damage_weight" not in df.columns
    assert "stress_weight" not in df.columns


# ---------------------------------------------------------------------------
# Test 2: Uncertainty weights get loaded and joined
# ---------------------------------------------------------------------------

def test_uncertainty_weights_loaded_and_joined(minimal_config, uncertainty_csv):
    """When uncertainty.file is set, load_predictions left-joins weights."""
    minimal_config["uncertainty"] = {
        "file": str(uncertainty_csv),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    df, depth, groups = load_predictions(minimal_config["input_file"], minimal_config)
    assert "damage_weight" in df.columns
    assert "stress_weight" in df.columns
    # All 10 original rows preserved with correct weights
    assert len(df) == 10
    # sample_index 0 should have damage_weight = 1.0
    row0 = df[df["sample_index"] == 0].iloc[0]
    assert row0["damage_weight"] == pytest.approx(1.0)
    assert row0["stress_weight"] == pytest.approx(0.9)


def test_uncertainty_weights_preserve_state_confidence(minimal_config, uncertainty_csv):
    """Both state_confidence and weight columns coexist in the DataFrame."""
    minimal_config["uncertainty"] = {
        "file": str(uncertainty_csv),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    df, depth, groups = load_predictions(minimal_config["input_file"], minimal_config)
    assert "state_confidence" in df.columns
    assert "damage_weight" in df.columns
    assert "stress_weight" in df.columns
    # state_confidence values are preserved
    assert df["state_confidence"].notna().sum() == 10


# ---------------------------------------------------------------------------
# Test 3: Validation
# ---------------------------------------------------------------------------

def test_validation_rejects_missing_weight_columns(minimal_config, uncertainty_csv):
    """Uncertainty CSV must contain damage_weight and stress_weight."""
    # Drop stress_weight from the uncertainty file
    bad_df = pd.read_csv(uncertainty_csv).drop(columns=["stress_weight"])
    bad_path = uncertainty_csv.parent / "bad_uncertainty.csv"
    bad_df.to_csv(bad_path, index=False)

    minimal_config["uncertainty"] = {
        "file": str(bad_path),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    with pytest.raises(ValueError, match="damage_weight|stress_weight|weight"):
        load_predictions(minimal_config["input_file"], minimal_config)


def test_validation_rejects_missing_join_columns(minimal_config, uncertainty_csv):
    """Uncertainty CSV must contain all join_columns."""
    bad_df = pd.read_csv(uncertainty_csv).drop(columns=["sample_index"])
    bad_path = uncertainty_csv.parent / "bad_uncertainty2.csv"
    bad_df.to_csv(bad_path, index=False)

    minimal_config["uncertainty"] = {
        "file": str(bad_path),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    with pytest.raises(ValueError, match="sample_index|join"):
        load_predictions(minimal_config["input_file"], minimal_config)


def test_validation_warns_on_out_of_range_weights(minimal_config, uncertainty_csv):
    """Weights outside [0, 1] should raise."""
    bad_df = pd.read_csv(uncertainty_csv)
    bad_df.loc[0, "damage_weight"] = 1.5
    bad_path = uncertainty_csv.parent / "bad_uncertainty3.csv"
    bad_df.to_csv(bad_path, index=False)

    minimal_config["uncertainty"] = {
        "file": str(bad_path),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    with pytest.raises(ValueError, match="weight|range|0.*1|clamp"):
        load_predictions(minimal_config["input_file"], minimal_config)


# ---------------------------------------------------------------------------
# Test 4: Continuous blending in fit_curve
# ---------------------------------------------------------------------------

def test_continuous_blending_weight_one_gives_least_smoothing():
    """Weight=1.0 → blend = low - 1.0*(low-high) = high (0.2)."""
    x = np.arange(5, dtype=float)
    y = np.array([0.0, 5.0, 0.0, 5.0, 0.0])  # zigzag
    cfg = {
        "method": "savgol",
        "savgol_window_length": 5,
        "savgol_polyorder": 2,
        "confidence_threshold": 0.7,
        "high_confidence_blend": 0.2,
        "low_confidence_blend": 0.8,
    }
    # All weights = 1.0 (high confidence) → continuous blend = 0.2
    weights = np.full(5, 1.0)
    result = fit_curve(x, y, "savgol", cfg, confidence=weights, continuous_blend=True)
    # With blend=0.2 → result = 0.8*y + 0.2*fitted (close to original)
    # Should differ from binary threshold path
    bin_result = fit_curve(x, y, "savgol", cfg, confidence=weights, continuous_blend=False)
    assert np.allclose(result, bin_result, atol=1e-6)


def test_continuous_blending_weight_01_gives_most_smoothing():
    """Weight=0.1 → blend = low - 0.1*(low-high) = 0.74."""
    x = np.arange(5, dtype=float)
    y = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
    cfg = {
        "method": "savgol",
        "savgol_window_length": 5,
        "savgol_polyorder": 2,
        "confidence_threshold": 0.7,
        "high_confidence_blend": 0.2,
        "low_confidence_blend": 0.8,
    }
    # All weights = 0.1 → continuous blend = 0.74
    weights = np.full(5, 0.1)
    result = fit_curve(x, y, "savgol", cfg, confidence=weights, continuous_blend=True)
    # With blend=0.74 → result = 0.26*y + 0.74*fitted (lots of smoothing)
    # Under binary threshold: conf=0.1 < 0.7 → blend = 0.8
    bin_result = fit_curve(x, y, "savgol", cfg, confidence=weights, continuous_blend=False)
    # Continuous should be different from binary
    assert not np.allclose(result, bin_result, atol=1e-6)


def test_continuous_blending_monotonic():
    """Higher weight → result closer to original (less smoothing)."""
    x = np.arange(5, dtype=float)
    y = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
    cfg = {
        "method": "savgol",
        "savgol_window_length": 5,
        "savgol_polyorder": 2,
        "confidence_threshold": 0.7,
        "high_confidence_blend": 0.2,
        "low_confidence_blend": 0.8,
    }
    r_low = fit_curve(x, y, "savgol", cfg, confidence=np.full(5, 0.1), continuous_blend=True)
    r_high = fit_curve(x, y, "savgol", cfg, confidence=np.full(5, 0.9), continuous_blend=True)
    r_full = fit_curve(x, y, "savgol", cfg, confidence=np.full(5, 1.0), continuous_blend=True)
    # Higher weight → closer to original → smaller MAE from original
    mae_low = np.mean(np.abs(r_low - y))
    mae_high = np.mean(np.abs(r_high - y))
    mae_full = np.mean(np.abs(r_full - y))
    assert mae_low >= mae_high >= mae_full  # monotonic: more weight = less smoothing


# ---------------------------------------------------------------------------
# Test 5: End-to-end run_fitting process with uncertainty weights
# ---------------------------------------------------------------------------

def test_process_with_uncertainty_weights(minimal_config, uncertainty_csv):
    """Run_fitting process completes with uncertainty weights, using them for blending."""
    minimal_config["uncertainty"] = {
        "file": str(uncertainty_csv),
        "join_columns": ["model", "source_file", "experiment_id", "sample_index"],
    }
    # Import process here to avoid circular issues
    from run_fitting import process
    res, summary = process(minimal_config, limit_one=True, make_plots=False, generate_dense=False)
    # Basic sanity checks
    assert len(res) == 10
    assert "fitted_damage" in res.columns
    assert "fitted_stress" in res.columns
    assert "damage_weight" in res.columns
    assert "stress_weight" in res.columns
    # Summary should have entries for both damage and stress
    assert len(summary) == 2
    assert set(summary["target"]) == {"damage", "stress"}


def test_process_without_uncertainty_weights(minimal_config):
    """Without uncertainty weights, process uses state_confidence (old behavior)."""
    from run_fitting import process
    res, summary = process(minimal_config, limit_one=True, make_plots=False, generate_dense=False)
    assert len(res) == 10
    assert "fitted_damage" in res.columns
    assert "fitted_stress" in res.columns
    assert "damage_weight" not in res.columns
