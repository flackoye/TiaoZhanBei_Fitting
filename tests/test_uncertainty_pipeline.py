"""Tests for uncertainty_analysis.pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import patch

from uncertainty_analysis.pipeline import (
    cross_validate_methods,
    select_method,
    uncertainty_to_weight,
)


# ---------------------------------------------------------------------------
# Tests for select_method
# ---------------------------------------------------------------------------

def test_select_method_uses_mean_aurc_then_spearman_tiebreak():
    """Mean AURC is primary sort (ascending), mean Spearman secondary (desc)."""
    metrics = pd.DataFrame({
        "method": ["a", "a", "b", "b"],
        "aurc":    [1.0, 1.2, 0.8, 0.9],
        "spearman": [.5, .4, .3, .3],
    })
    assert select_method(metrics) == "b"


def test_select_method_tiebreak_spearman():
    """When AURC is equal, higher mean Spearman wins."""
    metrics = pd.DataFrame({
        "method": ["x", "x", "y", "y"],
        "aurc":    [0.5, 0.5, 0.5, 0.5],
        "spearman": [0.9, 0.8, 0.7, 0.6],
    })
    # x has mean spearman 0.85, y has 0.65 → x wins
    assert select_method(metrics) == "x"


def test_select_method_name_tiebreak():
    """When both AURC and Spearman tie, earlier name (sorted) wins."""
    metrics = pd.DataFrame({
        "method": ["b", "b", "a", "a"],
        "aurc":    [0.3, 0.3, 0.3, 0.3],
        "spearman": [0.5, 0.5, 0.5, 0.5],
    })
    # All equal; sorted methods: a, b → a wins
    assert select_method(metrics) == "a"


# ---------------------------------------------------------------------------
# Tests for uncertainty_to_weight
# ---------------------------------------------------------------------------

def test_higher_uncertainty_gets_lower_weight():
    """Scores far from training distribution get lower confidence/weight."""
    confidence, weight = uncertainty_to_weight(
        np.array([0., 1., 2., 3.]), np.array([0., 3.]), minimum=.1)
    assert confidence[0] > confidence[1]
    assert weight[0] > weight[1]
    assert np.all((weight >= .1) & (weight <= 1.0))


def test_minimum_weight_clamps_low_end():
    """All weights should be >= minimum even for extreme test scores."""
    _, weight = uncertainty_to_weight(
        np.array([1., 2., 3.]), np.array([999.]), minimum=0.3)
    assert weight[0] == pytest.approx(0.3)


def test_perfectly_known_score_gets_weight_1():
    """A test score that falls at or below all train scores gets max weight."""
    confidence, weight = uncertainty_to_weight(
        np.array([10., 20., 30.]), np.array([5.]), minimum=0.0)
    assert confidence[0] == pytest.approx(1.0)
    assert weight[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for cross_validate_methods
# ---------------------------------------------------------------------------

def test_cross_validate_returns_expected_columns():
    """Result DataFrame must have the documented schema."""
    table = pd.DataFrame({
        "source_file":        ["a.csv"] * 6 + ["b.csv"] * 6,
        "experiment_id":      ["X"] * 12,
        "sample_index":       list(range(6)) * 2,
        "depth_cm":           [float(i) for i in range(6)] * 2,
        "segment_index":      [0] * 12,
        "true_value":         [50.0] * 12,
        "ensemble_mean":      [50.0] * 12,
        "ensemble_median":    [50.0] * 12,
        "ensemble_std":       [1.0] * 6 + [2.0] * 6,
        "ensemble_range":     [3.0] * 12,
        "ensemble_count":     [3] * 12,
        "mean_state_confidence": [0.8] * 12,
    })
    result = cross_validate_methods(table, random_state=42)
    assert list(result.columns) == ["source_file", "method", "aurc", "spearman", "n_test_samples"]
    assert set(result["method"]) == {"disagreement", "calibrated"}
    assert len(result) == 2 * 2  # 2 source_files × 2 methods
    assert result["n_test_samples"].iloc[0] == 6


def test_cross_validate_skipped_when_insufficient_train():
    """If train fold has no rows, skip that fold (no crash)."""
    table = pd.DataFrame({
        "source_file":        ["only.csv"] * 3,
        "experiment_id":      ["X"] * 3,
        "sample_index":       [0, 1, 2],
        "depth_cm":           [0., 1., 2.],
        "segment_index":      [0] * 3,
        "true_value":         [50.0] * 3,
        "ensemble_mean":      [50.0] * 3,
        "ensemble_median":    [50.0] * 3,
        "ensemble_std":       [1.0] * 3,
        "ensemble_range":     [2.0] * 3,
        "ensemble_count":     [3] * 3,
        "mean_state_confidence": [0.8] * 3,
    })
    result = cross_validate_methods(table, random_state=42)
    assert len(result) == 0  # only one source_file → no train for that fold


# ---------------------------------------------------------------------------
# Leak-prevention test
# ---------------------------------------------------------------------------

def test_cross_validate_prevents_data_leak():
    """Each fold's calibrator must never see the test source_file in training."""
    table = pd.DataFrame({
        "source_file":        ["a.csv"] * 4 + ["b.csv"] * 4 + ["c.csv"] * 4,
        "experiment_id":      ["X"] * 12,
        "sample_index":       list(range(4)) * 3,
        "depth_cm":           [float(i) for i in range(4)] * 3,
        "segment_index":      [0] * 12,
        "true_value":         [50.0] * 12,
        "ensemble_mean":      [50.0] * 12,
        "ensemble_median":    [50.0] * 12,
        "ensemble_std":       [1.0, 2.0, 3.0, 4.0] * 3,
        "ensemble_range":     [2.0, 4.0, 6.0, 8.0] * 3,
        "ensemble_count":     [3] * 12,
        "mean_state_confidence": [0.8] * 12,
    })

    train_source_files_per_fold = {}

    original_calibrator = None
    from uncertainty_analysis.methods import fit_residual_calibrator

    def tracking_calibrator(train, random_state=42):
        # Record which source_files appear in train for the current fold
        fold_sources = tuple(sorted(train["source_file"].unique()))
        # We don't know the test file yet, but record train sources
        train_source_files_per_fold.setdefault("unknown", []).append(fold_sources)
        return fit_residual_calibrator(train, random_state=random_state)

    with patch("uncertainty_analysis.pipeline.fit_residual_calibrator", side_effect=tracking_calibrator):
        result = cross_validate_methods(table, random_state=42)

    # Build a mapping from test file to training files from the result
    # cross_validate_methods processes source_files in sorted order
    unique_files = sorted(table["source_file"].unique())
    fold_index = 0
    for test_file in unique_files:
        train_rows = table[table["source_file"] != test_file]
        if len(train_rows) < 3:
            continue
        # The calibrator should have been called with this fold's train data
        # We need to match: the fold for test_file should NOT contain test_file
        # in the training data passed to calibrator
        # Let's check the recorded training sources
        if fold_index < len(train_source_files_per_fold.get("unknown", [])):
            train_sources = train_source_files_per_fold["unknown"][fold_index]
            assert test_file not in train_sources, (
                f"Data leak: test source_file '{test_file}' found in "
                f"training set for its own fold: {train_sources}"
            )
            fold_index += 1


# ---------------------------------------------------------------------------
# End-to-end CLI test
# ---------------------------------------------------------------------------

def test_cli_writes_contract(tmp_path):
    """Run the CLI end-to-end and verify all four output files exist with the
    expected schema."""
    repo_root = Path(__file__).resolve().parent.parent
    cli_path = repo_root / "run_uncertainty.py"

    # --- 1. Create a small input CSV ---
    rows = []
    rng = np.random.RandomState(42)
    for sf in ["site_a", "site_b", "site_c"]:
        for model_id in ["m1", "m2", "m3"]:
            for sample_idx in range(4):
                rows.append({
                    "model": model_id,
                    "source_file": sf,
                    "experiment_id": "exp1",
                    "sample_index": sample_idx,
                    "pred_damage_level": 50.0
                    + (int(model_id[1]) - 2) * 0.5
                    + sample_idx * 0.1,
                    "pred_stress_mpa": 10.0
                    + (int(model_id[1]) - 2) * 0.3
                    + sample_idx * 0.05,
                    "true_damage_level": 50.0,
                    "true_stress_mpa": 10.0,
                    "cumulative_depth_cm": float(sample_idx),
                    "segment_index": 0,
                    "state_confidence": 0.8,
                })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test_input.csv"
    df.to_csv(csv_path, index=False)

    # --- 2. Create config YAML ---
    config = {
        "input_file": str(csv_path),
        "output_dir": str(tmp_path / "output"),
        "position_columns": ["source_file", "experiment_id", "sample_index"],
        "depth_preference": ["cumulative_depth_cm", "depth_cm"],
        "random_state": 42,
        "minimum_weight": 0.1,
        "selection_metric": "aurc",
        "mc_dropout": {"enabled": False, "status": "unavailable_no_model_artifacts"},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    # --- 3. Run the CLI ---
    result = subprocess.run(
        [sys.executable, str(cli_path), "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, (
        f"CLI failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # --- 4. Assert four output files exist ---
    output_dir = tmp_path / "output"
    assert (output_dir / "point_uncertainty.csv").exists()
    assert (output_dir / "method_evaluation.csv").exists()
    assert (output_dir / "selected_methods.json").exists()
    assert (output_dir / "risk_coverage_curves.csv").exists()

    # --- 5. Check point_uncertainty.csv has all expected columns ---
    point_df = pd.read_csv(output_dir / "point_uncertainty.csv")
    EXPECTED = {
        "model",
        "source_file",
        "experiment_id",
        "sample_index",
        "cumulative_depth_cm",
        "damage_uncertainty",
        "stress_uncertainty",
        "damage_confidence",
        "stress_confidence",
        "damage_weight",
        "stress_weight",
        "damage_uncertainty_method",
        "stress_uncertainty_method",
    }
    missing = EXPECTED - set(point_df.columns)
    assert not missing, f"Missing columns in point_uncertainty.csv: {missing}"

    # --- 6. Spot-check types ---
    assert point_df["damage_uncertainty"].dtype.kind == "f"
    assert point_df["damage_confidence"].dtype.kind == "f"
    assert point_df["damage_weight"].dtype.kind == "f"
    assert point_df["damage_uncertainty_method"].dtype.kind in ("O", "U")

    # --- 7. Validate selected_methods.json ---
    with open(output_dir / "selected_methods.json") as f:
        sm = json.load(f)
    assert "damage" in sm
    assert "stress" in sm
    assert sm["damage"] in ("disagreement", "calibrated")
    assert sm["stress"] in ("disagreement", "calibrated")

    # --- 8. Validate risk_coverage_curves.csv ---
    rc_df = pd.read_csv(output_dir / "risk_coverage_curves.csv")
    assert {"coverage", "risk_mae", "target"}.issubset(set(rc_df.columns))
    assert set(rc_df["target"]) == {"damage", "stress"}

    # --- 9. Validate method_evaluation.csv ---
    me_df = pd.read_csv(output_dir / "method_evaluation.csv")
    assert {"source_file", "method", "aurc", "spearman", "n_test_samples", "target"}.issubset(
        set(me_df.columns)
    )
