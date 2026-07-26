from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd


def dataset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dirs() -> Dict[str, Path]:
    root = dataset_root()
    dirs = {
        "root": root,
        "csv": root / "csv_by_stress",
        "processed": root / "processed",
        "windows": root / "processed" / "model_windows",
        "reports": root / "reports",
        "figures": root / "figures",
        "scripts": root / "scripts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    cache = root / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    matplotlib.use("Agg")
    return dirs


DAMAGE_CLASS_TO_LEVEL = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80}
STRESS_CLASS_TO_MPA = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40}
STATE_TO_LABEL = {
    damage_class * 5 + stress_class: f"D{DAMAGE_CLASS_TO_LEVEL[damage_class]:02d}_S{STRESS_CLASS_TO_MPA[stress_class]:02d}"
    for damage_class in range(5)
    for stress_class in range(5)
}

CSV_PATTERN = "VTEST_S*.csv"
SEQ_WINDOW = 100
V2_MIN_WINDOW = 100
V1V3_MIN_WINDOW = 200
WINDOW_SIZES = [25, 50, 100, 200]
BASE_FEATURES = [
    "torque_mean",
    "torque_std",
    "torque_min",
    "torque_max",
    "torque_ptp",
    "torque_q25",
    "torque_q75",
    "torque_slope",
    "thrust_mean",
    "thrust_std",
    "thrust_min",
    "thrust_max",
    "thrust_ptp",
    "thrust_q25",
    "thrust_q75",
    "thrust_slope",
    "torque_thrust_corr",
    "torque_thrust_ratio",
    "depth_start",
    "depth_end",
]
MULTISCALE_FEATURES = [f"w{size}_{name}" for size in WINDOW_SIZES for name in BASE_FEATURES]
REQUIRED_COLUMNS = [
    "experiment_id",
    "sample_index",
    "cumulative_depth_cm",
    "depth_cm",
    "torque_nm",
    "thrust_kn",
    "true_damage_level",
    "true_damage_class",
    "true_stress_mpa",
    "true_stress_class",
    "true_state_class",
    "segment_index",
    "true_state_label",
]


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_sort_key(path: Path) -> Tuple[int, int, str]:
    match = re.search(r"VTEST_S(\d{2})(?:_(\d+))?\.csv$", path.name, flags=re.IGNORECASE)
    if not match:
        return 10_000, 10_000, path.name
    return int(match.group(1)), int(match.group(2) or 0), path.name


def validate_csv(path: Path) -> Dict[str, object]:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
    missing = [name for name in REQUIRED_COLUMNS if name not in head.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["true_stress_mpa", "true_damage_level", "segment_index"])
    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "rows": int(len(df)),
        "stress_mpa": sorted(pd.unique(df["true_stress_mpa"].dropna()).astype(int).tolist()),
        "damage_levels": sorted(pd.unique(df["true_damage_level"].dropna()).astype(int).tolist(), reverse=True),
        "segment_indices": sorted(pd.unique(df["segment_index"].dropna()).astype(int).tolist()),
    }


def discover_csv_files() -> Tuple[List[Path], Dict[str, object]]:
    dirs = ensure_dirs()
    csv_files = sorted(dirs["csv"].glob(CSV_PATTERN), key=csv_sort_key)
    if not csv_files:
        raise FileNotFoundError(f"No csv files found by pattern {dirs['csv'] / CSV_PATTERN}")
    summary = {
        "csv_pattern": CSV_PATTERN,
        "num_csv_files": len(csv_files),
        "files": [validate_csv(path) for path in csv_files],
    }
    save_json(dirs["reports"] / "csv_inventory_summary.json", summary)
    return csv_files, summary


def multiscale_features_matrix(values: np.ndarray) -> np.ndarray:
    # Build all rolling physical features once per borehole, then align rows by the window end index.
    frame = pd.DataFrame(values, columns=["depth", "torque", "thrust"])
    blocks = []
    for size in WINDOW_SIZES:
        roll = frame.rolling(window=size, min_periods=size)
        torque_mean = roll["torque"].mean()
        thrust_mean = roll["thrust"].mean()
        torque_min = roll["torque"].min()
        torque_max = roll["torque"].max()
        thrust_min = roll["thrust"].min()
        thrust_max = roll["thrust"].max()

        sum_depth = roll["depth"].sum()
        sum_depth2 = (frame["depth"] * frame["depth"]).rolling(window=size, min_periods=size).sum()
        denom = size * sum_depth2 - sum_depth * sum_depth

        sum_torque = roll["torque"].sum()
        sum_depth_torque = (frame["depth"] * frame["torque"]).rolling(window=size, min_periods=size).sum()
        torque_slope = (size * sum_depth_torque - sum_depth * sum_torque) / denom

        sum_thrust = roll["thrust"].sum()
        sum_depth_thrust = (frame["depth"] * frame["thrust"]).rolling(window=size, min_periods=size).sum()
        thrust_slope = (size * sum_depth_thrust - sum_depth * sum_thrust) / denom

        corr = roll["torque"].corr(frame["thrust"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ratio = (torque_mean / thrust_mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        blocks.append(
            pd.DataFrame(
                {
                    f"w{size}_torque_mean": torque_mean,
                    f"w{size}_torque_std": roll["torque"].std(ddof=0),
                    f"w{size}_torque_min": torque_min,
                    f"w{size}_torque_max": torque_max,
                    f"w{size}_torque_ptp": torque_max - torque_min,
                    f"w{size}_torque_q25": roll["torque"].quantile(0.25),
                    f"w{size}_torque_q75": roll["torque"].quantile(0.75),
                    f"w{size}_torque_slope": torque_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                    f"w{size}_thrust_mean": thrust_mean,
                    f"w{size}_thrust_std": roll["thrust"].std(ddof=0),
                    f"w{size}_thrust_min": thrust_min,
                    f"w{size}_thrust_max": thrust_max,
                    f"w{size}_thrust_ptp": thrust_max - thrust_min,
                    f"w{size}_thrust_q25": roll["thrust"].quantile(0.25),
                    f"w{size}_thrust_q75": roll["thrust"].quantile(0.75),
                    f"w{size}_thrust_slope": thrust_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0),
                    f"w{size}_torque_thrust_corr": corr,
                    f"w{size}_torque_thrust_ratio": ratio,
                    f"w{size}_depth_start": frame["depth"].shift(size - 1),
                    f"w{size}_depth_end": frame["depth"],
                }
            )
        )
    features = pd.concat(blocks, axis=1)
    return features.loc[V1V3_MIN_WINDOW - 1 :, MULTISCALE_FEATURES].to_numpy(dtype=np.float32)


def build_inputs_for_sequence(seq: pd.DataFrame, min_window: int, need_phys: bool) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = seq[["depth_cm", "torque_nm", "thrust_kn"]].to_numpy(dtype=np.float32)
    phys_matrix = multiscale_features_matrix(values) if need_phys else None
    seq_rows = []
    phys_rows = []
    meta_rows = []
    for end in range(min_window - 1, len(seq)):
        current = seq.iloc[end]
        seq_rows.append(values[end - SEQ_WINDOW + 1 : end + 1])
        if need_phys:
            phys_rows.append(phys_matrix[end - (V1V3_MIN_WINDOW - 1)])
        meta_rows.append(
            {
                "source_file": str(current.get("source_file", "")),
                "experiment_id": str(current["experiment_id"]),
                "sample_index": int(current["sample_index"]),
                "cumulative_depth_cm": float(current["cumulative_depth_cm"]),
                "depth_cm": float(current["depth_cm"]),
                "true_damage_level": int(current["true_damage_level"]),
                "true_damage_class": int(current["true_damage_class"]),
                "true_stress_mpa": int(current["true_stress_mpa"]),
                "true_stress_class": int(current["true_stress_class"]),
                "true_state_class": int(current["true_state_class"]),
                "true_state_label": str(current["true_state_label"]),
                "segment_index": int(current["segment_index"]),
            }
        )
    seq_arr = np.stack(seq_rows).astype(np.float32)
    phys_arr = np.stack(phys_rows).astype(np.float32) if need_phys else np.empty((len(meta_rows), 0), dtype=np.float32)
    return seq_arr, phys_arr, pd.DataFrame(meta_rows)


def window_paths(csv_path: Path) -> Dict[str, Path]:
    dirs = ensure_dirs()
    stem = csv_path.stem
    return {
        "v2_seq100": dirs["windows"] / f"{stem}_v2_seq100.npz",
        "v2_meta100": dirs["windows"] / f"{stem}_v2_meta100.csv",
        "v1v3_seq100_aligned200": dirs["windows"] / f"{stem}_v1v3_seq100_aligned200.npz",
        "v1v3_phys80": dirs["windows"] / f"{stem}_v1v3_phys80.npz",
        "v1v3_meta200": dirs["windows"] / f"{stem}_v1v3_meta200.csv",
    }


def save_window_inputs(csv_path: Path, force: bool = False) -> Dict[str, object]:
    paths = window_paths(csv_path)
    if not force and all(path.exists() for path in paths.values()):
        meta100 = pd.read_csv(paths["v2_meta100"], encoding="utf-8-sig")
        meta200 = pd.read_csv(paths["v1v3_meta200"], encoding="utf-8-sig")
        status = "reused"
    else:
        seq = pd.read_csv(csv_path, encoding="utf-8-sig")
        seq["source_file"] = csv_path.name
        seq100, _, meta100 = build_inputs_for_sequence(seq, min_window=V2_MIN_WINDOW, need_phys=False)
        seq200, phys200, meta200 = build_inputs_for_sequence(seq, min_window=V1V3_MIN_WINDOW, need_phys=True)
        np.savez_compressed(paths["v2_seq100"], seq=seq100)
        meta100.to_csv(paths["v2_meta100"], index=False, encoding="utf-8-sig")
        np.savez_compressed(paths["v1v3_seq100_aligned200"], seq=seq200)
        np.savez_compressed(paths["v1v3_phys80"], phys=phys200, feature_names=np.asarray(MULTISCALE_FEATURES))
        meta200.to_csv(paths["v1v3_meta200"], index=False, encoding="utf-8-sig")
        status = "written"

    return {
        "source_file": csv_path.name,
        "status": status,
        "v2_window_size": V2_MIN_WINDOW,
        "v2_stride": 1,
        "v2_windows": int(len(meta100)),
        "v1v3_min_window_size": V1V3_MIN_WINDOW,
        "v1v3_sequence_window_size": SEQ_WINDOW,
        "v1v3_stride": 1,
        "v1v3_windows": int(len(meta200)),
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }


def prepare_all_window_inputs(force: bool = False) -> Dict[str, object]:
    dirs = ensure_dirs()
    csv_files, csv_summary = discover_csv_files()
    rows = [save_window_inputs(path, force=force) for path in csv_files]
    summary = {
        "csv_summary": csv_summary,
        "window_dir": str(dirs["windows"].resolve()),
        "num_csv_files": len(csv_files),
        "window_rule": {
            "advancedV1": "80 multiscale physical features from windows 25/50/100/200, first valid end index 199, stride 1",
            "advancedV2": "100 point sequence window, first valid end index 99, stride 1",
            "advancedV3": "100 point sequence window aligned with 80 physical features, first valid end index 199, stride 1",
        },
        "files": rows,
    }
    pd.DataFrame(rows).drop(columns=["artifacts"], errors="ignore").to_csv(
        dirs["reports"] / "model_window_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_json(dirs["reports"] / "model_window_summary.json", summary)
    return summary


def load_v2_inputs(csv_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    paths = window_paths(csv_path)
    if not paths["v2_seq100"].exists() or not paths["v2_meta100"].exists():
        save_window_inputs(csv_path)
    seq = np.load(paths["v2_seq100"])["seq"].astype(np.float32)
    meta = pd.read_csv(paths["v2_meta100"], encoding="utf-8-sig")
    return seq, meta


def load_v1v3_inputs(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    paths = window_paths(csv_path)
    needed = [paths["v1v3_seq100_aligned200"], paths["v1v3_phys80"], paths["v1v3_meta200"]]
    if not all(path.exists() for path in needed):
        save_window_inputs(csv_path)
    seq = np.load(paths["v1v3_seq100_aligned200"])["seq"].astype(np.float32)
    phys = np.load(paths["v1v3_phys80"])["phys"].astype(np.float32)
    meta = pd.read_csv(paths["v1v3_meta200"], encoding="utf-8-sig")
    return seq, phys, meta


def add_training_script_paths() -> None:
    root = repo_root()
    for rel_path in [
        "processed_dataset_1o2a/combine_test/advancedV2/scripts",
        "processed_dataset_1o2a/combine_test/advancedV3/scripts",
    ]:
        path = str(root / rel_path)
        if path not in sys.path:
            sys.path.insert(0, path)


def load_v1_bundle():
    path = repo_root() / "processed_dataset_1o2a" / "combine_test" / "advancedV1" / "models" / "advanced_multiscale_model_bundle.pkl"
    return joblib.load(path)
