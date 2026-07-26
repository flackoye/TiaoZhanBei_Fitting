from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

_matplotlib_cache = Path(__file__).resolve().parents[1] / ".matplotlib_cache"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from vtest3_utils import (
    DAMAGE_CLASS_TO_LEVEL,
    MULTISCALE_FEATURES,
    STATE_TO_LABEL,
    STRESS_CLASS_TO_MPA,
    add_training_script_paths,
    discover_csv_files,
    ensure_dirs,
    load_v1_bundle,
    load_v1v3_inputs,
    load_v2_inputs,
    prepare_all_window_inputs,
    repo_root,
    save_json,
)


def scale_array(values: np.ndarray, mean, std) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    std_arr[std_arr < 1e-8] = 1.0
    return ((values - mean_arr) / std_arr).astype(np.float32)


def load_v2_model(device: torch.device):
    add_training_script_paths()
    from sequence_utils import CNNBiLSTMStateClassifier

    path = repo_root() / "processed_dataset_1o2a" / "combine_test" / "advancedV2" / "models" / "cnn_bilstm_state_classifier.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CNNBiLSTMStateClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_v3_model(device: torch.device):
    add_training_script_paths()
    from fusion_utils import PhysicsFusionCNNBiLSTM

    path = repo_root() / "processed_dataset_1o2a" / "combine_test" / "advancedV3" / "models" / "physics_fusion_cnn_bilstm.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PhysicsFusionCNNBiLSTM().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def finalize_predictions(result: pd.DataFrame) -> pd.DataFrame:
    result["pred_damage_level"] = result["pred_damage_class"].map(DAMAGE_CLASS_TO_LEVEL)
    result["pred_stress_mpa"] = result["pred_stress_class"].map(STRESS_CLASS_TO_MPA)
    result["pred_state_label"] = result["pred_state_class"].map(STATE_TO_LABEL)
    result["state_head_damage_class"] = result["pred_state_class"] // 5
    result["state_head_stress_class"] = result["pred_state_class"] % 5
    result["state_head_damage_level"] = result["state_head_damage_class"].map(DAMAGE_CLASS_TO_LEVEL)
    result["state_head_stress_mpa"] = result["state_head_stress_class"].map(STRESS_CLASS_TO_MPA)
    result["damage_correct"] = result["pred_damage_class"] == result["true_damage_class"]
    result["stress_correct"] = result["pred_stress_class"] == result["true_stress_class"]
    result["multitask_state_correct"] = result["damage_correct"] & result["stress_correct"]
    result["state_correct"] = result["pred_state_class"] == result["true_state_class"]
    if "state_confidence" not in result.columns:
        result["state_confidence"] = np.nan
    return result


def predict_v1(phys: np.ndarray, meta: pd.DataFrame, bundle) -> pd.DataFrame:
    # V1 is a statistical-feature model: physical features -> damage model + stress model.
    features = pd.DataFrame(phys, columns=MULTISCALE_FEATURES)
    damage_model = bundle["models"][bundle["selected_damage_model"]]
    stress_model = bundle["models"][bundle["selected_stress_model"]]
    result = meta.copy()
    result["model"] = "advancedV1_multiscale_extratrees"
    result["pred_damage_class"] = damage_model.predict(features[MULTISCALE_FEATURES]).astype(int)
    result["pred_stress_class"] = stress_model.predict(features[MULTISCALE_FEATURES]).astype(int)
    result["pred_state_class"] = result["pred_damage_class"] * 5 + result["pred_stress_class"]
    return finalize_predictions(result)


def predict_v2(seq_windows: np.ndarray, meta: pd.DataFrame, model, checkpoint, device: torch.device) -> pd.DataFrame:
    # V2 is a sequence model: 100 point depth/torque/thrust windows -> three classification heads.
    seq_scaled = scale_array(seq_windows, checkpoint["scaler"]["mean"], checkpoint["scaler"]["std"])
    preds: Dict[str, List[np.ndarray]] = {"damage": [], "stress": [], "state": []}
    state_probs = []
    with torch.no_grad():
        for start in range(0, len(seq_scaled), 512):
            batch = torch.from_numpy(seq_scaled[start : start + 512]).to(device)
            out = model(batch)
            for key in preds:
                preds[key].append(torch.argmax(out[key], dim=1).cpu().numpy())
            state_probs.append(torch.softmax(out["state"], dim=1).cpu().numpy())

    result = meta.copy()
    result["model"] = "advancedV2_cnn_bilstm"
    result["pred_damage_class"] = np.concatenate(preds["damage"]).astype(int)
    result["pred_stress_class"] = np.concatenate(preds["stress"]).astype(int)
    result["pred_state_class"] = np.concatenate(preds["state"]).astype(int)
    result["state_confidence"] = np.vstack(state_probs).max(axis=1)
    return finalize_predictions(result)


def predict_v3(seq_windows: np.ndarray, phys: np.ndarray, meta: pd.DataFrame, model, checkpoint, device: torch.device) -> pd.DataFrame:
    # V3 fuses both branches: sequence features + multiscale physical features.
    seq_scaled = scale_array(seq_windows, checkpoint["seq_scaler"]["mean"], checkpoint["seq_scaler"]["std"])
    phys_scaled = scale_array(phys, checkpoint["phys_scaler"]["mean"], checkpoint["phys_scaler"]["std"])
    preds: Dict[str, List[np.ndarray]] = {"damage": [], "stress": [], "state": []}
    state_probs = []
    with torch.no_grad():
        for start in range(0, len(seq_scaled), 512):
            seq_batch = torch.from_numpy(seq_scaled[start : start + 512]).to(device)
            phys_batch = torch.from_numpy(phys_scaled[start : start + 512]).to(device)
            out = model(seq_batch, phys_batch)
            for key in preds:
                preds[key].append(torch.argmax(out[key], dim=1).cpu().numpy())
            state_probs.append(torch.softmax(out["state"], dim=1).cpu().numpy())

    result = meta.copy()
    result["model"] = "advancedV3_physics_fusion"
    result["pred_damage_class"] = np.concatenate(preds["damage"]).astype(int)
    result["pred_stress_class"] = np.concatenate(preds["stress"]).astype(int)
    result["pred_state_class"] = np.concatenate(preds["state"]).astype(int)
    result["state_confidence"] = np.vstack(state_probs).max(axis=1)
    return finalize_predictions(result)


def metrics_for_group(group: pd.DataFrame) -> Dict[str, float]:
    return {
        "num_windows": int(len(group)),
        "damage_accuracy": float(group["damage_correct"].mean()),
        "stress_accuracy": float(group["stress_correct"].mean()),
        "multitask_state_accuracy": float(group["multitask_state_correct"].mean()),
        "state_head_accuracy": float(group["state_correct"].mean()),
        "state_head_macro_f1": float(f1_score(group["true_state_class"], group["pred_state_class"], average="macro", zero_division=0)),
        "mean_state_confidence": float(group["state_confidence"].mean()) if group["state_confidence"].notna().any() else np.nan,
    }


def build_group_metrics(predictions: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(metrics_for_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_state_classification_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        labels = sorted(set(group["true_state_class"].astype(int)) | set(group["pred_state_class"].astype(int)))
        precision, recall, f1, support = precision_recall_fscore_support(
            group["true_state_class"],
            group["pred_state_class"],
            labels=labels,
            zero_division=0,
        )
        for label, p_value, r_value, f1_value, count in zip(labels, precision, recall, f1, support):
            rows.append(
                {
                    "model": model_name,
                    "state_class": int(label),
                    "state_label": STATE_TO_LABEL.get(int(label), str(label)),
                    "precision": float(p_value),
                    "recall": float(r_value),
                    "f1": float(f1_value),
                    "support": int(count),
                }
            )
    return pd.DataFrame(rows)


def plot_cloud_for_file(group: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    source_name = str(group["source_file"].iloc[0]) or str(group["experiment_id"].iloc[0])
    stem = Path(source_name).stem
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), dpi=170, sharex=True)
    axes[0].scatter(group["cumulative_depth_cm"], group["true_damage_level"], s=9, c="#1f77b4", alpha=0.65, label="True damage")
    axes[0].scatter(group["cumulative_depth_cm"], group["pred_damage_level"], s=7, c="#d62728", alpha=0.45, label="Pred damage")
    axes[0].set_ylabel("Damage (%)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(group["cumulative_depth_cm"], group["true_stress_mpa"], s=9, c="#1f77b4", alpha=0.65, label="True stress")
    axes[1].scatter(group["cumulative_depth_cm"], group["pred_stress_mpa"], s=7, c="#d62728", alpha=0.45, label="Pred stress")
    axes[1].set_ylabel("Stress (MPa)")
    axes[1].set_xlabel("Cumulative depth (cm)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    boundaries = group.groupby("segment_index", sort=True)["cumulative_depth_cm"].min().tolist()[1:]
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary, color="#444444", linestyle="--", linewidth=0.9, alpha=0.55)
    fig.suptitle(f"{model_name} {stem}")
    fig.tight_layout()
    fig.savefig(model_dir / f"{stem}_damage_stress_cloud.png")
    plt.close(fig)


def save_all_outputs(predictions: pd.DataFrame, window_summary: Dict[str, object], device: torch.device) -> None:
    dirs = ensure_dirs()
    predictions.to_csv(dirs["reports"] / "all_model_predictions.csv", index=False, encoding="utf-8-sig")

    overall = build_group_metrics(predictions, ["model"])
    by_file = build_group_metrics(predictions, ["model", "source_file"])
    by_stress = build_group_metrics(predictions, ["model", "true_stress_mpa"])
    by_segment = build_group_metrics(predictions, ["model", "source_file", "segment_index"])
    state_report = build_state_classification_report(predictions)

    overall.to_csv(dirs["reports"] / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    by_file.to_csv(dirs["reports"] / "by_file_metrics.csv", index=False, encoding="utf-8-sig")
    by_stress.to_csv(dirs["reports"] / "by_stress_metrics.csv", index=False, encoding="utf-8-sig")
    by_segment.to_csv(dirs["reports"] / "by_segment_metrics.csv", index=False, encoding="utf-8-sig")
    state_report.to_csv(dirs["reports"] / "state_classification_report.csv", index=False, encoding="utf-8-sig")

    for (model_name, source_file), group in predictions.groupby(["model", "source_file"], sort=False):
        plot_cloud_for_file(group, str(model_name), dirs["figures"])

    save_json(
        dirs["reports"] / "model_inversion_summary.json",
        {
            "device": str(device),
            "window_summary": window_summary,
            "num_prediction_rows": int(len(predictions)),
            "models": predictions["model"].drop_duplicates().tolist(),
            "overall_metrics": overall.to_dict(orient="records"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load three trained models, run inversion, and save metrics/plots.")
    parser.add_argument("--force-windows", action="store_true", help="Rebuild cached model inputs before testing.")
    args = parser.parse_args()

    dirs = ensure_dirs()
    csv_files, _ = discover_csv_files()
    window_summary = prepare_all_window_inputs(force=args.force_windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("loading models...")
    v1_bundle = load_v1_bundle()
    v2_model, v2_checkpoint = load_v2_model(device)
    v3_model, v3_checkpoint = load_v3_model(device)

    predictions = []
    for csv_path in csv_files:
        print("testing:", csv_path.name, flush=True)
        seq100, meta100 = load_v2_inputs(csv_path)
        seq200, phys200, meta200 = load_v1v3_inputs(csv_path)
        predictions.append(predict_v1(phys200, meta200, v1_bundle))
        predictions.append(predict_v2(seq100, meta100, v2_model, v2_checkpoint, device))
        predictions.append(predict_v3(seq200, phys200, meta200, v3_model, v3_checkpoint, device))

    all_predictions = pd.concat(predictions, ignore_index=True)
    save_all_outputs(all_predictions, window_summary, device)
    overall = pd.read_csv(dirs["reports"] / "overall_metrics.csv", encoding="utf-8-sig")
    print("completed")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
