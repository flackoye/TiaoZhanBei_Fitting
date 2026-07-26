from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


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

INTEGER_COLUMNS = [
    "sample_index",
    "true_damage_level",
    "true_damage_class",
    "true_stress_mpa",
    "true_stress_class",
    "true_state_class",
    "segment_index",
]

FLOAT_COLUMNS = [
    "cumulative_depth_cm",
    "depth_cm",
    "torque_nm",
    "thrust_kn",
]


def dataset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input_path() -> Path:
    return dataset_root() / "csv_by_stress" / "VTEST_S99.xlsx"


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".csv")


def ensure_report_dir() -> Path:
    report_dir = dataset_root() / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def read_excel_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input Excel not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Excel is missing required columns: {missing}")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df[REQUIRED_COLUMNS].copy()
    for column in INTEGER_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    for column in FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    out["experiment_id"] = out["experiment_id"].astype(str)
    out["true_state_label"] = out["true_state_label"].astype(str)
    return out


def summarize(df: pd.DataFrame, input_path: Path, output_path: Path) -> Dict[str, object]:
    return {
        "source_excel": str(input_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "experiment_ids": sorted(df["experiment_id"].dropna().astype(str).unique().tolist()),
        "stress_mpa": sorted(df["true_stress_mpa"].dropna().astype(int).unique().tolist()),
        "damage_levels": sorted(df["true_damage_level"].dropna().astype(int).unique().tolist(), reverse=True),
        "segment_indices": sorted(df["segment_index"].dropna().astype(int).unique().tolist()),
        "depth_min_cm": float(df["cumulative_depth_cm"].min()),
        "depth_max_cm": float(df["cumulative_depth_cm"].max()),
    }


def convert_excel_to_csv(input_path: Path, output_path: Path, sheet_name: str | int = 0, overwrite: bool = False) -> Dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    df = read_excel_table(input_path, sheet_name=sheet_name)
    validate_columns(df)
    normalized = normalize_types(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = summarize(normalized, input_path, output_path)
    report_path = ensure_report_dir() / f"{output_path.stem}_excel_to_csv_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_json"] = str(report_path.resolve())
    return summary


def parse_sheet(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a standardized VTEST Excel file to csv_by_stress CSV format.")
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Source xlsx path.")
    parser.add_argument("--output", type=Path, default=None, help="Output csv path. Defaults to input path with .csv suffix.")
    parser.add_argument("--sheet", default="0", help="Sheet index or sheet name. Defaults to the first sheet.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it already exists.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    summary = convert_excel_to_csv(input_path, output_path, sheet_name=parse_sheet(args.sheet), overwrite=args.overwrite)
    print("converted:", summary["source_excel"])
    print("output:", summary["output_csv"])
    print("rows:", summary["rows"])
    print("stress_mpa:", summary["stress_mpa"])
    print("damage_levels:", summary["damage_levels"])
    print("summary:", summary["summary_json"])


if __name__ == "__main__":
    main()
