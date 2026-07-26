from __future__ import annotations

import argparse

from vtest3_utils import prepare_all_window_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sliding-window inputs for the three trained inversion models.")
    parser.add_argument("--force", action="store_true", help="Rebuild all cached window files.")
    args = parser.parse_args()

    summary = prepare_all_window_inputs(force=args.force)
    print("csv files:", summary["num_csv_files"])
    print("window dir:", summary["window_dir"])
    print("rules:")
    for model_name, rule in summary["window_rule"].items():
        print(f"  {model_name}: {rule}")


if __name__ == "__main__":
    main()
