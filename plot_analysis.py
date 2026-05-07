#!/usr/bin/env python3
"""Generate analysis plots from metrics CSV and Q-table checkpoints."""

from __future__ import annotations

import argparse

from src.analysis.plots import (
    discover_plot_model_folders,
    latest_metrics_csv,
    load_metrics_csv,
    save_all_plots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot model analysis outputs")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="metrics CSV path; defaults to newest out/analysis/metrics_*.csv",
    )
    parser.add_argument(
        "--model-folders",
        nargs="*",
        default=None,
        help="folders containing qtable*.npz checkpoints; defaults to folders under out/rl/bk",
    )
    parser.add_argument(
        "--base-label",
        default=None,
        help="base model_label in the metrics CSV when no row has is_base=yes",
    )
    parser.add_argument(
        "--output-dir",
        default="out/analysis",
        help="directory where PNG plots are saved",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = latest_metrics_csv() if args.metrics_csv is None else args.metrics_csv
    model_folders = args.model_folders or discover_plot_model_folders()
    rows = load_metrics_csv(metrics_path)
    paths = save_all_plots(rows, model_folders, args.output_dir, args.base_label)
    for path in paths:
        print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
