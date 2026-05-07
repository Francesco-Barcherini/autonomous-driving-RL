#!/usr/bin/env python3
"""Evaluate trained Q-table models on saved tracks."""

from __future__ import annotations

import argparse

from src.analysis.evaluation import build_model_specs, evaluate_models, write_metrics_csv
from src.config.settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained Q-table models")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument(
        "--model-folders",
        nargs="*",
        default=None,
        help="folders containing qtable*.npz checkpoints; defaults to folders under out/rl/bk",
    )
    parser.add_argument(
        "--base-qtable",
        default=None,
        help="base qtable*.npz for time-ratio plots; defaults to latest hard 5-lidar model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output CSV path; defaults to out/analysis/metrics_<timestamp>.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    specs = build_model_specs(config, args.model_folders, args.base_qtable)
    rows = evaluate_models(config, specs)
    path = write_metrics_csv(rows, args.output, config)
    print(f"saved metrics: {path}")


if __name__ == "__main__":
    main()
