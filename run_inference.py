#!/usr/bin/env python3
"""Run the trained policy in inference mode."""

from __future__ import annotations

import argparse

from src.config.settings import load_config
from src.rl.inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained SOM/Q-table inference")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument("--track", default=None, help="track index or JSON path; defaults to 0")
    parser.add_argument("--som", default=None, help="path to SOM .npz model; defaults to latest")
    parser.add_argument("--qtable", default=None, help="path to Q-table .npz; defaults to latest")
    parser.add_argument(
        "--5lidar",
        dest="five_lidar",
        action="store_true",
        help="require a five-lidar inference run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(
        load_config(args.config),
        track_selector=args.track,
        som_path=args.som,
        q_table_path=args.qtable,
        lidar_num_rays=5 if args.five_lidar else None,
    )


if __name__ == "__main__":
    main()
