#!/usr/bin/env python3
"""Train the Kohonen SOM from saved tracks."""

from __future__ import annotations

import argparse

from src.config.settings import load_config
from src.kohonen.training import train_som


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the lidar-input SOM")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument("--seed", type=int, default=None, help="random seed override")
    parser.add_argument("--headless", action="store_true", help="disable Pygame debug window")
    parser.add_argument(
        "--5lidar",
        dest="five_lidar",
        action="store_true",
        help="train a SOM from five lidar rays instead of three",
    )
    parser.add_argument(
        "--qtable",
        nargs="?",
        const="",
        default=None,
        help="collect SOM samples by running a Q-table policy; optional path, defaults to latest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = train_som(
        load_config(args.config),
        seed=args.seed,
        headless=args.headless,
        q_table_path=args.qtable,
        lidar_num_rays=5 if args.five_lidar else 3,
    )
    print(f"saved SOM model: {path}")


if __name__ == "__main__":
    main()
