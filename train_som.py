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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = train_som(load_config(args.config), seed=args.seed, headless=args.headless)
    print(f"saved SOM model: {path}")


if __name__ == "__main__":
    main()
