#!/usr/bin/env python3
"""Train the Q-learning policy."""

from __future__ import annotations

import argparse

from src.config.settings import load_config
from src.rl.training import train_q_learning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on saved tracks")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument("--som", default=None, help="path to SOM .npz model; defaults to latest")
    parser.add_argument("--episodes", type=int, default=None, help="episode count override")
    parser.add_argument("--seed", type=int, default=None, help="random seed override")
    parser.add_argument("--headless", action="store_true", help="disable Pygame debug window")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = train_q_learning(
        load_config(args.config),
        som_path=args.som,
        episodes=args.episodes,
        seed=args.seed,
        headless=args.headless,
    )
    print(f"saved Q-table: {path}")


if __name__ == "__main__":
    main()
