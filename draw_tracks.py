#!/usr/bin/env python3
"""Draw and save custom tracks."""

from __future__ import annotations

import argparse

from src.config.settings import load_config
from src.graphics.track.editor import run_obstacle_editor, run_track_editor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Pygame track editor")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument("--name", default="track", help="base name for saved track files")
    parser.add_argument(
        "--edit-obstacles",
        metavar="TRACK_JSON",
        default=None,
        help="edit obstacles in an existing track JSON; save with Ctrl+S",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.edit_obstacles:
        run_obstacle_editor(config, args.edit_obstacles)
    else:
        run_track_editor(config, name=args.name)


if __name__ == "__main__":
    main()
