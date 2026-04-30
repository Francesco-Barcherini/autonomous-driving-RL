#!/usr/bin/env python3
"""Draw and save custom tracks."""

from __future__ import annotations

import argparse

from src.config.settings import load_config
from src.graphics.track.editor import run_track_editor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Pygame track editor")
    parser.add_argument("--config", default="config.toml", help="path to TOML configuration")
    parser.add_argument("--name", default="track", help="base name for saved track files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_track_editor(load_config(args.config), name=args.name)


if __name__ == "__main__":
    main()
