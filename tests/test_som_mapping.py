"""Inspect the mapping from lidar distances to the SOM BMU weight.

Run from the project root, for example:

    python3 tests/test_som_mapping.py 50 80 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_config
from src.kohonen.som import SomDiscretizer, load_som_model


@dataclass(frozen=True)
class SomMapping:
    d_c: float
    d_r: float
    d_l: float
    feature_d_c: float
    feature_d_rl: float
    normalized_s1: float
    normalized_s2: float
    bmu_row: int
    bmu_col: int
    state: int
    s1: float
    s2: float


def map_distances_to_som_weight(
    som: SomDiscretizer,
    d_c: float,
    d_r: float,
    d_l: float,
) -> SomMapping:
    features = np.asarray([d_c, d_r - d_l], dtype=float)
    normalized = som.normalize_features(features)
    bmu_row, bmu_col = som.winner(normalized)
    weight = som.weights[bmu_row, bmu_col]

    return SomMapping(
        d_c=d_c,
        d_r=d_r,
        d_l=d_l,
        feature_d_c=float(features[0]),
        feature_d_rl=float(features[1]),
        normalized_s1=float(normalized[0]),
        normalized_s2=float(normalized[1]),
        bmu_row=bmu_row,
        bmu_col=bmu_col,
        state=bmu_row * som.grid_dim + bmu_col,
        s1=float(weight[0]),
        s2=float(weight[1]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map input distances d_c, d_r, d_l to the BMU SOM weight s1, s2.",
    )
    parser.add_argument("d_c", type=float, help="Center lidar distance.")
    parser.add_argument("d_r", type=float, help="Right lidar distance.")
    parser.add_argument("d_l", type=float, help="Left lidar distance.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.toml"),
        help="Path to config.toml.",
    )
    parser.add_argument(
        "--som",
        default=None,
        help="Path to a SOM .npz file. Defaults to the newest one in out/kohonen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    som_path = Path(args.som) if args.som else None
    som = load_som_model(som_path, config)
    mapping = map_distances_to_som_weight(som, args.d_c, args.d_r, args.d_l)

    print("Input distances:")
    print(f"  d_c = {mapping.d_c:.6g}")
    print(f"  d_r = {mapping.d_r:.6g}")
    print(f"  d_l = {mapping.d_l:.6g}")
    print("SOM input features:")
    print(f"  d_c       = {mapping.feature_d_c:.6g}")
    print(f"  d_r - d_l = {mapping.feature_d_rl:.6g}")
    print("Normalized input:")
    print(f"  s1_input = {mapping.normalized_s1:.6g}")
    print(f"  s2_input = {mapping.normalized_s2:.6g}")
    print("BMU:")
    print(f"  row   = {mapping.bmu_row}")
    print(f"  col   = {mapping.bmu_col}")
    print(f"  state = {mapping.state}")
    print("Output BMU weight:")
    print(f"  s1 = {mapping.s1:.6g}")
    print(f"  s2 = {mapping.s2:.6g}")


if __name__ == "__main__":
    main()
