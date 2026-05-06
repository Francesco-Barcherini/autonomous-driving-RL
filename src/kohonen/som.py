"""SOM model wrapper and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from src.config.settings import AppConfig
from src.environment.lidar import LidarReading
from src.environment.track import latest_file


@dataclass
class SomDiscretizer:
    """Frozen 2D SOM used to map lidar features to a tabular state."""

    weights: np.ndarray
    grid_dim: int
    max_distance: float
    source_path: Path | None = None

    @property
    def num_states(self) -> int:
        return self.grid_dim * self.grid_dim

    def normalize_features(self, features: tuple[float, float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=float).copy()
        max_distance = max(float(self.max_distance), 1e-12)
        return np.asarray(
            [
                np.clip(vector[0], 0.0, max_distance) / max_distance,
                np.clip(vector[1], -max_distance, max_distance) / (2.0 * max_distance),
            ],
            dtype=float,
        )

    def features_from_lidar(self, reading: LidarReading) -> np.ndarray:
        return np.asarray(reading.vector, dtype=float)

    def winner(self, normalized_features: np.ndarray) -> tuple[int, int]:
        sample = np.asarray(normalized_features, dtype=float)
        flat = self.weights.reshape(-1, self.weights.shape[-1])
        distances = np.linalg.norm(flat - sample, axis=1)
        index = int(np.argmin(distances))
        return divmod(index, self.grid_dim)

    def state_from_features(self, features: tuple[float, float] | np.ndarray) -> int:
        row, col = self.winner(self.normalize_features(features))
        return row * self.grid_dim + col

    def state_from_lidar(self, reading: LidarReading) -> int:
        return self.state_from_features(self.features_from_lidar(reading))

    def display_values_from_lidar(self, reading: LidarReading) -> tuple[float, float]:
        """Return d_c in [0, 1] and d_rl in [-1, 1] for UI display."""
        d_c, d_rl = self.features_from_lidar(reading)
        max_distance = max(float(self.max_distance), 1e-12)
        d_c_norm = d_c / max_distance
        d_rl_norm = d_rl / max_distance
        return (
            float(np.clip(d_c_norm, 0.0, 1.0)),
            float(np.clip(d_rl_norm, -1.0, 1.0)),
        )

    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"som_{timestamp}.npz"
        np.savez(
            path,
            weights=self.weights,
            grid_dim=np.asarray(self.grid_dim),
            max_distance=np.asarray(self.max_distance),
            metadata=np.asarray(json.dumps(metadata or {})),
        )
        self.source_path = path
        return path


def discretizer_from_config(config: AppConfig, weights: np.ndarray) -> SomDiscretizer:
    return SomDiscretizer(
        weights=weights,
        grid_dim=config.som.dim_grid_neurons,
        max_distance=config.lidar.max_distance,
    )


def load_som_model(path: str | Path | None, config: AppConfig) -> SomDiscretizer:
    model_path = Path(path) if path else latest_file(config.kohonen_dir, "som_*.npz")
    if model_path is None:
        raise FileNotFoundError("no SOM model found in out/kohonen; run train_som.py first")
    data = np.load(model_path, allow_pickle=False)
    max_distance = _load_max_distance(data, config)
    model = SomDiscretizer(
        weights=np.asarray(data["weights"], dtype=float),
        grid_dim=int(data["grid_dim"]),
        max_distance=max_distance,
        source_path=model_path,
    )
    return model


def _load_max_distance(data: np.lib.npyio.NpzFile, config: AppConfig) -> float:
    if "max_distance" in data.files:
        return float(data["max_distance"])
    if "max_d_c" in data.files:
        return float(data["max_d_c"])
    return float(config.lidar.max_distance)
