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
    norm_scale: float
    min_d_c: float
    max_d_c: float
    min_difference_r_l: float
    max_difference_r_l: float
    source_path: Path | None = None

    @property
    def num_states(self) -> int:
        return self.grid_dim * self.grid_dim

    def normalize_features(self, features: tuple[float, float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=float).copy()
        vector[0] = np.clip(vector[0], self.min_d_c, self.max_d_c)
        vector[1] = np.clip(
            vector[1],
            self.min_difference_r_l,
            self.max_difference_r_l,
        )
        return vector / self.norm_scale

    def features_from_lidar(self, reading: LidarReading) -> np.ndarray:
        return np.asarray(reading.vector, dtype=float)

    def winner(self, normalized_features: np.ndarray) -> tuple[int, int]:
        sample = np.asarray(normalized_features, dtype=float)
        flat = self.weights.reshape(-1, self.weights.shape[-1])
        sample_norm = np.linalg.norm(sample)
        weight_norms = np.linalg.norm(flat, axis=1)
        denom = np.maximum(weight_norms * max(sample_norm, 1e-12), 1e-12)
        distances = 1.0 - (flat @ sample) / denom
        index = int(np.argmin(distances))
        return divmod(index, self.grid_dim)

    def state_from_features(self, features: tuple[float, float] | np.ndarray) -> int:
        row, col = self.winner(self.normalize_features(features))
        return row * self.grid_dim + col

    def state_from_lidar(self, reading: LidarReading) -> int:
        return self.state_from_features(self.features_from_lidar(reading))

    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"som_{timestamp}.npz"
        np.savez(
            path,
            weights=self.weights,
            grid_dim=np.asarray(self.grid_dim),
            norm_scale=np.asarray(self.norm_scale),
            min_d_c=np.asarray(self.min_d_c),
            max_d_c=np.asarray(self.max_d_c),
            min_difference_r_l=np.asarray(self.min_difference_r_l),
            max_difference_r_l=np.asarray(self.max_difference_r_l),
            metadata=np.asarray(json.dumps(metadata or {})),
        )
        self.source_path = path
        return path


def discretizer_from_config(config: AppConfig, weights: np.ndarray) -> SomDiscretizer:
    max_abs_diff = max(abs(config.som.min_difference_r_l), abs(config.som.max_difference_r_l))
    norm_scale = float(np.linalg.norm([config.som.max_d_c, max_abs_diff]))
    return SomDiscretizer(
        weights=weights,
        grid_dim=config.som.dim_grid_neurons,
        norm_scale=max(norm_scale, 1e-12),
        min_d_c=config.som.min_d_c,
        max_d_c=config.som.max_d_c,
        min_difference_r_l=config.som.min_difference_r_l,
        max_difference_r_l=config.som.max_difference_r_l,
    )


def load_som_model(path: str | Path | None, config: AppConfig) -> SomDiscretizer:
    model_path = Path(path) if path else latest_file(config.kohonen_dir, "som_*.npz")
    if model_path is None:
        raise FileNotFoundError("no SOM model found in out/kohonen; run train_som.py first")
    data = np.load(model_path, allow_pickle=False)
    model = SomDiscretizer(
        weights=np.asarray(data["weights"], dtype=float),
        grid_dim=int(data["grid_dim"]),
        norm_scale=float(data["norm_scale"]),
        min_d_c=float(data["min_d_c"]),
        max_d_c=float(data["max_d_c"]),
        min_difference_r_l=float(data["min_difference_r_l"]),
        max_difference_r_l=float(data["max_difference_r_l"]),
        source_path=model_path,
    )
    return model
