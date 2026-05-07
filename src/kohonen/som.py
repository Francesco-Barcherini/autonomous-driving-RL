"""SOM model wrapper and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from src.config.settings import AppConfig
from src.environment.lidar import LidarReading, lidar_num_rays_from_input_len
from src.environment.track import latest_file


@dataclass
class SomDiscretizer:
    """Frozen 2D SOM used to map lidar features to a tabular state."""

    weights: np.ndarray
    grid_dim: int
    max_distance: float
    input_len: int = 0
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.input_len <= 0:
            self.input_len = int(self.weights.shape[-1])

    @property
    def num_states(self) -> int:
        return self.grid_dim * self.grid_dim

    @property
    def lidar_num_rays(self) -> int:
        return lidar_num_rays_from_input_len(self.input_len)

    def normalize_features(self, features: tuple[float, ...] | np.ndarray) -> np.ndarray:
        vector = self._coerce_feature_length(features)
        max_distance = max(float(self.max_distance), 1e-12)
        normalized = np.empty(self.input_len, dtype=float)
        normalized[0] = np.clip(vector[0], 0.0, max_distance) / max_distance
        if self.input_len > 1:
            normalized[1:] = np.clip(
                vector[1:],
                -max_distance,
                max_distance,
            ) / (2.0 * max_distance)
        return normalized

    def _coerce_feature_length(self, features: tuple[float, ...] | np.ndarray) -> np.ndarray:
        vector = np.asarray(features, dtype=float).reshape(-1)
        if len(vector) == self.input_len:
            return vector.copy()
        result = np.zeros(self.input_len, dtype=float)
        length = min(len(vector), self.input_len)
        if length:
            result[:length] = vector[:length]
        return result

    def features_from_lidar(self, reading: LidarReading) -> np.ndarray:
        return np.asarray(reading.vector, dtype=float)

    def winner(self, normalized_features: np.ndarray) -> tuple[int, int]:
        sample = np.asarray(normalized_features, dtype=float)
        flat = self.weights.reshape(-1, self.weights.shape[-1])
        distances = np.linalg.norm(flat - sample, axis=1)
        index = int(np.argmin(distances))
        return divmod(index, self.grid_dim)

    def state_from_features(self, features: tuple[float, ...] | np.ndarray) -> int:
        row, col = self.winner(self.normalize_features(features))
        return row * self.grid_dim + col

    def state_from_lidar(self, reading: LidarReading) -> int:
        return self.state_from_features(self.features_from_lidar(reading))

    def display_values_from_lidar(self, reading: LidarReading) -> tuple[float, ...]:
        """Return d_c in [0, 1] and difference features in [-1, 1] for display."""
        features = self._coerce_feature_length(self.features_from_lidar(reading))
        max_distance = max(float(self.max_distance), 1e-12)
        values = [float(np.clip(features[0] / max_distance, 0.0, 1.0))]
        values.extend(
            float(np.clip(value / max_distance, -1.0, 1.0))
            for value in features[1:]
        )
        return tuple(values)

    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"som_{timestamp}.npz"
        np.savez(
            path,
            weights=self.weights,
            grid_dim=np.asarray(self.grid_dim),
            max_distance=np.asarray(self.max_distance),
            input_len=np.asarray(self.input_len),
            lidar_num_rays=np.asarray(self.lidar_num_rays),
            metadata=np.asarray(json.dumps(metadata or {})),
        )
        self.source_path = path
        return path


def discretizer_from_config(config: AppConfig, weights: np.ndarray) -> SomDiscretizer:
    return SomDiscretizer(
        weights=weights,
        grid_dim=config.som.dim_grid_neurons,
        max_distance=config.lidar.max_distance,
        input_len=int(weights.shape[-1]),
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
        input_len=_load_input_len(data),
        source_path=model_path,
    )
    return model


def _load_max_distance(data: np.lib.npyio.NpzFile, config: AppConfig) -> float:
    if "max_distance" in data.files:
        return float(data["max_distance"])
    if "max_d_c" in data.files:
        return float(data["max_d_c"])
    return float(config.lidar.max_distance)


def _load_input_len(data: np.lib.npyio.NpzFile) -> int:
    if "input_len" in data.files:
        return int(data["input_len"])
    return int(np.asarray(data["weights"]).shape[-1])
