"""Training pipeline for the Kohonen SOM."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from minisom import MiniSom

from src.config.settings import AppConfig
from src.environment.car import Car, CarState
from src.environment.lidar import Lidar, LidarReading
from src.environment.track import Track, load_tracks
from src.graphics.renderer import Renderer
from src.kohonen.som import SomDiscretizer, discretizer_from_config


@dataclass
class SomSample:
    track: Track
    state: CarState
    lidar: LidarReading
    features: np.ndarray
    normalized: np.ndarray


def compute_sample_count(grid_dim: int, factor: float, num_tracks: int) -> int:
    base = (grid_dim**4 / 25.0) * factor
    return int(math.ceil(base / max(num_tracks, 1)) * max(num_tracks, 1))


def generate_som_samples(
    config: AppConfig,
    tracks: list[Track],
    rng: np.random.Generator,
) -> list[SomSample]:
    total = compute_sample_count(
        config.som.dim_grid_neurons,
        config.som.factor_samples,
        len(tracks),
    )
    lidar = Lidar(config.lidar.max_distance, config.lidar.side_angle_deg)
    placeholder = discretizer_from_config(
        config,
        np.zeros((config.som.dim_grid_neurons, config.som.dim_grid_neurons, 2), dtype=float),
    )
    samples: list[SomSample] = []
    for index in range(total):
        track = tracks[index % len(tracks)]
        x, y, heading = track.sample_drivable_point(
            rng,
            obstacle_clearance=max(config.car.length, config.car.width) / 2.0,
        )
        state = CarState(x, y, heading, 0.0)
        reading = lidar.scan(state, track)
        features = np.asarray(reading.vector, dtype=float) # [distance, difference_r_l]
        samples.append(
            SomSample(
                track=track,
                state=state,
                lidar=reading,
                features=features,
                normalized=placeholder.normalize_features(features),
            )
        )
    return samples


def train_som(
    config: AppConfig,
    seed: int | None = None,
    headless: bool = False,
) -> Path:
    config.ensure_output_dirs()
    track_pairs = load_tracks(config.tracks_dir)
    if not track_pairs:
        raise FileNotFoundError("no tracks found in out/tracks; run draw_tracks.py first")
    tracks = [track for _, track in track_pairs]
    rng = np.random.default_rng(seed if seed is not None else config.som.random_seed)
    samples = generate_som_samples(config, tracks, rng)
    data = np.asarray([sample.normalized for sample in samples], dtype=float)

    grid_dim = config.som.dim_grid_neurons
    som = MiniSom(
        grid_dim,
        grid_dim,
        2,
        sigma=grid_dim / 2.0,
        learning_rate=config.som.learning_rate,
        decay_function="inverse_decay_to_zero",
        neighborhood_function="gaussian",
        activation_distance="cosine",
        random_seed=seed if seed is not None else config.som.random_seed,
        sigma_decay_function="inverse_decay_to_one",
    )
    try:
        som.pca_weights_init(data)
    except Exception:
        som.random_weights_init(data)

    view = None if headless else SomTrainingView(config, grid_dim)
    total = len(data)
    order = np.arange(total)
    rng.shuffle(order)
    visible = True
    for iteration, sample_index in enumerate(order):
        vector = data[sample_index]
        som.update(vector, som.winner(vector), iteration, total)
        if view is not None:
            visible = view.process_events(visible)
            if visible or iteration % 10 == 0:
                sigma, learning_rate = _current_som_rates(som, iteration, total)
                weights = som.get_weights()
                view.draw(
                    samples[sample_index],
                    iteration + 1,
                    total,
                    sigma,
                    learning_rate,
                    SomDiscretizer(
                        weights=weights,
                        grid_dim=grid_dim,
                        norm_scale=1.0,
                        min_d_c=0.0,
                        max_d_c=1.0,
                        min_difference_r_l=-1.0,
                        max_difference_r_l=1.0,
                    ).state_from_features(vector),
                    data,
                    weights,
                    visible,
                )
    if view is not None:
        view.close()

    discretizer = discretizer_from_config(config, np.asarray(som.get_weights(), dtype=float))
    return discretizer.save(
        config.kohonen_dir,
        metadata={
            "num_samples": len(samples),
            "seed": seed if seed is not None else config.som.random_seed,
            "tracks": [path.name for path, _ in track_pairs],
        },
    )


class SomTrainingView:
    """Optional debug view toggled with the G key while training."""

    def __init__(self, config: AppConfig, grid_dim: int) -> None:
        self.renderer = Renderer(config, "SOM Training")
        self.grid_dim = grid_dim
        self.car = Car(
            length=config.car.length,
            width=config.car.width,
            max_speed=config.car.max_speed,
            min_speed=config.car.min_speed,
            state=CarState(0.0, 0.0, 0.0, 0.0),
        )

    def process_events(self, visible: bool) -> bool:
        pygame = self.renderer.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                return not visible
        return visible

    def draw(
        self,
        sample: SomSample,
        count: int,
        total: int,
        sigma: float,
        learning_rate: float,
        active_state: int,
        data: np.ndarray,
        weights: np.ndarray,
        visible: bool,
    ) -> None:
        if not visible:
            self.renderer.tick()
            return
        self.car.reset(sample.state)
        self.renderer.draw(
            sample.track,
            car=self.car,
            lidar=sample.lidar,
            som_state=active_state,
            som_dim=self.grid_dim,
            som_feature_points=data,
            som_weights=weights,
            lines=[
                f"samples {count}/{total}",
                f"sigma {sigma:.4f}",
                f"learning rate {learning_rate:.4f}",
                "G hides this view",
            ],
        )
        self.renderer.tick()

    def close(self) -> None:
        self.renderer.quit()


def _current_som_rates(som: MiniSom, iteration: int, total: int) -> tuple[float, float]:
    sigma = som._sigma_decay_function(som._sigma, iteration, total)
    learning_rate = som._learning_rate_decay_function(som._learning_rate, iteration, total)
    return float(sigma), float(learning_rate)
