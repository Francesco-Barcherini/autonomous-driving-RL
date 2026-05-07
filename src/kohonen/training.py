"""Training pipeline for the Kohonen SOM."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from minisom import MiniSom

from src.config.settings import AppConfig
from src.environment.car import Car, CarState
from src.environment.env import DrivingEnv
from src.environment.lidar import Lidar, LidarReading, input_len_from_lidar_num_rays
from src.environment.track import Track, load_tracks
from src.graphics.renderer import Renderer
from src.kohonen.som import SomDiscretizer, discretizer_from_config, load_som_model
from src.rl.actions import ActionSpace
from src.rl.q_learning import load_q_table


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
    lidar_num_rays: int = 3,
) -> list[SomSample]:
    total = compute_sample_count(
        config.som.dim_grid_neurons,
        config.som.factor_samples,
        len(tracks),
    )
    input_len = input_len_from_lidar_num_rays(lidar_num_rays)
    lidar = Lidar(config.lidar.max_distance, config.lidar.side_angle_deg, lidar_num_rays)
    placeholder = discretizer_from_config(
        config,
        np.zeros((config.som.dim_grid_neurons, config.som.dim_grid_neurons, input_len), dtype=float),
    )
    samples: list[SomSample] = []
    progress_interval = max(total // 20, 1)
    for index in range(total):
        track = tracks[index % len(tracks)]
        x, y, heading = track.sample_drivable_point(
            rng,
            obstacle_clearance=max(config.car.length, config.car.width) / 2.0,
        )
        state = CarState(x, y, heading, 0.0)
        reading = lidar.scan(state, track)
        features = np.asarray(reading.vector, dtype=float)
        samples.append(
            SomSample(
                track=track,
                state=state,
                lidar=reading,
                features=features,
                normalized=placeholder.normalize_features(features),
            )
        )
        collected = len(samples)
        if collected == 1 or collected % progress_interval == 0 or collected == total:
            print(f"collected SOM samples {collected}/{total}")
    return samples


def generate_policy_som_samples(
    config: AppConfig,
    tracks: list[Track],
    q_table_path: str | Path | None,
    rng: np.random.Generator,
    headless: bool = False,
    lidar_num_rays: int = 3,
) -> tuple[list[SomSample], Path]:
    input_len = input_len_from_lidar_num_rays(lidar_num_rays)
    placeholder = discretizer_from_config(
        config,
        np.zeros((config.som.dim_grid_neurons, config.som.dim_grid_neurons, input_len), dtype=float),
    )
    q_bundle = load_q_table(q_table_path, config)
    policy_som = load_som_model(q_bundle.som_path or None, config)
    if q_bundle.q_table.shape[0] != policy_som.num_states:
        raise ValueError(
            "Q-table state count does not match its SOM model: "
            f"{q_bundle.q_table.shape[0]} != {policy_som.num_states}"
        )
    actions = _actions_for_q_table(config, q_bundle.actions, q_bundle.q_table.shape[1])
    samples: list[SomSample] = []
    view = None if headless else SomCollectionView(config)
    visible = True
    episode = 0

    for track in tracks:
        env = DrivingEnv(config, track, rng, lidar_num_rays=lidar_num_rays)
        result = env.reset(random_start=False)
        episode += 1


        while not result.done:# and env.steps < config.rl.max_steps_per_episode:
            state = policy_som.state_from_lidar(result.lidar)
            action_index = int(np.argmax(q_bundle.q_table[state]))
            action = tuple(float(value) for value in actions[action_index])
            result = env.step(action)
            features = np.asarray(result.lidar.vector, dtype=float)
            if features[0] < 1.0 or rng.random() < 0.1:
                samples.append(
                    SomSample(
                        track=env.track,
                        state=env.car.state.copy(),
                        lidar=result.lidar,
                        features=features,
                        normalized=placeholder.normalize_features(features),
                    )
                )
            collected = len(samples)
            print(f"collected policy SOM samples {collected}, episode {episode}/{len(tracks)}", end="\r")
            if view is not None:
                visible = view.process_events(visible)
                view.draw(
                    env,
                    collected,
                    len(tracks),
                    episode,
                    state,
                    action_index,
                    visible,
                )

    if view is not None:
        view.close()
    return samples, q_bundle.source_path


def train_som(
    config: AppConfig,
    seed: int | None = None,
    headless: bool = False,
    q_table_path: str | Path | None = None,
    lidar_num_rays: int = 3,
) -> Path:
    config.ensure_output_dirs()
    track_pairs = load_tracks(config.tracks_dir)
    if not track_pairs:
        raise FileNotFoundError("no tracks found in out/tracks; run draw_tracks.py first")
    tracks = [track for _, track in track_pairs]
    rng = np.random.default_rng(seed if seed is not None else config.som.random_seed)
    if q_table_path is None:
        samples = generate_som_samples(config, tracks, rng, lidar_num_rays=lidar_num_rays)
        sample_source = "random"
        used_q_table_path = ""
    else:
        resolved_q_table_path = None if str(q_table_path) == "" else q_table_path
        samples, used_q_table_path = generate_policy_som_samples(
            config,
            tracks,
            resolved_q_table_path,
            rng,
            headless=headless,
            lidar_num_rays=lidar_num_rays,
        )
        sample_source = "policy"
    data = np.asarray([sample.normalized for sample in samples], dtype=float)
    if data.size == 0:
        raise RuntimeError("no SOM samples were collected")

    grid_dim = config.som.dim_grid_neurons
    input_len = data.shape[1]
    som = MiniSom(
        grid_dim,
        grid_dim,
        input_len,
        sigma=grid_dim / 2.0,
        learning_rate=config.som.learning_rate,
        decay_function="inverse_decay_to_zero",
        neighborhood_function="gaussian",
        activation_distance="euclidean",
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
                active_row, active_col = SomDiscretizer(
                    weights=weights,
                    grid_dim=grid_dim,
                    max_distance=1.0,
                    input_len=input_len,
                ).winner(vector)
                view.draw(
                    samples[sample_index],
                    iteration + 1,
                    total,
                    sigma,
                    learning_rate,
                    active_row * grid_dim + active_col,
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
            "sample_source": sample_source,
            "q_table_path": str(used_q_table_path),
            "lidar_num_rays": lidar_num_rays,
            "input_len": input_len,
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
                *_feature_display_lines(sample.normalized),
                "G hides this view",
            ],
        )
        self.renderer.tick()

    def close(self) -> None:
        self.renderer.quit()


class SomCollectionView:
    """Optional simulation view while collecting SOM samples from a policy."""

    def __init__(self, config: AppConfig) -> None:
        self.renderer = Renderer(config, "SOM Policy Sample Collection")

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
        env: DrivingEnv,
        count: int,
        total: int,
        episode: int,
        state: int,
        action_index: int,
        visible: bool,
    ) -> None:
        if not visible:
            return
        self.renderer.draw(
            env.track,
            car=env.car,
            lidar=env.last_lidar,
            lines=[
                f"collecting samples {count}/{total}",
                f"episode {episode}",
                f"step {env.steps}",
                f"policy state {state}",
                f"policy action {action_index}",
                *_lidar_display_lines(env.last_lidar),
                "G hides this view",
            ],
        )
        self.renderer.tick()

    def close(self) -> None:
        self.renderer.quit()


def _actions_for_q_table(
    config: AppConfig,
    saved_actions: np.ndarray,
    num_actions: int,
) -> np.ndarray:
    actions = np.asarray(saved_actions, dtype=float)
    if actions.shape[0] == num_actions:
        return actions
    configured_actions = ActionSpace.from_config(config).as_array()
    if configured_actions.shape[0] == num_actions:
        return configured_actions
    raise ValueError(
        f"Q-table has {num_actions} actions, but saved actions have "
        f"{actions.shape[0]} and configured actions have {configured_actions.shape[0]}"
    )


def _current_som_rates(som: MiniSom, iteration: int, total: int) -> tuple[float, float]:
    sigma = som._sigma_decay_function(som._sigma, iteration, total)
    learning_rate = som._learning_rate_decay_function(som._learning_rate, iteration, total)
    return float(sigma), float(learning_rate)


def _feature_display_lines(features: np.ndarray) -> list[str]:
    labels = ("d_c", "d_rl", "d_rrll")
    return [f"{label} {value:.3f}" for label, value in zip(labels, features)]


def _lidar_display_lines(reading: LidarReading | None) -> list[str]:
    if reading is None:
        return []
    return _feature_display_lines(np.asarray(reading.vector, dtype=float))
