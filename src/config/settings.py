"""Application configuration loading.

The project keeps user-tunable parameters in a TOML file at the repository
root. This module maps that file into dataclasses and resolves output paths
relative to the config file location.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass
class PathsConfig:
    tracks_dir: str = "out/tracks"
    kohonen_dir: str = "out/kohonen"
    rl_dir: str = "out/rl"


@dataclass
class SimulationConfig:
    dt: float = 0.1
    max_steps_per_episode: int = 1000
    random_seed: int = 42


@dataclass
class GraphicsConfig:
    screen_width: int = 1100
    screen_height: int = 760
    fps: int = 60
    background: tuple[int, int, int] = (18, 22, 28)
    road_color: tuple[int, int, int] = (68, 72, 78)
    road_outline_color: tuple[int, int, int] = (190, 198, 206)
    car_color: tuple[int, int, int] = (235, 196, 82)
    lidar_color: tuple[int, int, int] = (69, 182, 235)
    obstacle_color: tuple[int, int, int] = (220, 82, 82)


@dataclass
class TrackConfig:
    num_tracks: int = 5
    track_width: float = 80.0
    obstacle_radius: float = 13.0
    editor_min_point_distance: float = 8.0


@dataclass
class CarConfig:
    length: float = 34.0
    width: float = 18.0
    max_speed: float = 180.0
    min_speed: float = -45.0
    max_acceleration: float = 70.0
    max_steering_acceleration_deg: float = 120.0


@dataclass
class LidarConfig:
    max_distance: float = 220.0
    side_angle_deg: float = 45.0


@dataclass
class SomConfig:
    dim_grid_neurons: int = 6
    factor_samples: float = 2.0
    max_d_c: float = 220.0
    min_d_c: float = 0.0
    max_difference_r_l: float = 220.0
    min_difference_r_l: float = -220.0
    learning_rate: float = 1.0
    random_seed: int = 42


@dataclass
class RlConfig:
    reward_step: float = -1.0
    reward_fail: float = -10.0
    reward_success: float = 10.0
    reward_safe: float = 1.0
    reward_unsafe: float = -1.0
    reward_progress: float = 0.25
    reward_regress: float = -0.25
    dc_threshold_safe: float = 0.5
    drl_threshold_safe: float = 0.2
    dc_threshold_unsafe: float = 0.3
    drl_threshold_unsafe: float = 0.5
    gamma: float = 0.9
    alpha: float = 0.1
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    beta: float = 0.0
    beta_increment: float = 0.01
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    checkpoint_episodes: int = 100


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    graphics: GraphicsConfig = field(default_factory=GraphicsConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarConfig = field(default_factory=CarConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    som: SomConfig = field(default_factory=SomConfig)
    rl: RlConfig = field(default_factory=RlConfig)
    root_dir: Path = field(default_factory=lambda: Path.cwd())

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a config path relative to the config file directory."""
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root_dir / path

    @property
    def tracks_dir(self) -> Path:
        return self.resolve_path(self.paths.tracks_dir)

    @property
    def kohonen_dir(self) -> Path:
        return self.resolve_path(self.paths.kohonen_dir)

    @property
    def rl_dir(self) -> Path:
        return self.resolve_path(self.paths.rl_dir)

    def ensure_output_dirs(self) -> None:
        """Create output directories used by scripts."""
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.kohonen_dir.mkdir(parents=True, exist_ok=True)
        self.rl_dir.mkdir(parents=True, exist_ok=True)

    def to_plain_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root_dir"] = str(self.root_dir)
        return data


def _coerce_value(default: Any, value: Any) -> Any:
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(default, Path):
        return Path(value)
    return value


def _apply_section(section: Any, values: dict[str, Any]) -> None:
    if isinstance(section, CarConfig) and "max_steering_rate_deg" in values:
        values = dict(values)
        values["max_steering_acceleration_deg"] = values.pop("max_steering_rate_deg")
    valid_fields = {item.name: item for item in fields(section)}
    for key, value in values.items():
        if key not in valid_fields:
            continue
        current = getattr(section, key)
        setattr(section, key, _coerce_value(current, value))


def _apply_config(config: AppConfig, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(config, key):
            continue
        section = getattr(config, key)
        if is_dataclass(section) and isinstance(value, dict):
            _apply_section(section, value)
        elif key != "root_dir":
            setattr(config, key, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from TOML, falling back to dataclass defaults."""
    config_path = Path(path) if path else Path.cwd() / "config.toml"
    config = AppConfig(root_dir=config_path.resolve().parent)
    if config_path.exists():
        with config_path.open("rb") as handle:
            _apply_config(config, tomllib.load(handle))
    return config
