"""Headless model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config.settings import AppConfig
from src.environment.car import Car
from src.environment.env import DrivingEnv
from src.environment.lidar import LidarReading
from src.environment.track import Track, load_track, load_tracks
from src.kohonen.som import SomDiscretizer, load_som_model
from src.rl.actions import ActionSpace
from src.rl.q_learning import QTableBundle, load_q_table, safety_ratios


METRIC_FIELDS = [
    "model_label",
    "model_folder",
    "model_path",
    "is_base",
    "som_path",
    "lidar_num_rays",
    "track_name",
    "track_path",
    "success",
    "terminal_reason",
    "time",
    "distance_traveled",
    "circuit_distance",
    "distance_percentage",
    "safety_margin",
    "unsafe_steps",
    "total_steps",
    "unsafe_step_percentage",
    "unsafe_value",
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    folder: Path
    path: Path
    is_base: bool = False


def discover_model_folders(config: AppConfig) -> list[Path]:
    root = config.rl_dir / "bk"
    if not root.exists():
        root = config.rl_dir
    return sorted(
        {
            path.parent
            for path in root.rglob("qtable*.npz")
            if path.is_file()
        }
    )


def model_label(folder: str | Path) -> str:
    path = Path(folder)
    if path.parent.name == "bk":
        return path.name
    if path.parent == path:
        return path.name
    return f"{path.parent.name}/{path.name}"


def latest_hard_qtable(folder: str | Path) -> Path:
    paths = sorted(Path(folder).glob("qtable*hard*.npz"))
    if not paths:
        raise FileNotFoundError(f"no qtable*hard*.npz found in {folder}")
    return max(paths, key=lambda path: (_checkpoint_episode(path), path.name, path.stat().st_mtime))


def default_base_qtable(config: AppConfig) -> Path:
    return latest_hard_qtable(config.rl_dir / "bk" / "5-lidar")


def build_model_specs(
    config: AppConfig,
    model_folders: Iterable[str | Path] | None,
    base_qtable: str | Path | None,
) -> list[ModelSpec]:
    folders = [Path(folder) for folder in model_folders] if model_folders else discover_model_folders(config)
    specs = [
        ModelSpec(
            label=model_label(folder),
            folder=folder,
            path=latest_hard_qtable(folder),
            is_base=False,
        )
        for folder in folders
    ]
    base_path = Path(base_qtable) if base_qtable else default_base_qtable(config)
    if not base_path.exists():
        raise FileNotFoundError(f"base Q-table not found: {base_path}")
    if all(spec.path.resolve() != base_path.resolve() for spec in specs):
        specs.append(
            ModelSpec(
                label=f"base:{base_path.parent.name}",
                folder=base_path.parent,
                path=base_path,
                is_base=True,
            )
        )
    else:
        specs = [
            ModelSpec(spec.label, spec.folder, spec.path, spec.path.resolve() == base_path.resolve())
            for spec in specs
        ]
    return specs


def evaluate_models(
    config: AppConfig,
    specs: Iterable[ModelSpec],
    track_pairs: list[tuple[Path, Track]] | None = None,
) -> list[dict[str, object]]:
    pairs = track_pairs if track_pairs is not None else load_tracks(config.tracks_dir)
    if not pairs:
        raise FileNotFoundError("no tracks found; run draw_tracks.py first")
    rows: list[dict[str, object]] = []
    for spec in specs:
        policy = load_policy(config, spec.path)
        for track_path, track in pairs:
            rows.append(evaluate_model_on_track(config, spec, policy, track_path, track))
    return rows


@dataclass(frozen=True)
class Policy:
    bundle: QTableBundle
    som: SomDiscretizer
    actions: np.ndarray
    lidar_num_rays: int


def load_policy(config: AppConfig, qtable_path: str | Path) -> Policy:
    bundle = load_q_table(qtable_path, config)
    som = load_som_model(bundle.som_path or None, config)
    lidar_num_rays = bundle.lidar_num_rays or som.lidar_num_rays
    if lidar_num_rays != som.lidar_num_rays:
        raise ValueError(
            f"Q-table uses {lidar_num_rays} lidar rays but SOM expects {som.lidar_num_rays}"
        )
    actions = _actions_for_bundle(config, bundle)
    return Policy(bundle=bundle, som=som, actions=actions, lidar_num_rays=lidar_num_rays)


def evaluate_model_on_track(
    config: AppConfig,
    spec: ModelSpec,
    policy: Policy,
    track_path: Path,
    track: Track,
) -> dict[str, object]:
    env = DrivingEnv(config, track, lidar_num_rays=policy.lidar_num_rays)
    result = env.reset(random_start=False)
    distance_traveled = 0.0
    unsafe_steps = 0
    unsafe_value_sum = 0.0
    total_steps = 0
    margin = np.inf
    observed_margin = False
    previous_position = np.asarray((env.car.state.x, env.car.state.y), dtype=float)

    while not result.done:
        state = policy.som.state_from_lidar(result.lidar)
        action_index = int(np.argmax(policy.bundle.q_table[state]))
        action = tuple(float(value) for value in policy.actions[action_index])
        result = env.step(action)
        current_position = np.asarray((env.car.state.x, env.car.state.y), dtype=float)
        distance_traveled += float(np.linalg.norm(current_position - previous_position))
        previous_position = current_position
        total_steps += 1
        unsafeness = step_unsafeness(config, result.lidar)
        unsafe_value_sum += unsafeness
        if unsafeness > 0.0:
            unsafe_steps += 1
        if total_steps > config.analysis.safety_margin_ignore_steps:
            margin = min(margin, safety_margin(track, env.car))
            observed_margin = True
        margin_text = f"{margin:.2f}" if observed_margin else "n/a"
        print(f"Evaluating {spec.label} on {track_path.name}: step {total_steps}, distance {distance_traveled:.2f}, margin {margin_text}, unsafe value {unsafe_value_sum / total_steps if total_steps > 0 else 0.0}", end="\r")

    circuit_distance = track_circuit_distance(track)
    percentage = distance_percentage(distance_traveled, circuit_distance)
    unsafe_percentage = unsafe_steps / total_steps if total_steps > 0 else 0.0
    unsafe_value = unsafe_value_sum / total_steps if total_steps > 0 else 0.0
    final_margin = margin if observed_margin else np.nan
    return {
        "model_label": spec.label,
        "model_folder": str(spec.folder),
        "model_path": str(spec.path),
        "is_base": "yes" if spec.is_base else "no",
        "som_path": str(policy.som.source_path or ""),
        "lidar_num_rays": policy.lidar_num_rays,
        "track_name": track_path.name,
        "track_path": str(track_path),
        "success": "yes" if result.success else "no",
        "terminal_reason": result.reason,
        "time": total_steps,
        "distance_traveled": distance_traveled,
        "circuit_distance": circuit_distance,
        "distance_percentage": percentage,
        "safety_margin": final_margin,
        "unsafe_steps": unsafe_steps,
        "total_steps": total_steps,
        "unsafe_step_percentage": unsafe_percentage,
        "unsafe_value": unsafe_value,
    }


def track_circuit_distance(track: Track) -> float:
    return float(track.center_line_string.length)


def distance_percentage(distance_traveled: float, circuit_distance: float) -> float:
    return float(distance_traveled / circuit_distance) if circuit_distance > 0.0 else 0.0


def safety_margin(track: Track, car: Car) -> float:
    polygon = car.polygon()
    intersects_finish = polygon.intersects(track.finish_geometry)
    if (not track.contains_polygon(polygon) and not intersects_finish) or track.obstacle_hit(polygon):
        return 0.0
    boundary = _road_boundary_without_finish(track)
    distances = [] if boundary.is_empty else [polygon.distance(boundary)]
    distances.extend(polygon.distance(obstacle.geometry()) for obstacle in track.obstacles)
    return float(min(distances)) if distances else 0.0


def is_unsafe_step(config: AppConfig, reading: LidarReading) -> bool:
    return step_unsafeness(config, reading) > 0.0


def step_unsafeness(config: AppConfig, reading: LidarReading) -> float:
    dc_ratio, drl_ratio, drrll_ratio = safety_ratios(config, reading)
    unsafe = dc_ratio < config.rl.dc_threshold_unsafe or any(
        ratio > config.rl.drl_threshold_unsafe for ratio in (drl_ratio, drrll_ratio)
    )
    if not unsafe:
        return 0.0
    return float(max(0.0, 1.0 - dc_ratio, drl_ratio, drrll_ratio))


def _road_boundary_without_finish(track: Track):
    finish_buffer = track.finish_geometry.buffer(max(track.track_width * 1e-6, 1e-6))
    return track.road.boundary.difference(finish_buffer)


def write_metrics_csv(rows: list[dict[str, object]], path: str | Path | None, config: AppConfig) -> Path:
    if path is None:
        directory = config.root_dir / "out" / "analysis"
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = directory / f"metrics_{timestamp}.csv"
    else:
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
    return csv_path


def _actions_for_bundle(config: AppConfig, bundle: QTableBundle) -> np.ndarray:
    actions = np.asarray(bundle.actions, dtype=float)
    if actions.shape[0] == bundle.q_table.shape[1]:
        return actions
    configured = ActionSpace.from_config(config).as_array()
    if configured.shape[0] == bundle.q_table.shape[1]:
        return configured
    raise ValueError(
        f"Q-table has {bundle.q_table.shape[1]} actions, "
        f"but saved actions have {actions.shape[0]} and configured actions have {configured.shape[0]}"
    )


def _checkpoint_episode(path: Path) -> int:
    match = re.search(r"qtable_checkpoint_(\d+)", path.name)
    if match:
        return int(match.group(1))
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(data["episode"])
    except Exception:
        return -1
