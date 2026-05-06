"""Tabular Q-learning implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from src.config.settings import AppConfig
from src.environment.env import StepResult
from src.environment.lidar import LidarReading
from src.environment.track import Track, latest_file
from src.rl.actions import ActionSpace


@dataclass
class QTableBundle:
    q_table: np.ndarray
    actions: np.ndarray
    source_path: Path
    som_path: str
    episode: int


@dataclass
class QAgent:
    """Mutable tabular Q-learning agent."""

    q_table: np.ndarray
    gamma: float
    alpha: float
    epsilon: float
    epsilon_decay: float
    beta: float
    beta_increment: float

    @classmethod
    def fresh(cls, num_states: int, num_actions: int, config: AppConfig) -> "QAgent":
        return cls(
            q_table=np.zeros((num_states, num_actions), dtype=float),
            gamma=config.rl.gamma,
            alpha=config.rl.alpha,
            epsilon=config.rl.epsilon,
            epsilon_decay=config.rl.epsilon_decay,
            beta=config.rl.beta,
            beta_increment=config.rl.beta_increment,
        )

    @classmethod
    def resume(cls, q_table: np.ndarray, config: AppConfig) -> "QAgent":
        return cls(
            q_table=np.asarray(q_table, dtype=float).copy(),
            gamma=config.rl.gamma,
            alpha=config.rl.alpha,
            epsilon=config.rl.resume_epsilon,
            epsilon_decay=config.rl.epsilon_decay,
            beta=config.rl.resume_beta,
            beta_increment=config.rl.beta_increment,
        )

    def select_action(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            action = int(rng.integers(0, self.q_table.shape[1]))
        else:
            probabilities = stable_softmax(self.q_table[state], self.beta)
            action = int(rng.choice(np.arange(self.q_table.shape[1]), p=probabilities))
        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminal: bool,
    ) -> None:
        target = reward
        if not terminal:
            target += self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay

    def increment_beta(self) -> None:
        self.beta += self.beta_increment


def stable_softmax(values: np.ndarray, beta: float) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(beta * shifted)
    total = np.sum(exp_values)
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(values, dtype=float) / len(values)
    return exp_values / total


def compute_reward(
    config: AppConfig,
    result: StepResult,
    track: Track | None = None,
    previous_position: tuple[float, float] | None = None,
    current_position: tuple[float, float] | None = None,
    stuck_penalty: bool = False,
    action: tuple[float, float] | None = None,
) -> float:
    reward = float(config.rl.reward_step)
    if result.done:
        reward += config.rl.reward_success if result.success else config.rl.reward_fail
        return reward

    if track is not None and previous_position is not None and current_position is not None:
        reward += progress_reward(config, track, previous_position, current_position)
    if stuck_penalty:
        reward += config.rl.reward_stuck

    dc_ratio, drl_ratio = safety_ratios(config, result.lidar)
    if dc_ratio > config.rl.dc_threshold_safe and drl_ratio < config.rl.drl_threshold_safe:
        reward += config.rl.reward_safe
    if dc_ratio < config.rl.dc_threshold_unsafe or drl_ratio > config.rl.drl_threshold_unsafe:
        reward += config.rl.reward_unsafe

    # if action is not None:
    #     steering_diff_angle = abs(action[1])
    #     max_steering = float(config.car.max_steering_diff_angle_deg)
    #     if max_steering > 0.0:
    #         #reward += config.rl.reward_steering * (steering_diff_angle / max_steering)
    #         reward += -0.5 * (steering_diff_angle / max_steering)

    # if dc_ratio < drl_ratio * config.rl.front_blocked_drl_ratio:
    #     reward += config.rl.reward_front_blocked
    #     print("FRONT BLOCK")
    # if action is not None and dc_ratio < config.rl.dc_threshold_unsafe:
    #     max_steering = max(float(config.car.max_steering_diff_angle_deg), 1e-12)
    #     steering_ratio = min(abs(float(action[1])) / max_steering, 1.0)
    #     reward += config.rl.reward_front_unsafe_steering * steering_ratio
    #     print(f"STEER: reward={reward:.2f}")
    return reward


def progress_reward(
    config: AppConfig,
    track: Track,
    previous_position: tuple[float, float],
    current_position: tuple[float, float],
) -> float:
    previous_index = nearest_centerline_index(track, previous_position)
    current_index = nearest_centerline_index(track, current_position)
    if current_index > previous_index:
        return float(config.rl.reward_progress) * float(current_index - previous_index)
    if current_index < previous_index:
        return float(config.rl.reward_regress) * float(previous_index - current_index)
    return 0.0


def nearest_centerline_index(track: Track, position: tuple[float, float]) -> int:
    point = np.asarray(position, dtype=float)
    centerline = np.asarray(track.centerline, dtype=float)
    distances = np.linalg.norm(centerline - point, axis=1)
    return int(np.argmin(distances))


def safety_ratios(config: AppConfig, reading: LidarReading) -> tuple[float, float]:
    dc_ratio = reading.center.distance / max(config.lidar.max_distance, 1e-12)
    drl_ratio = abs(reading.right.distance - reading.left.distance) / max(config.lidar.max_distance, 1e-12)
    return float(dc_ratio), float(drl_ratio)


def save_q_table(
    directory: Path,
    agent: QAgent,
    actions: ActionSpace,
    config: AppConfig,
    som_path: Path | None,
    episode: int,
    checkpoint: bool = False,
    checkpoint_rewards: list[tuple[int, str, float]] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"qtable_checkpoint_{episode:06d}" if checkpoint else "qtable"
    path = directory / f"{prefix}_{timestamp}.npz"
    reward_rows = checkpoint_rewards or []
    np.savez(
        path,
        q_table=agent.q_table,
        actions=actions.as_array(),
        epsilon=np.asarray(agent.epsilon),
        beta=np.asarray(agent.beta),
        episode=np.asarray(episode),
        som_path=np.asarray(str(som_path or "")),
        config=np.asarray(json.dumps(config.to_plain_dict(), default=str)),
        checkpoint_reward_episodes=np.asarray([row[0] for row in reward_rows], dtype=int),
        checkpoint_reward_tracks=np.asarray([row[1] for row in reward_rows], dtype=str),
        checkpoint_reward_values=np.asarray([row[2] for row in reward_rows], dtype=float),
    )
    return path


def load_q_table(path: str | Path | None, config: AppConfig) -> QTableBundle:
    table_path = Path(path) if path else latest_file(config.rl_dir, "qtable*.npz")
    if table_path is None:
        raise FileNotFoundError("no Q-table found in out/rl; run train_rl.py first")
    data = np.load(table_path, allow_pickle=False)
    return QTableBundle(
        q_table=np.asarray(data["q_table"], dtype=float),
        actions=np.asarray(data["actions"], dtype=float),
        source_path=table_path,
        som_path=str(data["som_path"]),
        episode=int(data["episode"]),
    )
