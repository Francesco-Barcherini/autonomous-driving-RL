"""Discrete action space for the car controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.settings import AppConfig


@dataclass(frozen=True)
class ActionSpace:
    """Cartesian product of acceleration and steering angle deltas."""

    actions: tuple[tuple[float, float], ...]

    @classmethod
    def from_config(cls, config: AppConfig) -> "ActionSpace":
        accelerations = (
            -float(config.car.max_acceleration),
            0.0,
            float(config.car.max_acceleration),
        )
        steering_diffs = np.linspace(
            -float(config.car.max_steering_diff_angle_deg),
            float(config.car.max_steering_diff_angle_deg),
            9,
        )
        return cls(
            tuple(
                (accel, float(steering_diff))
                for accel in accelerations
                for steering_diff in steering_diffs
            )
        )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.actions[index]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=float)
