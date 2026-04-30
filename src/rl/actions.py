"""Discrete action space for the car controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.settings import AppConfig


@dataclass(frozen=True)
class ActionSpace:
    """Cartesian product of linear and angular accelerations."""

    actions: tuple[tuple[float, float], ...]

    @classmethod
    def from_config(cls, config: AppConfig) -> "ActionSpace":
        accelerations = (
            -float(config.car.max_acceleration),
            0.0,
            float(config.car.max_acceleration),
        )
        angular_accelerations = np.linspace(
            -float(config.car.max_steering_acceleration_deg),
            float(config.car.max_steering_acceleration_deg),
            9,
        )
        return cls(
            tuple(
                (accel, float(angular_acceleration))
                for accel in accelerations
                for angular_acceleration in angular_accelerations
            )
        )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return self.actions[index]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.actions, dtype=float)
