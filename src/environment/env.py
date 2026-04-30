"""Driving environment used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely.geometry import LineString

from src.config.settings import AppConfig
from src.environment.car import Car, CarState
from src.environment.lidar import Lidar, LidarReading
from src.environment.track import Track


TerminalReason = Literal["running", "success", "off_track", "collision", "timeout"]


@dataclass
class StepResult:
    observation: tuple[float, float]
    lidar: LidarReading
    done: bool
    success: bool
    reason: TerminalReason


class DrivingEnv:
    """Minimal episodic driving simulator."""

    def __init__(
        self,
        config: AppConfig,
        track: Track,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.track = track
        self.rng = rng or np.random.default_rng(config.simulation.random_seed)
        self.car = Car(
            length=config.car.length,
            width=config.car.width,
            max_speed=config.car.max_speed,
            min_speed=config.car.min_speed,
            state=CarState(0.0, 0.0, 0.0, 0.0),
        )
        self.lidar = Lidar(config.lidar.max_distance, config.lidar.side_angle_deg)
        self.steps = 0
        self.done = False
        self.reason: TerminalReason = "running"
        self.last_lidar: LidarReading | None = None
        self.previous_position = (0.0, 0.0)

    def reset(
        self,
        track: Track | None = None,
        random_start: bool = False,
        random_pose: bool = False,
    ) -> StepResult:
        if track is not None:
            self.track = track
        if random_pose:
            x, y, heading = self.track.sample_drivable_point(
                self.rng,
                obstacle_clearance=max(self.config.car.length, self.config.car.width) / 2.0,
            )
        else:
            x, y, heading = self.track.start_pose(
                self.rng if random_start else None,
                random_on_start_line=random_start,
            )
        self.car.reset(CarState(x, y, heading, 0.0))
        self.steps = 0
        self.done = False
        self.reason = "running"
        self.previous_position = (x, y)
        self.last_lidar = self.lidar.scan(self.car.state, self.track)
        return StepResult(self.observation(), self.last_lidar, False, False, "running")

    def observation(self) -> tuple[float, float]:
        if self.last_lidar is None:
            self.last_lidar = self.lidar.scan(self.car.state, self.track)
        return self.last_lidar.vector

    def step(self, action: tuple[float, float]) -> StepResult:
        if self.done:
            assert self.last_lidar is not None
            return StepResult(self.observation(), self.last_lidar, True, self.reason == "success", self.reason)

        acceleration, steering_acceleration = action
        self.previous_position = (self.car.state.x, self.car.state.y)
        self.car.step(acceleration, steering_acceleration, self.config.simulation.dt)
        self.steps += 1
        reason = self._terminal_reason()
        self.done = reason != "running"
        self.reason = reason
        self.last_lidar = self.lidar.scan(self.car.state, self.track)
        return StepResult(
            observation=self.observation(),
            lidar=self.last_lidar,
            done=self.done,
            success=reason == "success",
            reason=reason,
        )

    def _terminal_reason(self) -> TerminalReason:
        path = LineString([self.previous_position, (self.car.state.x, self.car.state.y)])
        if path.intersects(self.track.finish_geometry):
            return "success"
        polygon = self.car.polygon()
        if not self.track.contains_polygon(polygon) and not polygon.intersects(self.track.start_geometry):
            return "off_track"
        if self.track.obstacle_hit(polygon):
            return "collision"
        if self.steps >= self.config.simulation.max_steps_per_episode:
            return "timeout"
        return "running"
