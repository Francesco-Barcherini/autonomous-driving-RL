"""Car state and simple Euler kinematics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Polygon


@dataclass
class CarState:
    """Pose and scalar velocity of the simulated car."""

    x: float
    y: float
    heading: float
    velocity: float = 0.0

    def copy(self) -> "CarState":
        return CarState(
            self.x,
            self.y,
            self.heading,
            self.velocity,
        )


@dataclass
class Car:
    """Rectangular vehicle controlled by acceleration and heading delta."""

    length: float
    width: float
    max_speed: float
    min_speed: float
    state: CarState

    def reset(self, state: CarState) -> None:
        self.state = state.copy()

    def step(self, acceleration: float, steering_diff_angle_deg: float, dt: float) -> CarState:
        """Advance the car with explicit Euler integration."""
        self.state.velocity += acceleration * dt
        self.state.velocity = max(self.min_speed, min(self.max_speed, self.state.velocity))
        self.state.heading = _wrap_angle(
            self.state.heading + math.radians(steering_diff_angle_deg)
        )
        self.state.x += math.cos(self.state.heading) * self.state.velocity * dt
        self.state.y += math.sin(self.state.heading) * self.state.velocity * dt
        return self.state

    def corners(self) -> list[tuple[float, float]]:
        """Return rectangle corners in clockwise order."""
        half_l = self.length / 2.0
        half_w = self.width / 2.0
        local = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]
        cos_h = math.cos(self.state.heading)
        sin_h = math.sin(self.state.heading)
        return [
            (
                self.state.x + lx * cos_h - ly * sin_h,
                self.state.y + lx * sin_h + ly * cos_h,
            )
            for lx, ly in local
        ]

    def polygon(self) -> Polygon:
        return Polygon(self.corners())


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
