"""Three-ray lidar simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import LineString, Point

from src.environment.car import CarState
from src.environment.track import Track


@dataclass
class RayHit:
    angle: float
    distance: float
    start: tuple[float, float]
    end: tuple[float, float]
    hit: tuple[float, float]


@dataclass
class LidarReading:
    center: RayHit
    left: RayHit
    right: RayHit

    @property
    def vector(self) -> tuple[float, float]:
        return (self.center.distance, self.right.distance - self.left.distance)

    @property
    def rays(self) -> list[RayHit]:
        return [self.left, self.center, self.right]


@dataclass
class Lidar:
    """A front ray and two side rays at a fixed angle."""

    max_distance: float
    side_angle_deg: float = 45.0

    def scan(self, state: CarState, track: Track) -> LidarReading:
        left = self._cast(state, track, math.radians(self.side_angle_deg))
        center = self._cast(state, track, 0.0)
        right = self._cast(state, track, -math.radians(self.side_angle_deg))
        return LidarReading(center=center, left=left, right=right)

    def _cast(self, state: CarState, track: Track, offset: float) -> RayHit:
        angle = state.heading + offset
        start = (state.x, state.y)
        full_end = (
            state.x + math.cos(angle) * self.max_distance,
            state.y + math.sin(angle) * self.max_distance,
        )
        ray = LineString([start, full_end])
        nearest_hit = None
        nearest_distance = self.max_distance

        geometries = [track.road.boundary]
        geometries.extend(obstacle.geometry().boundary for obstacle in track.obstacles)

        origin = Point(start)
        for geometry in geometries:
            intersection = ray.intersection(geometry)
            if intersection.is_empty:
                continue
            for candidate in _candidate_points(intersection):
                if candidate.distance(track.finish_geometry) <= 1e-6:
                    continue
                distance = origin.distance(candidate)
                if 1e-7 <= distance < nearest_distance:
                    nearest_distance = float(distance)
                    nearest_hit = (float(candidate.x), float(candidate.y))

        hit = nearest_hit or full_end
        return RayHit(
            angle=angle,
            distance=nearest_distance,
            start=start,
            end=full_end,
            hit=hit,
        )


def _candidate_points(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [Point(coord) for coord in geometry.coords]
    if hasattr(geometry, "geoms"):
        points: list[Point] = []
        for part in geometry.geoms:
            points.extend(_candidate_points(part))
        return points
    return []
