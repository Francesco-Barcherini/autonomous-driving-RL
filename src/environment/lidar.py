"""Configurable three- or five-ray lidar simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import LineString, Point

from src.environment.car import CarState
from src.environment.track import Track


def input_len_from_lidar_num_rays(num_rays: int) -> int:
    if num_rays == 5:
        return 3
    if num_rays == 3:
        return 2
    raise ValueError("Lidar num_rays must be either 3 or 5")


def lidar_num_rays_from_input_len(input_len: int) -> int:
    return 5 if input_len >= 3 else 3


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
    middle_left: RayHit | None = None
    middle_right: RayHit | None = None

    @property
    def vector(self) -> tuple[float, ...]:
        vector = [self.center.distance, self.right.distance - self.left.distance]
        if self.middle_left is not None and self.middle_right is not None:
            vector.append(self.middle_right.distance - self.middle_left.distance)
        return tuple(vector)

    @property
    def rays(self) -> list[RayHit]:
        if self.middle_left is not None and self.middle_right is not None:
            return [self.left, self.middle_left, self.center, self.middle_right, self.right]
        return [self.left, self.center, self.right]


@dataclass
class Lidar:
    """A front ray, plus two or four side rays at fixed angles."""

    max_distance: float
    side_angle_deg: float = 45.0
    num_rays: int = 3

    def __post_init__(self) -> None:
        input_len_from_lidar_num_rays(self.num_rays)

    def scan(self, state: CarState, track: Track) -> LidarReading:
        left = self._cast(state, track, math.radians(self.side_angle_deg))
        center = self._cast(state, track, 0.0)
        right = self._cast(state, track, -math.radians(self.side_angle_deg))
        if self.num_rays == 5:
            middle_left = self._cast(state, track, math.radians(self.side_angle_deg / 2.0))
            middle_right = self._cast(state, track, -math.radians(self.side_angle_deg / 2.0))
            return LidarReading(
                center=center,
                left=left,
                right=right,
                middle_left=middle_left,
                middle_right=middle_right,
            )
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
        finish_distance = _nearest_intersection_distance(ray, track.finish_geometry, origin)
        for geometry in geometries:
            intersection = ray.intersection(geometry)
            if intersection.is_empty:
                continue
            for candidate in _candidate_points(intersection):
                if candidate.distance(track.finish_geometry) <= 1e-6:
                    continue
                distance = origin.distance(candidate)
                if finish_distance is not None and distance >= finish_distance - 1e-7:
                    continue
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


def _nearest_intersection_distance(
    ray: LineString,
    geometry: LineString,
    origin: Point,
) -> float | None:
    intersection = ray.intersection(geometry)
    if intersection.is_empty:
        return None
    distances = [
        origin.distance(candidate)
        for candidate in _candidate_points(intersection)
        if origin.distance(candidate) >= 1e-7
    ]
    return float(min(distances)) if distances else None


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
