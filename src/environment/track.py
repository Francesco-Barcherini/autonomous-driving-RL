"""Track, obstacle, and JSON persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, Point, Polygon


Point2D = tuple[float, float]


@dataclass
class Obstacle:
    """Circular obstacle placed inside a track."""

    x: float
    y: float
    radius: float

    @property
    def center(self) -> Point2D:
        return (self.x, self.y)

    def geometry(self) -> Polygon:
        return Point(self.x, self.y).buffer(self.radius)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Obstacle":
        return cls(float(data["x"]), float(data["y"]), float(data["radius"]))


@dataclass
class Track:
    """A drivable road built from a centerline buffer."""

    name: str
    track_width: float
    centerline: list[Point2D]
    start_line: list[Point2D]
    finish_line: list[Point2D]
    obstacles: list[Obstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.centerline = [_as_point_tuple(point) for point in self.centerline]
        self.start_line = [_as_point_tuple(point) for point in self.start_line]
        self.finish_line = [_as_point_tuple(point) for point in self.finish_line]

    @property
    def road(self) -> Polygon:
        if len(self.centerline) < 2:
            return Polygon()
        line = LineString(self.centerline)
        return line.buffer(
            self.track_width / 2.0,
            cap_style=2,
            join_style=1,
        )

    @property
    def center_line_string(self) -> LineString:
        return LineString(self.centerline)

    @property
    def start_geometry(self) -> LineString:
        return LineString(self.start_line)

    @property
    def finish_geometry(self) -> LineString:
        return LineString(self.finish_line)

    def is_valid(self) -> bool:
        return (
            len(self.centerline) >= 2
            and len(self.start_line) == 2
            and len(self.finish_line) == 2
            and not self.road.is_empty
        )

    def contains_point(self, point: Point2D) -> bool:
        return bool(self.road.covers(Point(point)))

    def contains_polygon(self, polygon: Polygon) -> bool:
        return bool(self.road.covers(polygon))

    def obstacle_hit(self, polygon: Polygon) -> bool:
        return any(polygon.intersects(obstacle.geometry()) for obstacle in self.obstacles)

    def toggle_obstacle(self, point: Point2D, radius: float) -> bool:
        """Remove a nearby obstacle, or add one when the point is drivable."""
        px, py = point
        for index, obstacle in enumerate(self.obstacles):
            if math.hypot(px - obstacle.x, py - obstacle.y) <= obstacle.radius * 1.35:
                del self.obstacles[index]
                return False
        if self.contains_point(point):
            self.obstacles.append(Obstacle(px, py, radius))
            return True
        return False

    def start_pose(
        self,
        rng: np.random.Generator | None = None,
        random_on_start_line: bool = False,
    ) -> tuple[float, float, float]:
        """Return the default start pose, or sample one on the start line."""
        if not random_on_start_line or len(self.start_line) != 2:
            first = self.centerline[0]
            return (first[0], first[1], self.initial_heading())

        generator = rng or np.random.default_rng()
        t = float(generator.random())
        (x0, y0), (x1, y1) = self.start_line
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        return (x, y, self.initial_heading())

    def initial_heading(self) -> float:
        """Heading from the first to the second centerline point."""
        if len(self.centerline) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.centerline[0], self.centerline[1]
        return math.atan2(y1 - y0, x1 - x0)

    def heading_near(self, point: Point2D) -> float:
        """Estimate tangent heading of the centerline near a point."""
        if len(self.centerline) < 2:
            return 0.0
        line = self.center_line_string
        distance = line.project(Point(point))
        epsilon = min(max(self.track_width * 0.2, 1.0), max(line.length * 0.05, 1.0))
        d0 = max(0.0, distance - epsilon)
        d1 = min(line.length, distance + epsilon)
        p0 = line.interpolate(d0)
        p1 = line.interpolate(d1)
        if p0.distance(p1) < 1e-9 and len(self.centerline) >= 2:
            p0 = Point(self.centerline[0])
            p1 = Point(self.centerline[1])
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def sample_drivable_point(
        self,
        rng: np.random.Generator,
        obstacle_clearance: float = 0.0,
        max_attempts: int = 2000,
    ) -> tuple[float, float, float]:
        """Sample a random road point that is not inside an obstacle."""
        road = self.road
        minx, miny, maxx, maxy = road.bounds
        for _ in range(max_attempts):
            x = float(rng.uniform(minx, maxx))
            y = float(rng.uniform(miny, maxy))
            point = Point(x, y)
            if not road.covers(point):
                continue
            if any(point.distance(Point(o.x, o.y)) <= o.radius + obstacle_clearance for o in self.obstacles):
                continue
            return (x, y, self.heading_near((x, y)))
        raise RuntimeError(f"could not sample a drivable point from track {self.name!r}")

    def with_obstacles(self, obstacles: Iterable[Obstacle]) -> "Track":
        return Track(
            name=self.name,
            track_width=self.track_width,
            centerline=list(self.centerline),
            start_line=list(self.start_line),
            finish_line=list(self.finish_line),
            obstacles=list(obstacles),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "track_width": self.track_width,
            "centerline": [list(point) for point in self.centerline],
            "start_line": [list(point) for point in self.start_line],
            "finish_line": [list(point) for point in self.finish_line],
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            name=str(data.get("name", "track")),
            track_width=float(data["track_width"]),
            centerline=[_as_point_tuple(point) for point in data["centerline"]],
            start_line=[_as_point_tuple(point) for point in data["start_line"]],
            finish_line=[_as_point_tuple(point) for point in data["finish_line"]],
            obstacles=[Obstacle.from_dict(item) for item in data.get("obstacles", [])],
        )

    def save(self, directory: Path, timestamp: bool = True) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if timestamp else "track"
        filename = f"{self.name}_{suffix}.json".replace(" ", "_")
        path = directory / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


def _as_point_tuple(point: Iterable[float]) -> Point2D:
    x, y = point
    return (float(x), float(y))


def load_track(path: str | Path) -> Track:
    with Path(path).open("r", encoding="utf-8") as handle:
        return Track.from_dict(json.load(handle))


def load_tracks(directory: str | Path) -> list[tuple[Path, Track]]:
    paths = sorted(Path(directory).glob("*.json"))
    return [(path, load_track(path)) for path in paths]


def latest_file(directory: str | Path, pattern: str = "*.npz") -> Path | None:
    paths = sorted(Path(directory).glob(pattern), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None
