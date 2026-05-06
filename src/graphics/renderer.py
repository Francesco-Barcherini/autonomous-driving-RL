"""Shared Pygame rendering helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pygame

from src.config.settings import AppConfig
from src.environment.car import Car
from src.environment.lidar import LidarReading
from src.environment.track import Obstacle, Track


class Renderer:
    """Small renderer for tracks, car state, lidar, SOM state, and Q-values."""

    def __init__(self, config: AppConfig, caption: str = "Self-Driving Car 2D Demo") -> None:
        self.pygame = pygame
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode(
            (config.graphics.screen_width, config.graphics.screen_height),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)

    def tick(self) -> None:
        self.clock.tick(self.config.graphics.fps)

    def quit(self) -> None:
        self.pygame.quit()

    def draw(
        self,
        track: Track,
        car: Car | None = None,
        lidar: LidarReading | None = None,
        som_state: int | None = None,
        som_dim: int | None = None,
        action: tuple[float, float] | None = None,
        q_table: np.ndarray | None = None,
        som_feature_points: np.ndarray | None = None,
        som_weights: np.ndarray | None = None,
        som_weight_state: int | None = None,
        lines: Iterable[str] = (),
    ) -> None:
        pygame = self.pygame
        self.screen.fill(self.config.graphics.background)
        self._draw_track(track)
        if lidar is not None:
            self._draw_lidar(lidar)
        if car is not None:
            self._draw_car(car)
        self._draw_obstacles(track.obstacles)
        screen_width, _ = self.screen.get_size()
        panel_x = max(16, screen_width - 270)
        self._draw_panel(
            panel_x,
            som_state,
            som_dim,
            action,
            q_table,
            som_feature_points,
            som_weights,
            som_weight_state if som_weight_state is not None else som_state,
            lines,
        )
        pygame.display.flip()

    def _draw_track(self, track: Track) -> None:
        pygame = self.pygame
        for polygon in _iter_polygons(track.road):
            exterior = [(int(x), int(y)) for x, y in polygon.exterior.coords]
            if len(exterior) >= 3:
                pygame.draw.polygon(self.screen, self.config.graphics.road_color, exterior)
                pygame.draw.lines(
                    self.screen,
                    self.config.graphics.road_outline_color,
                    True,
                    exterior,
                    2,
                )
            for interior in polygon.interiors:
                points = [(int(x), int(y)) for x, y in interior.coords]
                if len(points) >= 3:
                    pygame.draw.polygon(self.screen, self.config.graphics.background, points)
        if len(track.start_line) == 2:
            pygame.draw.line(self.screen, (89, 210, 121), track.start_line[0], track.start_line[1], 4)
        if len(track.finish_line) == 2:
            pygame.draw.line(self.screen, (235, 235, 92), track.finish_line[0], track.finish_line[1], 4)

    def _draw_obstacles(self, obstacles: list[Obstacle]) -> None:
        pygame = self.pygame
        for obstacle in obstacles:
            pygame.draw.circle(
                self.screen,
                self.config.graphics.obstacle_color,
                (int(obstacle.x), int(obstacle.y)),
                int(obstacle.radius),
            )

    def _draw_car(self, car: Car) -> None:
        pygame = self.pygame
        points = [(int(x), int(y)) for x, y in car.corners()]
        pygame.draw.polygon(self.screen, self.config.graphics.car_color, points)
        pygame.draw.lines(self.screen, (25, 25, 25), True, points, 2)

    def _draw_lidar(self, reading: LidarReading) -> None:
        pygame = self.pygame
        for ray in reading.rays:
            pygame.draw.line(
                self.screen,
                self.config.graphics.lidar_color,
                (int(ray.start[0]), int(ray.start[1])),
                (int(ray.hit[0]), int(ray.hit[1])),
                2,
            )
            pygame.draw.circle(
                self.screen,
                self.config.graphics.lidar_color,
                (int(ray.hit[0]), int(ray.hit[1])),
                4,
            )

    def _draw_panel(
        self,
        x: int,
        som_state: int | None,
        som_dim: int | None,
        action: tuple[float, float] | None,
        q_table: np.ndarray | None,
        som_feature_points: np.ndarray | None,
        som_weights: np.ndarray | None,
        som_weight_state: int | None,
        lines: Iterable[str],
    ) -> None:
        y = 16
        for line in lines:
            self._text(line, x, y)
            y += 22
        if action is not None:
            self._text(f"accel {action[0]:.1f}  steer diff {action[1]:.1f}", x, y)
            y += 28
        if som_dim is not None:
            self._draw_som_grid(x, y, som_dim, som_state)
            y += som_dim * 18 + 24
        if som_weights is not None:
            self._draw_som_feature_map(x, y, som_feature_points, som_weights, som_weight_state)
            y += 194
        if q_table is not None:
            self._draw_q_heatmap(x, y, q_table)

    def _draw_som_grid(self, x: int, y: int, dim: int, active_state: int | None) -> None:
        pygame = self.pygame
        size = 16
        for row in range(dim):
            for col in range(dim):
                state = row * dim + col
                rect = pygame.Rect(x + col * size, y + row * size, size - 2, size - 2)
                color = (77, 91, 114)
                if state == active_state:
                    color = (245, 207, 82)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_q_heatmap(self, x: int, y: int, q_table: np.ndarray) -> None:
        pygame = self.pygame
        rows = min(q_table.shape[0], 36)
        cols = min(q_table.shape[1], 27)
        cell = 6
        values = q_table[:rows, :cols]
        max_abs = float(np.max(np.abs(values))) if values.size else 1.0
        max_abs = max(max_abs, 1e-9)
        for row in range(rows):
            for col in range(cols):
                value = float(values[row, col]) / max_abs
                if value >= 0:
                    color = (45, int(100 + 120 * value), 95)
                else:
                    color = (int(110 + 120 * abs(value)), 66, 75)
                rect = pygame.Rect(x + col * cell, y + row * cell, cell - 1, cell - 1)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_som_feature_map(
        self,
        x: int,
        y: int,
        feature_points: np.ndarray | None,
        weights: np.ndarray,
        active_state: int | None,
    ) -> None:
        pygame = self.pygame
        width = 230
        height = 170
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (30, 35, 43), rect)
        pygame.draw.rect(self.screen, (102, 114, 130), rect, 1)
        title = "input / weights" if feature_points is not None else "SOM weights"
        self._text(title, x, y - 20)

        points = np.empty((0, 2), dtype=float)
        if feature_points is not None:
            points = np.asarray(feature_points, dtype=float).reshape(-1, 2)
        weight_points = np.asarray(weights, dtype=float).reshape(-1, 2)
        if weight_points.size == 0:
            return

        all_points = weight_points if points.size == 0 else np.vstack([points, weight_points])
        min_xy = np.min(all_points, axis=0)
        max_xy = np.max(all_points, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-9)
        padding = span * 0.08
        min_xy -= padding
        max_xy += padding
        span = np.maximum(max_xy - min_xy, 1e-9)

        if points.size:
            max_dots = 700
            if len(points) > max_dots:
                stride = int(np.ceil(len(points) / max_dots))
                points = points[::stride]

            for point in points:
                px, py = _map_feature_point(point, min_xy, span, rect)
                pygame.draw.circle(self.screen, (91, 169, 216), (px, py), 2)

        for index, point in enumerate(weight_points):
            px, py = _map_feature_point(point, min_xy, span, rect)
            if index == active_state:
                pygame.draw.rect(self.screen, (246, 92, 92), pygame.Rect(px - 5, py - 5, 10, 10))
                pygame.draw.rect(self.screen, (255, 245, 170), pygame.Rect(px - 5, py - 5, 10, 10), 1)
            else:
                pygame.draw.rect(self.screen, (245, 207, 82), pygame.Rect(px - 3, py - 3, 6, 6))
        if active_state is not None:
            d_c, d_rl = weight_points[active_state]
            self._text(f"d_c_som: {d_c:.2f} d_rl_som: {d_rl:.2f}", x, y + height + 8)

    def _text(self, text: str, x: int, y: int) -> None:
        surface = self.font.render(text, True, (226, 231, 236))
        self.screen.blit(surface, (x, y))


def _iter_polygons(geometry) -> list:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]


def _map_feature_point(
    point: np.ndarray,
    min_xy: np.ndarray,
    span: np.ndarray,
    rect: pygame.Rect,
) -> tuple[int, int]:
    x_norm = (point[0] - min_xy[0]) / span[0]
    y_norm = (point[1] - min_xy[1]) / span[1]
    x = rect.left + 8 + int(x_norm * max(rect.width - 16, 1))
    y = rect.bottom - 8 - int(y_norm * max(rect.height - 16, 1))
    return (x, y)
