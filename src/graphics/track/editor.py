"""Interactive Pygame track editor."""

from __future__ import annotations

import math

import pygame

from src.config.settings import AppConfig
from src.environment.track import Obstacle, Track
from src.graphics.renderer import Renderer


def run_track_editor(config: AppConfig, name: str = "track") -> None:
    """Run the track editor and save JSON tracks into the configured folder."""
    config.ensure_output_dirs()
    renderer = Renderer(config, "Draw Tracks")
    points: list[tuple[float, float]] = []
    obstacles: list[Obstacle] = []
    track_width = config.track.track_width
    obstacle_radius = config.track.obstacle_radius
    pending_line: tuple[str, tuple[float, float]] | None = None
    start_line: list[tuple[float, float]] | None = None
    finish_line: list[tuple[float, float]] | None = None
    running = True
    left_down = False
    saved_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    track_width += 5.0
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    track_width = max(20.0, track_width - 5.0)
                elif event.key == pygame.K_c:
                    points.clear()
                    obstacles.clear()
                    start_line = None
                    finish_line = None
                    saved_message = ""
                elif event.key == pygame.K_s:
                    pending_line = ("start", pygame.mouse.get_pos())
                elif event.key == pygame.K_f:
                    pending_line = ("finish", pygame.mouse.get_pos())
                elif event.key == pygame.K_a and len(points) >= 2:
                    start_line, finish_line = _auto_lines(points, track_width)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if len(points) >= 2:
                        if start_line is None or finish_line is None:
                            start_line, finish_line = _auto_lines(points, track_width)
                        track = Track(
                            name=name,
                            track_width=track_width,
                            centerline=points,
                            start_line=start_line,
                            finish_line=finish_line,
                            obstacles=obstacles,
                        )
                        path = track.save(config.tracks_dir)
                        saved_message = f"saved {path.name}"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    left_down = True
                    if pending_line is not None:
                        pending_line = (pending_line[0], event.pos)
                    else:
                        _append_point(points, event.pos, config.track.editor_min_point_distance)
                elif event.button == 3:
                    _toggle_obstacle(points, obstacles, event.pos, obstacle_radius, track_width)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    left_down = False
                    if pending_line is not None:
                        line_name, first = pending_line
                        line = [(float(first[0]), float(first[1])), (float(event.pos[0]), float(event.pos[1]))]
                        if line_name == "start":
                            start_line = line
                        else:
                            finish_line = line
                        pending_line = None
            elif event.type == pygame.MOUSEMOTION and left_down and pending_line is None:
                _append_point(points, event.pos, config.track.editor_min_point_distance)

        preview_track = _preview_track(name, points, track_width, start_line, finish_line, obstacles)
        lines = [
            "Left drag: centerline",
            "Right click: obstacle",
            "+/- width  S/F lines",
            "A auto gates  Enter save",
            f"width {track_width:.0f}  obstacles {len(obstacles)}",
        ]
        if saved_message:
            lines.append(saved_message)
        renderer.draw(preview_track, lines=lines)
        renderer.tick()

    renderer.quit()


def _append_point(points: list[tuple[float, float]], point, min_distance: float) -> None:
    candidate = (float(point[0]), float(point[1]))
    if not points:
        points.append(candidate)
        return
    if math.hypot(candidate[0] - points[-1][0], candidate[1] - points[-1][1]) >= min_distance:
        points.append(candidate)


def _toggle_obstacle(
    points: list[tuple[float, float]],
    obstacles: list[Obstacle],
    point,
    radius: float,
    track_width: float,
) -> None:
    px, py = float(point[0]), float(point[1])
    for index, obstacle in enumerate(obstacles):
        if math.hypot(px - obstacle.x, py - obstacle.y) <= obstacle.radius * 1.35:
            del obstacles[index]
            return
    if len(points) < 2:
        return
    preview = _preview_track("preview", points, track_width, None, None, obstacles)
    if preview.contains_point((px, py)):
        obstacles.append(Obstacle(px, py, radius))


def _auto_lines(
    points: list[tuple[float, float]],
    track_width: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    return [
        _gate(points[0], points[1], track_width),
        _gate(points[-1], points[-2], track_width),
    ]


def _gate(
    point: tuple[float, float],
    neighbor: tuple[float, float],
    track_width: float,
) -> list[tuple[float, float]]:
    dx = neighbor[0] - point[0]
    dy = neighbor[1] - point[1]
    length = max(math.hypot(dx, dy), 1e-9)
    nx = -dy / length
    ny = dx / length
    half = track_width * 0.55
    return [
        (point[0] - nx * half, point[1] - ny * half),
        (point[0] + nx * half, point[1] + ny * half),
    ]


def _preview_track(
    name: str,
    points: list[tuple[float, float]],
    track_width: float,
    start_line: list[tuple[float, float]] | None,
    finish_line: list[tuple[float, float]] | None,
    obstacles: list[Obstacle],
) -> Track:
    if len(points) < 2:
        centerline = [(0.0, 0.0), (0.1, 0.0)]
    else:
        centerline = points
    if start_line is None or finish_line is None:
        auto_start, auto_finish = _auto_lines(centerline, track_width)
        start_line = start_line or auto_start
        finish_line = finish_line or auto_finish
    return Track(name, track_width, list(centerline), start_line, finish_line, list(obstacles))
