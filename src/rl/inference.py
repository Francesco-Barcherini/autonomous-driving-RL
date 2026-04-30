"""Inference-time Pygame demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from src.config.settings import AppConfig
from src.environment.env import DrivingEnv
from src.environment.track import Track, load_track, load_tracks
from src.graphics.renderer import Renderer
from src.kohonen.som import load_som_model
from src.rl.actions import ActionSpace
from src.rl.q_learning import load_q_table


def run_inference(
    config: AppConfig,
    track_selector: str | None = None,
    som_path: str | Path | None = None,
    q_table_path: str | Path | None = None,
) -> None:
    """Run the trained policy in an interactive Pygame window."""
    config.ensure_output_dirs()
    track_pairs = load_tracks(config.tracks_dir)
    if not track_pairs:
        raise FileNotFoundError("no tracks found in out/tracks; run draw_tracks.py first")
    current_index = _resolve_track_index(track_selector, track_pairs)
    q_bundle = load_q_table(q_table_path, config)
    selected_som = som_path or (q_bundle.som_path if q_bundle.som_path else None)
    som = load_som_model(selected_som, config)
    actions = q_bundle.actions
    if actions.shape[0] != q_bundle.q_table.shape[1]:
        actions = ActionSpace.from_config(config).as_array()

    renderer = Renderer(config, "Inference")
    rng = np.random.default_rng(config.simulation.random_seed)
    track = load_track(track_pairs[current_index][0])
    env = DrivingEnv(config, track, rng)
    result = env.reset(random_start=False)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    track = load_track(track_pairs[current_index][0])
                    result = env.reset(track=track, random_start=False)
                else:
                    requested = _event_digit(event)
                    if requested is None:
                        continue
                    if requested < len(track_pairs):
                        current_index = requested
                        track = load_track(track_pairs[current_index][0])
                        result = env.reset(track=track, random_start=False)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):
                    env.track.toggle_obstacle(event.pos, config.track.obstacle_radius)

        if not result.done:
            state = som.state_from_lidar(result.lidar)
            action_index = int(np.argmax(q_bundle.q_table[state]))
            action = tuple(float(value) for value in actions[action_index])
            result = env.step(action)
        else:
            state = som.state_from_lidar(result.lidar)
            action_index = int(np.argmax(q_bundle.q_table[state]))
            action = tuple(float(value) for value in actions[action_index])

        renderer.draw(
            env.track,
            car=env.car,
            lidar=result.lidar,
            som_state=state,
            som_dim=som.grid_dim,
            action=action,
            lines=[
                f"track {current_index}: {track_pairs[current_index][0].name}",
                f"state {state}  action {action_index}",
                f"result {result.reason}",
                "Esc quit  R reset",
                "0-9 select track",
                "click add/remove obstacle",
            ],
        )
        renderer.tick()

    renderer.quit()


def _resolve_track_index(
    selector: str | None,
    track_pairs: list[tuple[Path, Track]],
) -> int:
    if selector is None:
        return 0
    if selector.isdigit():
        index = int(selector)
        if 0 <= index < len(track_pairs):
            return index
        raise IndexError(f"track index {index} is outside 0..{len(track_pairs) - 1}")
    requested = Path(selector).resolve()
    for index, (path, _) in enumerate(track_pairs):
        if path.resolve() == requested:
            return index
    raise FileNotFoundError(f"track {selector!r} was not found in {track_pairs[0][0].parent}")


def _event_digit(event: pygame.event.Event) -> int | None:
    if event.unicode.isdigit():
        return int(event.unicode)
    keypad_digits = {
        pygame.K_KP0: 0,
        pygame.K_KP1: 1,
        pygame.K_KP2: 2,
        pygame.K_KP3: 3,
        pygame.K_KP4: 4,
        pygame.K_KP5: 5,
        pygame.K_KP6: 6,
        pygame.K_KP7: 7,
        pygame.K_KP8: 8,
        pygame.K_KP9: 9,
    }
    return keypad_digits.get(event.key)
