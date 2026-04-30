"""Q-learning training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.settings import AppConfig
from src.environment.env import DrivingEnv
from src.environment.track import Track, load_tracks
from src.graphics.renderer import Renderer
from src.kohonen.som import SomDiscretizer, load_som_model
from src.rl.actions import ActionSpace
from src.rl.q_learning import QAgent, compute_reward, save_q_table


def train_q_learning(
    config: AppConfig,
    som_path: str | Path | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    headless: bool = False,
) -> Path:
    config.ensure_output_dirs()
    config.simulation.max_steps_per_episode = config.rl.max_steps_per_episode
    som = load_som_model(som_path, config)
    track_pairs = load_tracks(config.tracks_dir)
    if not track_pairs:
        raise FileNotFoundError("no tracks found in out/tracks; run draw_tracks.py first")

    tracks = [track for _, track in track_pairs]
    rng = np.random.default_rng(seed if seed is not None else config.simulation.random_seed)
    actions = ActionSpace.from_config(config)
    agent = QAgent.fresh(som.num_states, len(actions), config)
    episode_count = episodes if episodes is not None else config.rl.num_episodes
    view = None if headless else RlTrainingView(config, som.grid_dim)
    visible = True

    last_path = None
    for episode in range(episode_count):
        track = tracks[episode % len(tracks)]
        env = DrivingEnv(config, track, rng)
        result = env.reset()
        state = som.state_from_lidar(result.lidar)

        total_reward = 0
        print("")

        while not result.done:
            action_index = agent.select_action(state, rng)
            action = actions[action_index]
            result = env.step(action)
            next_state = som.state_from_lidar(result.lidar)
            reward = compute_reward(
                config,
                result,
                track=env.track,
                previous_position=env.previous_position,
                current_position=(env.car.state.x, env.car.state.y),
            )
            total_reward += reward
            print(total_reward, end="\r")
            agent.update(state, action_index, reward, next_state, result.done)

            if view is not None:
                visible = view.process_events(visible)
                view.draw(
                    env,
                    som,
                    next_state,
                    action,
                    agent.q_table,
                    episode + 1,
                    episode_count,
                    agent,
                    visible,
                )
            state = next_state

        agent.decay_epsilon()
        agent.increment_beta()
        if (episode + 1) % config.rl.checkpoint_episodes == 0:
            last_path = save_q_table(
                config.rl_dir,
                agent,
                actions,
                config,
                som.source_path,
                episode + 1,
                checkpoint=True,
            )

    if view is not None:
        view.close()

    return save_q_table(
        config.rl_dir,
        agent,
        actions,
        config,
        som.source_path,
        episode_count,
        checkpoint=False,
    ) or last_path


class RlTrainingView:
    """Optional debug view toggled with G while Q-learning runs."""

    def __init__(self, config: AppConfig, grid_dim: int) -> None:
        self.renderer = Renderer(config, "Q-Learning Training")
        self.grid_dim = grid_dim

    def process_events(self, visible: bool) -> bool:
        pygame = self.renderer.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                return not visible
        return visible

    def draw(
        self,
        env: DrivingEnv,
        som: SomDiscretizer,
        state: int,
        action: tuple[float, float],
        q_table: np.ndarray,
        episode: int,
        total_episodes: int,
        agent: QAgent,
        visible: bool,
    ) -> None:
        if not visible:
            self.renderer.tick()
            return
        self.renderer.draw(
            env.track,
            car=env.car,
            lidar=env.last_lidar,
            som_state=state,
            som_dim=som.grid_dim,
            action=action,
            q_table=q_table,
            lines=[
                f"episode {episode}/{total_episodes}",
                f"step {env.steps}",
                f"gamma {agent.gamma:.2f} alpha {agent.alpha:.2f}",
                f"epsilon {agent.epsilon:.4f}",
                f"beta {agent.beta:.4f} inc {agent.beta_increment:.4f}",
                "G hides this view",
            ],
        )
        self.renderer.tick()

    def close(self) -> None:
        self.renderer.quit()
