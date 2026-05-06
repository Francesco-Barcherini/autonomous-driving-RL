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
from src.rl.q_learning import (
    QAgent,
    compute_reward,
    load_q_table,
    nearest_centerline_index,
    save_q_table,
)


def train_q_learning(
    config: AppConfig,
    som_path: str | Path | None = None,
    q_table_path: str | Path | None = None,
    episodes: int | None = None,
    seed: int | None = None,
    headless: bool = False,
    resume: bool = True,
) -> Path:
    config.ensure_output_dirs()
    som = load_som_model(som_path, config)
    track_pairs = load_tracks(config.tracks_dir)
    if not track_pairs:
        raise FileNotFoundError("no tracks found in out/tracks; run draw_tracks.py first")

    tracks = [track for _, track in track_pairs]
    rng = np.random.default_rng(seed if seed is not None else config.simulation.random_seed)
    actions = ActionSpace.from_config(config)
    agent, start_episode = _load_or_create_agent(
        config,
        som.num_states,
        len(actions),
        q_table_path,
        resume,
    )
    episode_count = episodes if episodes is not None else config.rl.num_episodes
    target_episode = start_episode + episode_count
    view = None if headless else RlTrainingView(config, som.grid_dim)
    visible = True
    last_hidden_print: int | None = None

    last_path = None
    checkpoint_rewards: list[tuple[int, str, float]] = []
    total_reward = 0.0
    for episode in range(episode_count):
        global_episode = start_episode + episode + 1
        last_hidden_print = _print_hidden_episode(
            view,
            visible,
            global_episode,
            target_episode,
            last_hidden_print,
            total_reward,
        )
        track_index = episode % len(tracks)
        track = tracks[track_index]
        track_name = track_pairs[track_index][0].name
        env = DrivingEnv(config, track, rng)
        result = env.reset()
        state = som.state_from_lidar(result.lidar)

        total_reward = 0.0
        stuck_steps = 0

        while not result.done:
            action_index = agent.select_action(state, rng)
            action = actions[action_index]
            previous_index = nearest_centerline_index(
                env.track,
                (env.car.state.x, env.car.state.y),
            )
            result = env.step(action)
            current_index = nearest_centerline_index(
                env.track,
                (env.car.state.x, env.car.state.y),
            )
            if current_index > previous_index:
                stuck_steps = 0
            else:
                stuck_steps += 1
            stuck_penalty = config.rl.num_stuck > 0 and stuck_steps % config.rl.num_stuck == 0
            next_state = som.state_from_lidar(result.lidar)
            reward = compute_reward(
                config,
                result,
                track=env.track,
                previous_position=env.previous_position,
                current_position=(env.car.state.x, env.car.state.y),
                stuck_penalty=stuck_penalty,
                action=action,
            )
            total_reward += reward
            agent.update(state, action_index, reward, next_state, result.done)

            if view is not None:
                visible = view.process_events(visible)
                last_hidden_print = _print_hidden_episode(
                    view,
                    visible,
                    global_episode,
                    target_episode,
                    last_hidden_print,
                    total_reward,
                )
                view.draw(
                    env,
                    som,
                    next_state,
                    action,
                    agent.q_table,
                    global_episode,
                    target_episode,
                    agent,
                    total_reward,
                    stuck_steps,
                    visible,
                )
            state = next_state

        agent.decay_epsilon()
        agent.increment_beta()
        checkpoint_rewards.append((global_episode, track_name, total_reward))
        if global_episode % config.rl.checkpoint_episodes == 0:
            last_path = save_q_table(
                config.rl_dir,
                agent,
                actions,
                config,
                som.source_path,
                global_episode,
                checkpoint=True,
                checkpoint_rewards=checkpoint_rewards,
            )
            checkpoint_rewards = []

    if view is not None:
        view.close()

    return save_q_table(
        config.rl_dir,
        agent,
        actions,
        config,
        som.source_path,
        target_episode,
        checkpoint=False,
        checkpoint_rewards=checkpoint_rewards,
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
        total_reward: float,
        stuck_steps: int,
        visible: bool,
    ) -> None:
        if not visible:
            return
        self.renderer.draw(
            env.track,
            car=env.car,
            lidar=env.last_lidar,
            som_state=state,
            som_dim=som.grid_dim,
            som_weights=som.weights,
            som_weight_state=state,
            action=action,
            q_table=q_table,
            lines=[
                f"episode {episode}/{total_episodes}",
                f"step {env.steps}",
                f"episode reward {total_reward:.2f}",
                f"stuck steps {stuck_steps}",
                f"gamma {agent.gamma:.2f} alpha {agent.alpha:.2f}",
                f"epsilon {agent.epsilon:.4f}",
                f"beta {agent.beta:.4f} inc {agent.beta_increment:.4f}",
                *_lidar_display_lines(som, env.last_lidar),
                "G hides this view",
            ],
        )
        self.renderer.tick()

    def close(self) -> None:
        self.renderer.quit()


def _lidar_display_lines(
    som: SomDiscretizer,
    reading,
) -> list[str]:
    if reading is None:
        return []
    d_c, d_rl = som.display_values_from_lidar(reading)
    return [f"d_c {d_c:.3f}", f"d_rl {d_rl:.3f}"]


def _load_or_create_agent(
    config: AppConfig,
    num_states: int,
    num_actions: int,
    q_table_path: str | Path | None,
    resume: bool,
) -> tuple[QAgent, int]:
    if not resume and q_table_path is None:
        print("starting Q-learning from a fresh Q-table")
        return QAgent.fresh(num_states, num_actions, config), 0

    try:
        bundle = load_q_table(q_table_path, config)
    except FileNotFoundError:
        if q_table_path is not None:
            raise
        print("starting Q-learning from a fresh Q-table")
        return QAgent.fresh(num_states, num_actions, config), 0

    expected_shape = (num_states, num_actions)
    if bundle.q_table.shape != expected_shape:
        print(
            "ignored resume Q-table with shape "
            f"{bundle.q_table.shape}; expected {expected_shape}"
        )
        return QAgent.fresh(num_states, num_actions, config), 0

    print(
        f"resuming Q-learning from {bundle.source_path} at episode {bundle.episode}; "
        f"epsilon={config.rl.resume_epsilon}, beta={config.rl.resume_beta}"
    )
    return QAgent.resume(bundle.q_table, config), bundle.episode


def _print_hidden_episode(
    view: RlTrainingView | None,
    visible: bool,
    episode: int,
    target_episode: int,
    last_printed_episode: int | None,
    total_reward: float,
) -> int | None:
    if view is not None and visible:
        return last_printed_episode
    if last_printed_episode != episode:
        print(f"episode {episode}/{target_episode}, prev total reward: {total_reward:.2f}")
        return episode
    return last_printed_episode
