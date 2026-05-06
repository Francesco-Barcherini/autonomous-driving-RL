import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.config.settings import load_config
from src.rl.actions import ActionSpace
from src.rl.q_learning import QAgent, save_q_table, stable_softmax


HAS_SHAPELY = importlib.util.find_spec("shapely") is not None


@unittest.skipUnless(HAS_SHAPELY, "shapely is required for q_learning imports")
class QLearningTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self) -> None:
        probabilities = stable_softmax(np.asarray([1000.0, 1001.0, 999.0]), beta=2.0)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertEqual(int(np.argmax(probabilities)), 1)

    def test_terminal_q_update_uses_immediate_reward(self) -> None:
        agent = QAgent.fresh(num_states=2, num_actions=3, config=load_config("config.toml"))
        agent.update(state=0, action=1, reward=5.0, next_state=1, terminal=True)
        self.assertAlmostEqual(agent.q_table[0, 1], 0.5)

    def test_checkpoint_saves_recent_episode_rewards(self) -> None:
        config = load_config("config.toml")
        actions = ActionSpace.from_config(config)
        agent = QAgent.fresh(num_states=2, num_actions=len(actions), config=config)
        with tempfile.TemporaryDirectory() as directory:
            path = save_q_table(
                directory=Path(directory),
                agent=agent,
                actions=actions,
                config=config,
                som_path=None,
                episode=2,
                checkpoint=True,
                checkpoint_rewards=[(1, "track_a.json", 3.5), (2, "track_b.json", -1.0)],
            )
            with np.load(path, allow_pickle=False) as data:
                episodes = np.asarray(data["checkpoint_reward_episodes"])
                tracks = np.asarray(data["checkpoint_reward_tracks"])
                rewards = np.asarray(data["checkpoint_reward_values"])
        np.testing.assert_array_equal(episodes, np.asarray([1, 2]))
        np.testing.assert_array_equal(tracks, np.asarray(["track_a.json", "track_b.json"]))
        np.testing.assert_allclose(rewards, np.asarray([3.5, -1.0]))


if __name__ == "__main__":
    unittest.main()
