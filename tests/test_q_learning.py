import importlib.util
import unittest

import numpy as np

from src.config.settings import load_config
from src.rl.q_learning import QAgent, stable_softmax


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


if __name__ == "__main__":
    unittest.main()
