import unittest

from src.config.settings import load_config
from src.rl.actions import ActionSpace


class ActionSpaceTests(unittest.TestCase):
    def test_action_space_has_27_actions(self) -> None:
        config = load_config("config.toml")
        actions = ActionSpace.from_config(config)
        self.assertEqual(len(actions), 27)
        self.assertEqual(
            actions[0],
            (-config.car.max_acceleration, -config.car.max_steering_acceleration_deg),
        )
        self.assertEqual(actions[13], (0.0, 0.0))
        self.assertEqual(
            actions[-1],
            (config.car.max_acceleration, config.car.max_steering_acceleration_deg),
        )


if __name__ == "__main__":
    unittest.main()
