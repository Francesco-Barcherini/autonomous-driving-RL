from pathlib import Path
import unittest

from src.config.settings import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_default_config(self) -> None:
        config = load_config(Path("config.toml"))
        self.assertEqual(config.som.dim_grid_neurons, 6)
        self.assertEqual(config.rl.num_episodes, 1000)
        self.assertTrue(str(config.tracks_dir).endswith("out/tracks"))


if __name__ == "__main__":
    unittest.main()
