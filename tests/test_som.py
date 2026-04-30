import importlib.util
import unittest

import numpy as np

from src.config.settings import load_config
from src.kohonen.som import discretizer_from_config


HAS_SHAPELY = importlib.util.find_spec("shapely") is not None


@unittest.skipUnless(HAS_SHAPELY, "shapely is required for SOM imports")
class SomTests(unittest.TestCase):
    def test_normalization_and_state_range(self) -> None:
        config = load_config("config.toml")
        weights = np.ones((config.som.dim_grid_neurons, config.som.dim_grid_neurons, 2))
        model = discretizer_from_config(config, weights)
        vector = model.normalize_features((config.som.max_d_c, config.som.max_difference_r_l))
        self.assertLessEqual(float(np.linalg.norm(vector)), 1.0)
        state = model.state_from_features((10.0, 0.0))
        self.assertGreaterEqual(state, 0)
        self.assertLess(state, model.num_states)


if __name__ == "__main__":
    unittest.main()
