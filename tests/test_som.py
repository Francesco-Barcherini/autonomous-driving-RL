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
        vector = model.normalize_features((config.lidar.max_distance, config.lidar.max_distance))
        np.testing.assert_allclose(vector, np.asarray([1.0, 0.5]))
        state = model.state_from_features((10.0, 0.0))
        self.assertGreaterEqual(state, 0)
        self.assertLess(state, model.num_states)

    def test_five_lidar_normalization_uses_three_features(self) -> None:
        config = load_config("config.toml")
        weights = np.ones((config.som.dim_grid_neurons, config.som.dim_grid_neurons, 3))
        model = discretizer_from_config(config, weights)
        vector = model.normalize_features(
            (
                config.lidar.max_distance,
                config.lidar.max_distance,
                -config.lidar.max_distance,
            )
        )
        np.testing.assert_allclose(vector, np.asarray([1.0, 0.5, -0.5]))
        self.assertEqual(model.lidar_num_rays, 5)


if __name__ == "__main__":
    unittest.main()
