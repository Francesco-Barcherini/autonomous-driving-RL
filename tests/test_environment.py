import importlib.util
import tempfile
from pathlib import Path
import unittest

from src.environment.car import Car, CarState
from src.environment.env import DrivingEnv
from src.environment.lidar import Lidar
from src.environment.track import Obstacle, Track, load_track
from src.config.settings import load_config


HAS_SHAPELY = importlib.util.find_spec("shapely") is not None


@unittest.skipUnless(HAS_SHAPELY, "shapely is required for environment tests")
class EnvironmentTests(unittest.TestCase):
    def test_car_euler_step(self) -> None:
        car = Car(30.0, 10.0, 100.0, -20.0, CarState(0.0, 0.0, 0.0, 0.0))
        car.step(acceleration=10.0, steering_diff_angle_deg=0.0, dt=1.0)
        self.assertAlmostEqual(car.state.velocity, 10.0)
        self.assertAlmostEqual(car.state.x, 10.0)
        self.assertAlmostEqual(car.state.y, 0.0)

    def test_steering_diff_changes_heading_directly(self) -> None:
        car = Car(30.0, 10.0, 100.0, -20.0, CarState(0.0, 0.0, 0.0, 0.0))
        car.step(acceleration=0.0, steering_diff_angle_deg=2.0, dt=1.0)
        self.assertAlmostEqual(car.state.heading, 0.03490658504)

    def test_track_json_roundtrip(self) -> None:
        track = Track(
            "roundtrip",
            80.0,
            [(0.0, 0.0), (200.0, 0.0)],
            [(0.0, -40.0), (0.0, 40.0)],
            [(200.0, -40.0), (200.0, 40.0)],
            [Obstacle(100.0, 0.0, 10.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = track.save(Path(directory), timestamp=False)
            loaded = load_track(path)
        self.assertEqual(loaded.name, "roundtrip")
        self.assertEqual(len(loaded.obstacles), 1)
        self.assertTrue(loaded.is_valid())

    def test_default_start_pose_uses_first_fitting_centerline_point(self) -> None:
        config = load_config("config.toml")
        track = Track(
            "start",
            80.0,
            [(10.0, 20.0), (110.0, 20.0), (160.0, 70.0)],
            [(10.0, -20.0), (10.0, 60.0)],
            [(160.0, 30.0), (160.0, 110.0)],
            [],
        )
        x, y, heading = track.start_pose(
            car_length=config.car.length,
            car_width=config.car.width,
        )
        self.assertEqual((x, y), (110.0, 20.0))
        self.assertAlmostEqual(heading, 0.7853981633974483)

    def test_env_reset_starts_with_car_fully_inside_track(self) -> None:
        config = load_config("config.toml")
        track = Track(
            "inside_start",
            config.track.track_width,
            [(100.0, 100.0), (300.0, 100.0)],
            [(100.0, 50.0), (100.0, 150.0)],
            [(300.0, 50.0), (300.0, 150.0)],
            [],
        )
        env = DrivingEnv(config, track)
        result = env.reset()
        self.assertTrue(track.contains_polygon(env.car.polygon()))
        result = env.step((0.0, 0.0))
        self.assertEqual(result.reason, "running")

    def test_finish_success_uses_car_polygon_not_center_path(self) -> None:
        config = load_config("config.toml")
        track = Track(
            "finish_polygon",
            config.track.track_width,
            [(100.0, 100.0), (200.0, 100.0)],
            [(100.0, 50.0), (100.0, 150.0)],
            [(200.0, 50.0), (200.0, 150.0)],
            [],
        )
        env = DrivingEnv(config, track)
        env.reset()
        env.previous_position = (183.0, 100.0)
        env.car.reset(CarState(183.0, 100.0, 0.0, 0.0))
        self.assertTrue(env.car.polygon().intersects(track.finish_geometry))
        self.assertEqual(env._terminal_reason(), "success")

    def test_lidar_hits_nearest_obstacle(self) -> None:
        track = Track(
            "lidar",
            100.0,
            [(0.0, 0.0), (300.0, 0.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(300.0, -50.0), (300.0, 50.0)],
            [Obstacle(80.0, 0.0, 10.0)],
        )
        reading = Lidar(max_distance=250.0).scan(CarState(10.0, 0.0, 0.0), track)
        self.assertGreater(reading.center.distance, 55.0)
        self.assertLess(reading.center.distance, 65.0)

    def test_lidar_ignores_finish_line(self) -> None:
        track = Track(
            "finish_lidar",
            100.0,
            [(0.0, 0.0), (300.0, 0.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(300.0, -50.0), (300.0, 50.0)],
            [],
        )
        reading = Lidar(max_distance=250.0).scan(CarState(260.0, 0.0, 0.0), track)
        self.assertAlmostEqual(reading.center.distance, 250.0)

    def test_five_lidar_returns_middle_rays_and_three_features(self) -> None:
        track = Track(
            "five_lidar",
            100.0,
            [(0.0, 0.0), (300.0, 0.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(300.0, -50.0), (300.0, 50.0)],
            [],
        )
        reading = Lidar(max_distance=250.0, num_rays=5).scan(CarState(100.0, 0.0, 0.0), track)
        self.assertEqual(len(reading.rays), 5)
        self.assertIsNotNone(reading.middle_left)
        self.assertIsNotNone(reading.middle_right)
        self.assertEqual(len(reading.vector), 3)


if __name__ == "__main__":
    unittest.main()
