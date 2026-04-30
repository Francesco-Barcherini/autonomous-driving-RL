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
        car.step(acceleration=10.0, steering_acceleration_deg=0.0, dt=1.0)
        self.assertAlmostEqual(car.state.velocity, 10.0)
        self.assertAlmostEqual(car.state.x, 10.0)
        self.assertAlmostEqual(car.state.y, 0.0)

    def test_angular_acceleration_changes_angular_velocity(self) -> None:
        car = Car(30.0, 10.0, 100.0, -20.0, CarState(0.0, 0.0, 0.0, 0.0))
        car.step(acceleration=0.0, steering_acceleration_deg=90.0, dt=1.0)
        self.assertAlmostEqual(car.state.angular_velocity, 1.57079632679)
        self.assertAlmostEqual(car.state.heading, 1.57079632679)

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

    def test_default_start_pose_uses_first_centerline_segment(self) -> None:
        track = Track(
            "start",
            80.0,
            [(10.0, 20.0), (110.0, 20.0), (160.0, 70.0)],
            [(10.0, -20.0), (10.0, 60.0)],
            [(160.0, 30.0), (160.0, 110.0)],
            [],
        )
        x, y, heading = track.start_pose()
        self.assertEqual((x, y), (10.0, 20.0))
        self.assertAlmostEqual(heading, 0.0)

    def test_start_line_intersection_prevents_initial_off_track_failure(self) -> None:
        config = load_config("config.toml")
        track = Track(
            "start_exception",
            config.track.track_width,
            [(100.0, 100.0), (300.0, 100.0)],
            [(100.0, 50.0), (100.0, 150.0)],
            [(300.0, 50.0), (300.0, 150.0)],
            [],
        )
        env = DrivingEnv(config, track)
        result = env.reset()
        self.assertFalse(track.contains_polygon(env.car.polygon()))
        self.assertTrue(env.car.polygon().intersects(track.start_geometry))
        result = env.step((0.0, 0.0))
        self.assertEqual(result.reason, "running")

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


if __name__ == "__main__":
    unittest.main()
