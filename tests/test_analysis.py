import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.analysis.evaluation import (
    METRIC_FIELDS,
    distance_percentage,
    is_unsafe_step,
    latest_hard_qtable,
    safety_margin,
    step_unsafeness,
    track_circuit_distance,
    write_metrics_csv,
)
from src.analysis.plots import (
    _distance_rows_by_model_track,
    _histogram_scale,
    checkpoint_dataset_marker,
    moving_average,
    plot_distance_percentage_by_track,
    reward_curve_for_folder,
    weighted_time,
)
from src.config.settings import load_config
from src.environment.car import Car, CarState
from src.environment.lidar import LidarReading, RayHit
from src.environment.track import Obstacle, Track


class AnalysisTests(unittest.TestCase):
    def test_distance_percentage_uses_centerline_length(self) -> None:
        track = Track(
            "distance",
            100.0,
            [(0.0, 0.0), (30.0, 40.0), (60.0, 40.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(60.0, -10.0), (60.0, 90.0)],
            [],
        )
        circuit_distance = track_circuit_distance(track)
        self.assertAlmostEqual(circuit_distance, 80.0)
        self.assertAlmostEqual(distance_percentage(40.0, circuit_distance), 0.5)

    def test_safety_margin_uses_car_polygon_to_border_and_obstacles(self) -> None:
        track = Track(
            "margin",
            100.0,
            [(0.0, 0.0), (300.0, 0.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(300.0, -50.0), (300.0, 50.0)],
            [Obstacle(150.0, 0.0, 10.0)],
        )
        car = Car(20.0, 10.0, 100.0, 0.0, CarState(100.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(safety_margin(track, car), 30.0)
        car.reset(CarState(150.0, 0.0, 0.0, 0.0))
        self.assertEqual(safety_margin(track, car), 0.0)

    def test_safety_margin_ignores_finish_line_contact(self) -> None:
        track = Track(
            "finish_margin",
            100.0,
            [(0.0, 0.0), (300.0, 0.0)],
            [(0.0, -50.0), (0.0, 50.0)],
            [(300.0, -50.0), (300.0, 50.0)],
            [],
        )
        car = Car(20.0, 10.0, 100.0, 0.0, CarState(300.0, 0.0, 0.0, 0.0))
        self.assertGreater(safety_margin(track, car), 0.0)

    def test_unsafe_step_for_three_and_five_lidar(self) -> None:
        config = load_config("config.toml")
        config.rl.dc_threshold_unsafe = 0.5
        config.rl.drl_threshold_unsafe = 0.5
        safe = LidarReading(center=_ray(100.0), left=_ray(100.0), right=_ray(100.0))
        unsafe_front = LidarReading(center=_ray(10.0), left=_ray(100.0), right=_ray(100.0))
        unsafe_middle = LidarReading(
            center=_ray(100.0),
            left=_ray(100.0),
            right=_ray(100.0),
            middle_left=_ray(100.0),
            middle_right=_ray(0.0),
        )
        self.assertFalse(is_unsafe_step(config, safe))
        self.assertTrue(is_unsafe_step(config, unsafe_front))
        self.assertTrue(is_unsafe_step(config, unsafe_middle))
        self.assertAlmostEqual(step_unsafeness(config, safe), 0.0)
        self.assertAlmostEqual(step_unsafeness(config, unsafe_front), 0.9333333333333333)
        self.assertAlmostEqual(step_unsafeness(config, unsafe_middle), 2.0 / 3.0)

    def test_weighted_time_uses_distance_percentage(self) -> None:
        self.assertAlmostEqual(weighted_time({"time": "100", "distance_percentage": "0.25"}), 25.0)

    def test_latest_hard_qtable_selects_highest_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "qtable_checkpoint_000100_hard.npz").touch()
            selected = folder / "qtable_checkpoint_000200_hard.npz"
            selected.touch()
            (folder / "qtable_checkpoint_000300_easy.npz").touch()
            self.assertEqual(latest_hard_qtable(folder), selected)

    def test_reward_curve_uses_track_window_moving_average(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _save_reward_checkpoint(
                folder / "qtable_checkpoint_000002_hard.npz",
                [1, 2],
                ["a", "b"],
                [10.0, 20.0],
            )
            _save_reward_checkpoint(
                folder / "qtable_checkpoint_000004_hard.npz",
                [3, 4],
                ["a", "b"],
                [30.0, 50.0],
            )
            episodes, rewards = reward_curve_for_folder(folder)
        np.testing.assert_array_equal(episodes, np.asarray([2, 3, 4]))
        np.testing.assert_allclose(rewards, np.asarray([15.0, 25.0, 40.0]))
        avg_episodes, avg_values = moving_average(
            np.asarray([1, 2, 3]),
            np.asarray([3.0, 6.0, 12.0]),
            2,
        )
        np.testing.assert_array_equal(avg_episodes, np.asarray([2, 3]))
        np.testing.assert_allclose(avg_values, np.asarray([4.5, 9.0]))

    def test_checkpoint_dataset_marker_detects_suffix(self) -> None:
        self.assertIsNone(checkpoint_dataset_marker("qtable_checkpoint_000100_20260507_120000_123456.npz"))
        self.assertEqual(
            checkpoint_dataset_marker("qtable_checkpoint_000100_20260507_120000_123456_non_obstacles.npz"),
            "non_obstacles",
        )
        self.assertEqual(checkpoint_dataset_marker("qtable_checkpoint_000100_hard.npz"), "hard")

    def test_metrics_csv_writer_includes_required_columns(self) -> None:
        config = load_config("config.toml")
        row = {field: "" for field in METRIC_FIELDS}
        row["model_label"] = "model"
        row["track_name"] = "track.json"
        with tempfile.TemporaryDirectory() as directory:
            path = write_metrics_csv([row], Path(directory) / "metrics.csv", config)
            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["model_label"], "model")
        self.assertEqual(set(METRIC_FIELDS), set(rows[0].keys()))
        self.assertIn("unsafe_value", rows[0])

    def test_distance_percentage_plot_groups_tracks_and_success(self) -> None:
        rows = [
            {
                "model_label": "m1",
                "track_name": "a.json",
                "distance_percentage": "0.2",
                "success": "no",
                "model_path": "qtable_checkpoint_000010_easy.npz",
            },
            {
                "model_label": "m1",
                "track_name": "a.json",
                "distance_percentage": "1.0",
                "success": "yes",
                "model_path": "qtable_checkpoint_000020_hard.npz",
            },
            {"model_label": "m1", "track_name": "b.json", "distance_percentage": "0.4", "success": "no"},
            {"model_label": "m2", "track_name": "a.json", "distance_percentage": "0.8", "success": "no"},
        ]
        collapsed = _distance_rows_by_model_track(rows)
        self.assertEqual(len(collapsed), 3)
        self.assertEqual(
            [row for row in collapsed if row["model_label"] == "m1" and row["track_name"] == "a.json"][0]["distance_percentage"],
            "1.0",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = plot_distance_percentage_by_track(rows, Path(directory) / "distance.png")
            self.assertTrue(output.exists())

    def test_histogram_scale_uses_global_bins_and_limits(self) -> None:
        bin_edges, x_limits, y_limit = _histogram_scale({"a": [0.0, 1.0], "b": [10.0]}, 5)
        self.assertIsNotNone(bin_edges)
        self.assertEqual(len(bin_edges), 6)
        self.assertEqual(x_limits, (0.0, 10.0))
        self.assertGreaterEqual(y_limit, 1.0)


def _ray(distance: float) -> RayHit:
    return RayHit(
        angle=0.0,
        distance=distance,
        start=(0.0, 0.0),
        end=(distance, 0.0),
        hit=(distance, 0.0),
    )


def _save_reward_checkpoint(
    path: Path,
    episodes: list[int],
    tracks: list[str],
    rewards: list[float],
) -> None:
    np.savez(
        path,
        checkpoint_reward_episodes=np.asarray(episodes),
        checkpoint_reward_tracks=np.asarray(tracks),
        checkpoint_reward_values=np.asarray(rewards),
    )


if __name__ == "__main__":
    unittest.main()
