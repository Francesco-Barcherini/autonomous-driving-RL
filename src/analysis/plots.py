"""Plotting helpers for model analysis outputs."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from src.analysis.evaluation import _checkpoint_episode, model_label


def discover_plot_model_folders(root: str | Path = "out/rl/bk") -> list[Path]:
    root_path = Path(root)
    return sorted({path.parent for path in root_path.rglob("qtable*.npz")})


def latest_metrics_csv(directory: str | Path = "out/analysis") -> Path:
    paths = sorted(Path(directory).glob("metrics_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics_*.csv found in {directory}")
    return paths[-1]


def load_metrics_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reward_curve_for_folder(folder: str | Path) -> tuple[np.ndarray, np.ndarray]:
    episodes, rewards, _markers = reward_curve_with_markers(folder)
    return episodes, rewards


def reward_curve_with_markers(folder: str | Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float, str]]]:
    entries: list[tuple[int, str, float]] = []
    unique_tracks: set[str] = set()
    marker_labels: dict[int, str] = {}
    for path in sorted(Path(folder).glob("qtable*.npz"), key=_checkpoint_episode):
        marker = checkpoint_dataset_marker(path)
        checkpoint_episode = _checkpoint_episode(path)
        if marker is not None and checkpoint_episode >= 0:
            marker_labels[checkpoint_episode] = marker
        with np.load(path, allow_pickle=False) as data:
            if not all(
                key in data.files
                for key in (
                    "checkpoint_reward_episodes",
                    "checkpoint_reward_tracks",
                    "checkpoint_reward_values",
                )
            ):
                continue
            episodes = np.asarray(data["checkpoint_reward_episodes"], dtype=int)
            tracks = np.asarray(data["checkpoint_reward_tracks"], dtype=str)
            rewards = np.asarray(data["checkpoint_reward_values"], dtype=float)
            for episode, track, reward in zip(episodes, tracks, rewards):
                if track == 'track_20260506_231648_568004.json' or track == 'track_20260504_025438_813461.json':
                    continue  # Exclude this track due to anomalous reward values
                # exclude all qtables in 3-lidar-5deg-12som after the hard
                if path.parent.name == "3-lidar-5deg-12som" and episode >= 1800:
                    continue
                entries.append((int(episode), str(track), float(reward)))
                unique_tracks.add(str(track))
    if not entries:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), []
    entries.sort(key=lambda item: item[0])
    window = max(len(unique_tracks), 1)
    episodes = np.asarray([item[0] for item in entries], dtype=int)
    rewards = np.asarray([item[2] for item in entries], dtype=float)
    avg_episodes, avg_rewards = moving_average(episodes, rewards, window)
    markers = _markers_on_curve(avg_episodes, avg_rewards, marker_labels)
    return avg_episodes, avg_rewards, markers


def checkpoint_dataset_marker(path: str | Path) -> str | None:
    stem = Path(path).stem
    match = re.match(
        r"^qtable_checkpoint_\d+(?:_\d{8}_\d{6}_\d+)?(?:_(?P<marker>[A-Za-z].*))?$",
        stem,
    )
    if match:
        return match.group("marker")
    return None


def moving_average(
    episodes: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return episodes, values
    window = max(min(int(window), len(values)), 1)
    averaged = np.convolve(values, np.ones(window, dtype=float) / window, mode="valid")
    return episodes[window - 1 :], averaged


def save_all_plots(
    metrics_rows: list[dict[str, str]],
    model_folders: Iterable[str | Path],
    output_dir: str | Path,
    base_label: str | None = None,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_labels = [model_label(folder) for folder in model_folders]
    colors = _label_colors(sorted({*model_labels, *(row.get("model_label", "") for row in metrics_rows)}))
    paths = [
        plot_reward_by_episode(model_folders, output / "reward_by_episode.png", colors),
        plot_distance_percentage_by_track(metrics_rows, output / "distance_percentage_by_track.png", colors),
    ]
    paths.extend(plot_time_ratio_histograms(metrics_rows, output, base_label, colors))
    paths.extend(
        plot_metric_histograms_by_model(
            metrics_rows,
            output,
            "safety_margin",
            "Safety margin",
            "safety_margin_histogram",
            colors,
            bins=15,
        )
    )
    paths.extend(
        plot_metric_histograms_by_model(
            metrics_rows,
            output,
            "unsafe_value",
            "Unsafe value",
            "unsafe_value_histogram",
            colors,
            fallback_metric="unsafe_step_percentage",
            bins=30,
        )
    )
    return paths


def plot_reward_by_episode(
    model_folders: Iterable[str | Path],
    output_path: str | Path,
    colors: dict[str, object] | None = None,
) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 6))
    for folder in model_folders:
        episodes, rewards, markers = reward_curve_with_markers(folder)
        if len(episodes) == 0:
            continue
        label = model_label(folder)
        color = None if colors is None else colors.get(label)
        ax.plot(episodes, rewards, label=label, color=color)
        if markers:
            marker_x = [item[0] for item in markers]
            marker_y = [item[1] for item in markers]
            ax.scatter(marker_x, marker_y, marker="x", s=56, color=color, zorder=3)
            for episode, reward, marker_label in markers:
                ax.annotate(
                    marker_label,
                    (episode, reward),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=7,
                    rotation=25,
                )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_title("Training reward by episode")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return _save(fig, output_path)


def plot_time_ratio_histograms(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    base_label: str | None,
    colors: dict[str, object] | None = None,
) -> list[Path]:
    base_rows = _base_rows(rows, base_label)
    base_weighted_time_by_track = {
        row["track_name"]: weighted_time(row)
        for row in base_rows
        if weighted_time(row) > 0.0
    }
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row in base_rows:
            continue
        if _float(row.get("distance_percentage", "")) <= 0.75:
            continue
        base_weighted_time = base_weighted_time_by_track.get(row["track_name"], 0.0)
        model_weighted_time = weighted_time(row)
        if base_weighted_time <= 0.0:
            continue
        grouped.setdefault(row["model_label"], []).append(model_weighted_time / base_weighted_time)
    return _plot_histograms_by_model(
        grouped,
        output_dir,
        "time_ratio_histogram",
        "Time ratio",
        "(time * distance percentage) / base",
        colors,
        bins=80,
    )


def weighted_time(row: dict[str, str]) -> float:
    return _float(row.get("time", "")) / _float(row.get("distance_percentage", ""))


def plot_metric_histograms_by_model(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    metric: str,
    title: str,
    filename_prefix: str,
    colors: dict[str, object] | None = None,
    fallback_metric: str | None = None,
    bins: int = 150,
) -> list[Path]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _row_metric(row, metric, fallback_metric)
        if value is not None:
            grouped.setdefault(row["model_label"], []).append(value)
    return _plot_histograms_by_model(grouped, output_dir, filename_prefix, title, metric, colors, bins=bins)


def plot_distance_percentage_by_track(
    rows: list[dict[str, str]],
    output_path: str | Path,
    colors: dict[str, object] | None = None,
) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 6))
    collapsed_rows = _distance_rows_by_model_track(rows)
    tracks = sorted({row.get("track_name", "") for row in collapsed_rows if row.get("track_name", "")})
    x_by_track = {track: index for index, track in enumerate(tracks)}
    labels = sorted({row.get("model_label", "") for row in collapsed_rows if row.get("model_label", "")})
    offsets = _model_offsets(labels)
    for label_index, label in enumerate(labels):
        label_rows = [row for row in collapsed_rows if row.get("model_label") == label]
        color = None if colors is None else colors.get(label)
        success_count = sum(1 for row in label_rows if row.get("success") == "yes")
        total_count = len(label_rows)
        success_rate = 100.0 * success_count / total_count if total_count > 0 else 0.0
        legend_label = f"{label}"
        plot_points = []
        ax.scatter([], [], color=color, marker="o", label=legend_label, s=42)
        for row_index, row in enumerate(label_rows):
            track = row.get("track_name", "")
            if track not in x_by_track:
                continue
            marker = "" if row.get("success") == "yes" else "o"
            ax.scatter(
                x_by_track[track],# + offsets.get(label, 0.0),
                _float(row.get("distance_percentage", "")),
                color=color,
                marker=marker,
                #label=legend_label if row_index == 0 else None,
                s=42,
            )
            plot_points.append((x_by_track[track], _float(row.get("distance_percentage", ""))))
        #ax.plot([x + offsets.get(label, 0.0) for x, _ in plot_points], [y for _, y in plot_points], color=color)
        # plot the line from one point to the next only if both points are not successful
        for i in range(1, len(plot_points)):
            x1, y1 = plot_points[i - 1]
            x2, y2 = plot_points[i]
            if label_rows[i - 1].get("success") != "yes" and label_rows[i].get("success") != "yes":
                ax.plot([x1, x2], [y1, y2], color=color)
    ax.set_xticks(list(x_by_track.values()))
    ax.set_xticklabels([str(index) for index in range(len(tracks))])
    ax.set_xlabel("Track")
    ax.set_ylabel("Distance percentage")
    ax.set_title("Distance percentage by track")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, output_path)


def _plot_histograms_by_model(
    grouped: dict[str, list[float]],
    output_dir: str | Path,
    filename_prefix: str,
    title: str,
    xlabel: str,
    colors: dict[str, object] | None = None,
    bins: int = 150,
) -> list[Path]:
    plt = _pyplot()
    paths: list[Path] = []
    bin_edges, x_limits, y_limit = _histogram_scale(grouped, bins)
    if bin_edges is None:
        return paths
    for label, values in grouped.items():
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        color = None if colors is None else colors.get(label)
        ax.hist(finite_values, bins=bin_edges, alpha=0.65, label=label, color=color)
        ax.set_xlim(x_limits)
        ax.set_ylim((0.0, y_limit))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{title}: {label}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        paths.append(_save(fig, Path(output_dir) / f"{filename_prefix}_{_safe_filename(label)}.png"))
    return paths


def _distance_rows_by_model_track(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        label = row.get("model_label", "")
        track = row.get("track_name", "")
        if not label or not track:
            continue
        key = (label, track)
        current = by_key.get(key)
        if current is None or _is_hard_row(row) and not _is_hard_row(current):
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _is_hard_row(row: dict[str, str]) -> bool:
    return "hard" in Path(row.get("model_path", "")).name


def _model_offsets(labels: list[str]) -> dict[str, float]:
    if len(labels) <= 1:
        return {label: 0.0 for label in labels}
    spread = 0.65
    return {
        label: -spread / 2.0 + spread * index / (len(labels) - 1)
        for index, label in enumerate(labels)
    }


def _histogram_scale(
    grouped: dict[str, list[float]],
    bins: int,
) -> tuple[np.ndarray | None, tuple[float, float], float]:
    all_values = [
        value
        for values in grouped.values()
        for value in values
        if math.isfinite(value)
    ]
    if not all_values:
        return None, (0.0, 1.0), 1.0
    minimum = min(all_values)
    maximum = max(all_values)
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 0.5)
        minimum -= padding
        maximum += padding
    bin_edges = np.linspace(minimum, maximum, max(int(bins), 1) + 1)
    y_limit = 1.0
    for values in grouped.values():
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            continue
        counts, _ = np.histogram(finite_values, bins=bin_edges)
        y_limit = max(y_limit, float(np.max(counts)))
    return bin_edges, (float(minimum), float(maximum)), y_limit * 1.1


def _base_rows(rows: list[dict[str, str]], base_label: str | None) -> list[dict[str, str]]:
    base_rows = [row for row in rows if row.get("is_base") == "yes"]
    if base_rows:
        return base_rows
    if base_label is None:
        return []
    return [row for row in rows if row.get("model_label") == base_label]


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _row_metric(row: dict[str, str], metric: str, fallback_metric: str | None) -> float | None:
    if row.get(metric, "") != "":
        value = _float(row[metric])
    elif fallback_metric is not None:
        value = _float(row.get(fallback_metric, ""))
    else:
        return None
    return value if math.isfinite(value) else None


def _markers_on_curve(
    episodes: np.ndarray,
    rewards: np.ndarray,
    marker_labels: dict[int, str],
) -> list[tuple[int, float, str]]:
    markers: list[tuple[int, float, str]] = []
    if len(episodes) == 0:
        return markers
    for episode, label in sorted(marker_labels.items()):
        index = int(np.searchsorted(episodes, episode, side="right") - 1)
        if index < 0:
            index = 0
        markers.append((int(episodes[index]), float(rewards[index]), label))
    return markers


def _label_colors(labels: Iterable[str]) -> dict[str, object]:
    plt = _pyplot()
    cmap = plt.get_cmap("tab10")
    return {label: cmap(index % 10) for index, label in enumerate(label for label in labels if label)}


def _safe_filename(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe or "model"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    fig.clf()
    return path
