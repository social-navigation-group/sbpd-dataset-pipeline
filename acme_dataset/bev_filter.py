"""Filtering of metric BEV pedestrian trajectories.

Port of ``trajectory-transform/trajectory_filter.py`` shipped with the BEV
annotations. Every function takes and returns ``(N, 2)`` arrays of metric
positions sampled at ``fps``. Each point gets one of three labels:

- ``"dynamic"``: valid, moving point
- ``"stationary"``: local speed (sliding window) below ``speed_threshold``
- ``"out_of_bounds"``: outside ``coordinate_bounds`` (unreliable homography region)
"""

from typing import Optional, Tuple

import numpy as np

LABEL_DYNAMIC = "dynamic"
LABEL_STATIONARY = "stationary"
LABEL_OUT_OF_BOUNDS = "out_of_bounds"

DEFAULT_TIME_THRESHOLD = 3.2           # minimum trajectory duration (s)
DEFAULT_SPEED_THRESHOLD = 0.5          # stationary below this local speed (m/s)
DEFAULT_TARGET_FPS = 2.5               # output rate of filter_and_downsample
DEFAULT_VELOCITY_JUMP_THRESHOLD = 5.0  # jumps above 2x this speed are interpolated (m/s)
DEFAULT_SMOOTHING_WINDOW = 5           # moving-average window (frames), 0 disables
DEFAULT_STATIONARY_WINDOW = 8          # stationary-detection window (frames)
DEFAULT_COORDINATE_BOUNDS = None       # e.g. (-25, 25) metres for x and y


def path_length(points: np.ndarray) -> float:
    """Total length of a polyline in metres."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def correct_velocity_jumps(points: np.ndarray, fps: float,
                           velocity_threshold: float = DEFAULT_VELOCITY_JUMP_THRESHOLD) -> Tuple[np.ndarray, int]:
    """Linearly interpolate isolated outliers (runs of at most 3 points whose
    step speed exceeds ``2 * velocity_threshold``). Longer runs and runs at the
    trajectory ends are left unchanged.

    Returns the corrected points and the number of corrected points.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n < 3:
        return points, 0
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
    is_bad = np.zeros(n, dtype=bool)
    jumps = np.nonzero(speed > 2 * velocity_threshold)[0]
    is_bad[jumps] = True
    is_bad[jumps + 1] = True

    corrections = 0
    i = 0
    while i < n:
        if not is_bad[i]:
            i += 1
            continue
        start = i
        while i < n and is_bad[i]:
            i += 1
        end = i
        if start > 0 and end < n and end - start <= 3:
            anchor_start, anchor_end = points[start - 1].copy(), points[end].copy()
            for j in range(start, end):
                t = (j - start + 1) / (end - start + 1)
                points[j] = anchor_start + t * (anchor_end - anchor_start)
                corrections += 1
    return points, corrections


def smooth_trajectory(points: np.ndarray, window_size: int = DEFAULT_SMOOTHING_WINDOW) -> np.ndarray:
    """Centred moving average; the window shrinks near the ends."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3 or window_size < 2:
        return points.copy()
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    csum = np.vstack([np.zeros((1, points.shape[1])), np.cumsum(points, axis=0)])
    idx = np.arange(n)
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, n)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def label_coordinate_bounds(points: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    """``"dynamic"`` inside ``bounds`` (applied to x and y), ``"out_of_bounds"`` outside."""
    points = np.asarray(points, dtype=float)
    lo, hi = bounds
    inside = ((points >= lo) & (points <= hi)).all(axis=1)
    return np.where(inside, LABEL_DYNAMIC, LABEL_OUT_OF_BOUNDS).astype(object)


def label_stationary_points(points: np.ndarray, fps: float,
                            speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
                            window_size: int = DEFAULT_STATIONARY_WINDOW,
                            labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Relabel ``"dynamic"`` points as ``"stationary"`` where the average speed
    over a centred window is below ``speed_threshold``."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    labels = np.full(n, LABEL_DYNAMIC, dtype=object) if labels is None else np.array(labels, dtype=object)
    if n < 2:
        return labels
    window_size = max(2, min(window_size, n))
    half = window_size // 2
    step = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    for i in np.nonzero(labels == LABEL_DYNAMIC)[0]:
        start, end = max(0, i - half), min(n, i + half + 1)
        if end - start < 2:
            continue
        speed = (step[end - 1] - step[start]) / ((end - start - 1) / fps)
        if speed < speed_threshold:
            labels[i] = LABEL_STATIONARY
    return labels


def _base_labels(points: np.ndarray, coordinate_bounds) -> np.ndarray:
    if coordinate_bounds is None:
        return np.full(len(points), LABEL_DYNAMIC, dtype=object)
    return label_coordinate_bounds(points, coordinate_bounds)


def filter_trajectory(
    points: np.ndarray,
    fps: float,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    velocity_jump_threshold: float = DEFAULT_VELOCITY_JUMP_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    stationary_window: int = DEFAULT_STATIONARY_WINDOW,
    coordinate_bounds: Optional[Tuple[float, float]] = DEFAULT_COORDINATE_BOUNDS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clean one trajectory at its original frame rate (no downsampling).

    Drops trajectories shorter than ``time_threshold`` seconds or without any
    dynamic point, corrects velocity jumps, smooths and labels each point.

    Returns ``(points, labels)`` or None if the trajectory is rejected.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < int(time_threshold * fps):
        return None
    points, _ = correct_velocity_jumps(points, fps, velocity_jump_threshold)
    if smoothing_window > 0:
        points = smooth_trajectory(points, smoothing_window)
    labels = label_stationary_points(points, fps, speed_threshold, stationary_window,
                                     _base_labels(points, coordinate_bounds))
    if not (labels == LABEL_DYNAMIC).any():
        return None
    return points, labels


def filter_and_downsample(
    points: np.ndarray,
    fps: float,
    target_fps: float = DEFAULT_TARGET_FPS,
    time_threshold: float = DEFAULT_TIME_THRESHOLD,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    velocity_jump_threshold: float = DEFAULT_VELOCITY_JUMP_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    stationary_window: int = DEFAULT_STATIONARY_WINDOW,
    coordinate_bounds: Optional[Tuple[float, float]] = DEFAULT_COORDINATE_BOUNDS,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Filter one trajectory and downsample it to ``target_fps``.

    Same pipeline as ``trajectory-transform/transform_trajectories.py``.

    Returns ``(points, labels, frame_offsets)`` where ``frame_offsets`` are the
    indices of the kept points in the original-rate trajectory, or None if the
    trajectory is rejected.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < int(time_threshold * fps):
        return None
    labels = _base_labels(points, coordinate_bounds)
    if (labels == LABEL_OUT_OF_BOUNDS).all():
        return None
    points, _ = correct_velocity_jumps(points, fps, velocity_jump_threshold)
    if smoothing_window > 0:
        points = smooth_trajectory(points, smoothing_window)
    labels = label_stationary_points(points, fps, speed_threshold, stationary_window, labels)
    if not (labels == LABEL_DYNAMIC).any():
        return None

    factor = max(1, int(fps / target_fps))
    offsets = np.arange(0, len(points), factor)
    points, _ = correct_velocity_jumps(points[::factor], target_fps, velocity_jump_threshold)
    if smoothing_window > 0:
        points = smooth_trajectory(points, smoothing_window)
    labels = label_stationary_points(points, target_fps, speed_threshold,
                                     max(2, stationary_window // factor),
                                     _base_labels(points, coordinate_bounds))
    if not (labels == LABEL_DYNAMIC).any() or len(points) < int(time_threshold * target_fps):
        return None
    return points, labels, offsets
