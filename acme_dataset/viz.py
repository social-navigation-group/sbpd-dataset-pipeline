"""Plotting helpers for quick looks at ACME data."""

import zlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .constants import BEV_IGNORE_TAG, BEV_TAG_COLORS


def track_color(track_id) -> Tuple[int, int, int]:
    """Stable RGB color for a track id."""
    idx = zlib.crc32(str(track_id).encode()) % 997 * 3
    return (37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255


def draw_detections(image: np.ndarray, detections: pd.DataFrame, thickness: int = 2,
                    show_ids: bool = True) -> np.ndarray:
    """Copy of ``image`` with pedestrian boxes (``x, y, width, height, track_id`` rows) drawn on."""
    import cv2

    out = np.ascontiguousarray(image.copy())
    for row in detections.itertuples(index=False):
        x0, y0 = int(row.x), int(row.y)
        x1, y1 = int(row.x + row.width), int(row.y + row.height)
        color = track_color(row.track_id)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        if show_ids:
            cv2.putText(out, str(row.track_id), (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """Blend a boolean mask over an RGB image."""
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def odometry_in_start_frame(odometry: pd.DataFrame) -> np.ndarray:
    """``(N, 2)`` robot positions expressed in the frame of the first pose."""
    xy = odometry[["position_x", "position_y"]].to_numpy()
    yaw0 = odometry["yaw"].iloc[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return (xy - xy[0]) @ np.array([[c, -s], [s, c]]).T


def plot_odometry(odometry: pd.DataFrame, axes=None):
    """Path (relative to the start pose), speed and yaw rate over time."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    xy = odometry_in_start_frame(odometry)
    t = odometry["timestamp"].to_numpy() - odometry["timestamp"].iloc[0]
    axes[0].plot(xy[:, 0], xy[:, 1], "-", lw=2)
    axes[0].plot(*xy[0], "go", label="start")
    axes[0].plot(*xy[-1], "rs", label="end")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Path in start frame")
    axes[0].axis("equal")
    axes[0].legend()
    axes[1].plot(t, odometry["linear_vel_x"], label="v_x")
    axes[1].plot(t, odometry["linear_vel_y"], label="v_y")
    axes[1].set(xlabel="time (s)", ylabel="m/s", title="Linear velocity")
    axes[1].legend()
    axes[2].plot(t, odometry["angular_vel"])
    axes[2].set(xlabel="time (s)", ylabel="rad/s", title="Angular velocity")
    for ax in axes:
        ax.grid(alpha=0.3)
    return axes


def pointcloud_to_bev(points: np.ndarray, x_range=(-15.0, 15.0), y_range=(-15.0, 15.0),
                      z_range=(-2.0, 3.0), resolution: float = 0.05):
    """Max-height top-down raster of a point cloud (x forward, y left).

    Returns ``(height_map, extent)`` for ``plt.imshow(height_map, extent=extent)``;
    empty cells are NaN.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = ((x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
            & (z >= z_range[0]) & (z < z_range[1]))
    x, y, z = x[keep], y[keep], z[keep]
    width = int(round((y_range[1] - y_range[0]) / resolution))
    height = int(round((x_range[1] - x_range[0]) / resolution))
    row = np.clip(((x_range[1] - x) / resolution).astype(int), 0, height - 1)
    col = np.clip(((y_range[1] - y) / resolution).astype(int), 0, width - 1)
    bev = np.full((height, width), -np.inf)
    np.maximum.at(bev, (row, col), z)
    bev[np.isinf(bev)] = np.nan
    return bev, [y_range[1], y_range[0], x_range[0], x_range[1]]


def plot_pointcloud_bev(points: np.ndarray, ax=None, **kwargs):
    """Show :func:`pointcloud_to_bev` with the robot at the origin."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    bev, extent = pointcloud_to_bev(points, **kwargs)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("black")
    im = ax.imshow(bev, cmap=cmap, extent=extent, origin="upper", interpolation="none")
    ax.plot(0, 0, "w^", ms=8)
    ax.set(xlabel="y, left (m)", ylabel="x, forward (m)")
    return im


def plot_bev_session(session, trajectories: Optional[pd.DataFrame] = None, ax=None,
                     show_tags: bool = True, max_tracks: Optional[int] = None):
    """Draw tag regions and pixel trajectories of a :class:`~acme_dataset.bev.BEVSession`."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))
    frame = session.frame()
    if frame is not None:
        ax.imshow(frame)
    if show_tags:
        for region in session.annotations():
            if region["tag_id"] == BEV_IGNORE_TAG:
                continue
            color = BEV_TAG_COLORS.get(region["tag_id"], "#ffffff")
            ax.add_patch(Polygon(region["points"], closed=True, facecolor=color, alpha=0.25,
                                 edgecolor=color, lw=2, label=region["tag_name"]))
    df = session.trajectories() if trajectories is None else trajectories
    cmap = plt.get_cmap("tab20")
    for i, (_, track) in enumerate(df.groupby("ped_id", sort=False)):
        if max_tracks is not None and i >= max_tracks:
            break
        ax.plot(track["px"], track["py"], "-", color=cmap(i % 20), lw=1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    ax.set_title(f"{session.dataset} / {session.session}")
    ax.axis("off")
    return ax
