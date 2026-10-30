"""Per-trajectory and per-institution statistics of the human-readable data."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .human_readable import Trajectory, find_trajectory_dirs

PathLike = Union[str, Path]


def trajectory_stats(traj: Union[Trajectory, PathLike], masks: bool = False) -> dict:
    """Statistics of one trajectory.

    Robot motion comes from odometry, crowd measures from the pedestrian
    boxes. With ``masks=True`` the mean fraction of the image covered by
    people is computed as well (reads every mask, slower).
    """
    traj = traj if isinstance(traj, Trajectory) else Trajectory(traj)
    odom = traj.odometry
    positions = odom[["position_x", "position_y"]].to_numpy()
    speed = np.hypot(odom["linear_vel_x"].to_numpy(), odom["linear_vel_y"].to_numpy())
    peds_per_frame = traj.detections.groupby("frame_idx").size().reindex(range(len(traj)), fill_value=0)
    stats = {
        "institution": traj.institution,
        "trajectory": traj.name,
        "frames": len(traj),
        "duration_s": float(traj.timestamps[-1] - traj.timestamps[0]) if len(traj) > 1 else 0.0,
        "distance_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0,
        "mean_speed_mps": float(speed.mean()) if len(speed) else np.nan,
        "max_speed_mps": float(speed.max()) if len(speed) else np.nan,
        "mean_abs_angular_vel": float(odom["angular_vel"].abs().mean()) if len(odom) else np.nan,
        "unique_pedestrians": int(traj.detections["track_id"].nunique()),
        "mean_pedestrians_per_frame": float(peds_per_frame.mean()) if len(peds_per_frame) else 0.0,
        "max_pedestrians_per_frame": int(peds_per_frame.max()) if len(peds_per_frame) else 0,
        "frames_with_pedestrians": float((peds_per_frame > 0).mean()) if len(peds_per_frame) else 0.0,
        "speech_events": len(traj.speech),
    }
    if masks:
        coverage = [m.mean() for m in (traj.mask(i) for i in range(len(traj))) if m is not None]
        stats["mean_person_pixel_fraction"] = float(np.mean(coverage)) if coverage else np.nan
    return stats


def _safe_stats(args):
    path, masks = args
    try:
        return trajectory_stats(path, masks)
    except Exception as e:  # keep going on a broken trajectory
        return {"institution": Path(path).parent.name, "trajectory": Path(path).name, "error": repr(e)}


def dataset_stats(root: PathLike, institution: Optional[str] = None, masks: bool = False,
                  workers: int = 1) -> pd.DataFrame:
    """:func:`trajectory_stats` for every trajectory below ``root``."""
    jobs = [(p, masks) for p in find_trajectory_dirs(root, institution)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_safe_stats, jobs, chunksize=4))
    else:
        rows = [_safe_stats(job) for job in jobs]
    return pd.DataFrame(rows)


def summarize(stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`dataset_stats` per institution (plus a total row)."""
    ok = stats[stats["error"].isna()] if "error" in stats else stats
    if ok.empty:
        return pd.DataFrame()

    def agg(df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "trajectories": len(df),
            "hours": df["duration_s"].sum() / 3600.0,
            "distance_km": df["distance_m"].sum() / 1000.0,
            "mean_trajectory_s": df["duration_s"].mean(),
            "mean_speed_mps": np.average(df["mean_speed_mps"], weights=df["frames"]) if df["frames"].sum() else np.nan,
            "mean_pedestrians_per_frame": np.average(df["mean_pedestrians_per_frame"], weights=df["frames"])
            if df["frames"].sum() else np.nan,
            "unique_pedestrians": df["unique_pedestrians"].sum(),
        })

    per_inst = pd.DataFrame({inst: agg(df) for inst, df in ok.groupby("institution")}).T
    per_inst.loc["total"] = agg(ok)
    per_inst.index.name = "institution"
    return per_inst
