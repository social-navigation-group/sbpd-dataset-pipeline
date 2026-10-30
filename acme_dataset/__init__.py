"""Tools to download, load, explore and process the ACME social navigation dataset.

Quick start::

    from acme_dataset import download, list_trajectories

    download("human_readable", institution="go2nus", local_dir="ACME-dataset")
    for traj in list_trajectories("ACME-dataset"):
        print(traj, traj.odometry.shape, traj.detections.shape)
"""

from .bev import BEVIndex, BEVSession, list_bev_sessions
from .download import download, list_remote
from .human_readable import Trajectory, find_trajectory_dirs, list_trajectories
from .stats import dataset_stats, summarize, trajectory_stats
from .tags import filter_by_tags, load_scenario_tags, tag_counts, tags_for

__version__ = "0.1.0"

__all__ = [
    "BEVIndex", "BEVSession", "list_bev_sessions",
    "download", "list_remote",
    "Trajectory", "find_trajectory_dirs", "list_trajectories",
    "dataset_stats", "summarize", "trajectory_stats",
    "filter_by_tags", "load_scenario_tags", "tag_counts", "tags_for",
]
