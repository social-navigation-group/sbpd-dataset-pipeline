"""Checks against a real copy of the dataset (or the sample).

Skipped unless ``ACME_ROOT`` points to a dataset root, e.g.::

    ACME_ROOT="/path/to/ACME-dataset" pytest tests/test_sample_data.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

from acme_dataset import find_trajectory_dirs, Trajectory
from acme_dataset.rosbag import find_bags, metadata_summary

ROOT = os.environ.get("ACME_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set ACME_ROOT to run against real data")


def test_human_readable_consistency():
    dirs = find_trajectory_dirs(ROOT)
    assert dirs, "no trajectories found"
    for path in dirs[:20]:
        traj = Trajectory(path)
        n = len(traj)
        assert len(traj.odometry) == n
        assert traj.detections["frame_idx"].max() < n
        assert np.all(np.diff(traj.timestamps) > 0)
        assert traj.image(0) is not None and traj.pointcloud(0) is not None
        # boxes must cover the person masks (catches box-center convention errors)
        covered = total = 0
        for i in range(n):
            mask = traj.mask(i)
            if mask is None:
                break
            boxes = np.zeros_like(mask)
            for x, y, w, h in traj.detections_at(i)[["x", "y", "width", "height"]].to_numpy():
                boxes[max(int(y), 0):max(int(y + h), 0), max(int(x), 0):max(int(x + w), 0)] = True
            covered, total = covered + (mask & boxes).sum(), total + mask.sum()
        if total:
            assert covered / total > 0.6, f"{path}: boxes cover only {covered / total:.0%} of person masks"


def test_raw_bag_metadata():
    for bag in find_bags(Path(ROOT) / "ros2 bags")[:5]:
        duration, topics = metadata_summary(bag)
        assert duration > 0 and len(topics) > 0
