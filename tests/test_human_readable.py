import numpy as np
import pytest

from acme_dataset import Trajectory, dataset_stats, find_trajectory_dirs, list_trajectories, summarize
from acme_dataset.pointcloud import read_pcd, write_pcd


def test_trajectory_tables(trajectory_dir):
    traj = Trajectory(trajectory_dir)
    assert len(traj) == 5
    assert traj.institution == "go2nus"
    assert list(traj.odometry.columns[:2]) == ["frame_idx", "timestamp"]
    assert traj.odometry["timestamp"].iloc[1] == pytest.approx(traj.timestamps[1])
    det = traj.detections
    assert len(det) == 7 and set(det["track_id"]) == {"7", "9"}
    assert det.loc[det["frame_idx"] == 3, "x"].tolist() == [10.0, 33.0]
    assert traj.speech["text"].tolist() == ["excuse me"]


def test_trajectory_sensors(trajectory_dir):
    traj = Trajectory(trajectory_dir)
    assert traj.cameras == ["imgs"] and traj.mask_folders == ["masked_imgs"]
    assert traj.image(2).shape == (48, 64, 3)
    mask = traj.mask(0)
    assert mask.dtype == bool and mask[10, 12] and not mask[40, 50]
    assert traj.pointcloud(4).shape == (10, 4) and traj.pointcloud(4)[0, 0] == 4
    frame = traj.frame(1)
    assert frame.timestamp == pytest.approx(traj.timestamps[1]) and len(frame.detections) == 2
    with pytest.raises(IndexError):
        traj.frame(5)
    assert traj.summary()["unique_pedestrians"] == 2


def test_falls_back_to_pcd_files(trajectory_dir):
    for npz in (trajectory_dir / "pcd").glob("*.npz"):
        points = np.load(npz)["arr_0"]
        write_pcd(points, npz.with_suffix(".pcd"))
        npz.unlink()
    assert Trajectory(trajectory_dir).pointcloud(3)[0, 0] == 3


@pytest.mark.parametrize("binary", [False, True])
def test_pcd_roundtrip(tmp_path, binary):
    points = np.random.default_rng(0).normal(size=(100, 4)).astype(np.float32)
    path = write_pcd(points, tmp_path / "x.pcd", binary=binary)
    np.testing.assert_allclose(read_pcd(path), points, rtol=1e-6)


def test_discovery(trajectory_dir):
    root = trajectory_dir.parents[2]
    assert find_trajectory_dirs(root) == [trajectory_dir]
    assert find_trajectory_dirs(trajectory_dir) == [trajectory_dir]
    assert find_trajectory_dirs(root, institution="go2nus") == [trajectory_dir]
    assert len(list_trajectories(root / "human_readable_sampled")) == 1


def test_stats(trajectory_dir):
    stats = dataset_stats(trajectory_dir.parents[2], masks=True)
    row = stats.iloc[0]
    assert row["distance_m"] == pytest.approx(2.0)
    assert row["duration_s"] == pytest.approx(1.0)
    assert row["mean_speed_mps"] == pytest.approx(2.0)
    assert row["max_pedestrians_per_frame"] == 2
    assert 0 < row["mean_person_pixel_fraction"] < 1
    summary = summarize(stats)
    assert summary.loc["total", "trajectories"] == 1
