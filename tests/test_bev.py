import numpy as np
import pytest

from acme_dataset import BEVIndex, BEVSession, list_bev_sessions
from acme_dataset import bev_filter as bf
from acme_dataset.bev import apply_homography, distort_points, undistort_points


def test_index_and_lookup(bev_root):
    sessions = list_bev_sessions(bev_root)
    assert sessions[["dataset", "session"]].values.tolist() == [["toy", "s1"]]
    session = BEVSession(bev_root, "toy", "s1")
    assert session.stem == "toy_s1_transformed"
    assert session.frame().shape == (100, 200, 3)
    assert set(session.masks()) == {1, 3}
    assert session.tag_names()[3] == "Narrow Corridor"
    with pytest.raises(KeyError):
        BEVSession(bev_root, "toy", "missing")


def test_stem_variants(bev_root):
    index = BEVIndex(bev_root)
    index.frames["umich_2025_09_03_-_15:47_-_GX1_BEV1_transformed"] = None
    index.frames["short-term_completed_a_0_1_2025_05_27_-_x_transformed"] = None
    assert index.find_stem("umich", "GX1_BEV1") == "umich_2025_09_03_-_15:47_-_GX1_BEV1_transformed"
    assert index.find_stem("short-term", "completed_a_0_1") == "short-term_completed_a_0_1_2025_05_27_-_x_transformed"
    assert index.find_stem("toy", "s") is None


def test_trajectories(bev_root):
    session = BEVSession(bev_root, "toy", "s1")
    raw = session.trajectories(filtered=False)
    assert set(raw["ped_id"]) == {"human1", "human2", "human3"}
    walker = raw[raw["ped_id"] == "human1"]
    assert walker["frame"].iloc[0] == 3 and walker["x"].iloc[0] == pytest.approx(1.0)

    clean = session.trajectories()
    assert set(clean["ped_id"]) == {"human1"}  # stationary and short tracks dropped
    assert (clean["label"] == "dynamic").all()
    np.testing.assert_allclose(clean[["px", "py"]].to_numpy() / 10, clean[["x", "y"]].to_numpy(), atol=1e-9)

    down = session.trajectories(target_fps=2.5)
    assert np.diff(down["frame"]).tolist() == [4] * (len(down) - 1)


def test_tag_segments(bev_root):
    session = BEVSession(bev_root, "toy", "s1")
    segments = session.tag_segments()
    assert sorted(segments["tag_id"].unique()) == [1, 3]
    assert segments.groupby("tag_id")["segment"].nunique().tolist() == [1, 1]
    left = segments[segments["tag_id"] == 1]
    assert (left["px"] < 100).all()


def test_geometry_roundtrip():
    H = np.array([[0.02, 0.001, -3.0], [0.0005, 0.025, -1.0], [1e-5, 2e-5, 1.0]])
    pts = np.array([[100.0, 200.0], [640.0, 360.0]])
    np.testing.assert_allclose(apply_homography(apply_homography(pts, H), np.linalg.inv(H)), pts, atol=1e-6)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    D = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    # cv2.undistortPoints is iterative, so the round trip is approximate
    np.testing.assert_allclose(distort_points(undistort_points(pts, K, D), K, D), pts, atol=0.01)


def test_filter_primitives():
    line = np.stack([np.arange(50) * 0.1, np.zeros(50)], axis=1)
    jumped = line.copy()
    jumped[20] += [5.0, 5.0]
    fixed, n = bf.correct_velocity_jumps(jumped, fps=10)
    # both steps around the outlier exceed the limit, so its neighbours are re-interpolated too
    assert n == 3 and np.allclose(fixed, line)
    np.testing.assert_allclose(bf.smooth_trajectory(line, 5)[2:-2], line[2:-2], atol=1e-12)
    still = np.zeros((40, 2))
    assert (bf.label_stationary_points(still, fps=10) == bf.LABEL_STATIONARY).all()
    assert bf.filter_trajectory(still, fps=10) is None
    assert bf.filter_trajectory(line[:10], fps=10) is None
    points, labels = bf.filter_trajectory(line, fps=10, coordinate_bounds=(-1, 3))
    assert (labels[points[:, 0] > 3] == bf.LABEL_OUT_OF_BOUNDS).all()
