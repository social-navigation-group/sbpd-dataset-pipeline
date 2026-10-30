import json

import numpy as np
import pytest

from acme_dataset import Trajectory
from acme_dataset.convert import convert_bag, guess_institution
from acme_dataset.rosbag import (Bag, decode_detections, detect_bbox_encoding, find_bags,
                                 image_topic_for, metadata_summary)

TRACKS = "/tracks_camera_camera_color_image_raw_compressed"


def test_bag_reader(synthetic_bag):
    path, boxes = synthetic_bag
    duration, topics = metadata_summary(path)
    assert duration == pytest.approx(4.9, abs=0.01)
    assert dict(zip(topics["topic"], topics["count"]))["/utlidar/robot_odom"] == 50
    with Bag(path) as bag:
        assert bag.read_topic(TRACKS, limit=2)[1][0] > bag.read_topic(TRACKS, limit=1)[0][0]
        msg = bag.read_topic(TRACKS, limit=1)[0][1]
        assert decode_detections(msg, "legacy")[0] == boxes
        assert decode_detections(msg, "legacy")[1] == ["100", "101"]
        assert decode_detections(msg, "center")[0] != boxes
        assert detect_bbox_encoding(bag, TRACKS) == "legacy"
        odom = bag.odometry("/utlidar/robot_odom")
        assert len(odom) == 50 and odom["position_x"].iloc[10] == pytest.approx(1.0)


def test_image_topic_for():
    topics = ["/camera/camera/color/image_raw/compressed", "/odom"]
    assert image_topic_for(TRACKS, topics) == topics[0]


def test_find_bags(synthetic_bag):
    path, _ = synthetic_bag
    assert find_bags(path.parents[1]) == [path]
    assert guess_institution(path) == "go2nus"
    assert guess_institution("ROS2_Bags/nus/x.zip") == "go2nus"


def test_convert(synthetic_bag, tmp_path):
    path, boxes = synthetic_bag
    out = convert_bag(path, tmp_path / "out")
    assert out.name == "toy_bag_0"
    traj = Trajectory(out)
    # a sample is emitted on the first message >= 0.25 s after the previous one,
    # so with 10 Hz inputs samples are 0.3 s apart
    assert len(traj) == 16
    np.testing.assert_allclose(np.diff(traj.timestamps), 0.3, atol=1e-6)
    assert traj.image(0).shape == (48, 64, 3)
    assert traj.mask(0)[10, 6] and not traj.mask(0)[0, 0]
    np.testing.assert_allclose(traj.pointcloud(0), [[1, 2, 3, 10], [4, 5, 6, 20]])
    assert (out / "pcd" / "0.pcd").exists() and not (out / "pcd" / "0.npz").exists()
    assert traj.mask_folders == ["masks"]
    first = traj.detections_at(0)
    assert first[["x", "y", "width", "height"]].to_numpy().tolist() == boxes
    assert traj.speech["text"].tolist() == ["excuse me"]
    assert len(json.loads((out / "tracks.json").read_text())) == len(traj)
    assert convert_bag(path, tmp_path / "out") is None  # existing output is skipped


def test_convert_slack(synthetic_bag, tmp_path):
    from acme_dataset.convert import load_topic_config

    path, _ = synthetic_bag
    full = Trajectory(convert_bag(path, tmp_path / "a"))
    config = dict(load_topic_config()["go2nus"], start_slack=2, end_slack=1)
    sliced = Trajectory(convert_bag(path, tmp_path / "b", config=config))
    assert len(sliced) == len(full) - 3
    np.testing.assert_allclose(sliced.timestamps, full.timestamps[2:-1])
