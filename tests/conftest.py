"""Synthetic ACME-like data so the tests run without downloading the dataset."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

IMG_W, IMG_H = 64, 48


def _jpeg(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def trajectory_dir(tmp_path: Path) -> Path:
    """A 5-frame human-readable trajectory folder."""
    root = tmp_path / "human_readable_sampled" / "go2nus" / "test_traj_0"
    for sub in ("imgs", "masked_imgs", "pcd"):
        (root / sub).mkdir(parents=True)
    n = 5
    t0 = 1_700_000_000.0
    (root / "timestamps.txt").write_text("".join(f"{t0 + 0.25 * i}\n" for i in range(n)))
    pd.DataFrame({
        "frame_idx": range(n), "position_x": np.arange(n) * 0.5, "position_y": 0.0, "yaw": 0.0,
        "linear_vel_x": 2.0, "linear_vel_y": 0.0, "angular_vel": 0.1,
    }).to_csv(root / "traj_data.csv", index=False)
    tracks = [{"frame_idx": i, "detections": [
        {"track_id": "7", "bbox": {"x": 10.0, "y": 5.0, "width": 8.0, "height": 20.0}},
        {"track_id": "9", "bbox": {"x": 30.0 + i, "y": 6.0, "width": 6.0, "height": 18.0}},
    ][: 1 + (i % 2)]} for i in range(n)]
    (root / "tracks.json").write_text(json.dumps(tracks))
    for i in range(n):
        Image.fromarray(np.full((IMG_H, IMG_W, 3), 40 * i, np.uint8)).save(root / "imgs" / f"{i}.jpg")
        mask = np.zeros((IMG_H, IMG_W), np.uint8)
        mask[5:25, 10:18] = 255
        Image.fromarray(mask).save(root / "masked_imgs" / f"{i}.jpg")
        np.savez(root / "pcd" / f"{i}.npz", np.full((10, 4), i, np.float32))
    (root / "speech.txt").write_text(f"{t0 + 0.3}: excuse me\n")
    return root


@pytest.fixture
def bev_root(tmp_path: Path) -> Path:
    """A BEV annotation tree with one session ``toy/s1`` and an identity-like homography."""
    root = tmp_path / "bev" / "bev_annotations"
    stem = "toy_s1_transformed"
    for sub in ("frames", "traj_frames", "annotations", f"masks/{stem}",
                "trajectory-transform/trajectories/toy", "trajectory-transform/homographies/toy"):
        (root / sub).mkdir(parents=True)
    w, h = 200, 100
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(root / "frames" / f"{stem}.jpg")
    left = np.zeros((h, w), np.uint8)
    left[:, :100] = 255
    Image.fromarray(left).save(root / "masks" / stem / "tag_1_Open_Area.jpg")
    Image.fromarray(255 - left).save(root / "masks" / stem / "tag_3_Narrow_Corridor.jpg")
    (root / "annotations" / f"{stem}.json").write_text(json.dumps([
        {"tag_id": 1, "tag_name": "Open Area", "points": [[0, 0], [99, 0], [99, 99], [0, 99]]},
        {"tag_id": 3, "tag_name": "Narrow Corridor", "points": [[100, 0], [199, 0], [199, 99], [100, 99]]},
    ]))
    # 0.1 m per pixel
    (root / "trajectory-transform/homographies/toy/s1_homography.txt").write_text(
        "0.1 0 0\n0 0.1 0\n0 0 1\n")
    walker = [[10.0 + 2.0 * i, 50.0] for i in range(80)]   # 2 m/s, crosses both regions
    still = [[150.0, 20.0] for _ in range(60)]             # never moves
    short = [[20.0, 20.0 + i] for i in range(10)]          # 1 s long
    lines = []
    for name, start, pts in (("human1", 3, walker), ("human2", 0, still), ("human3", 5, short)):
        lines += [f"[{name}]", f"traj_start = {start}",
                  "trajectories = [ " + ", ".join(f"[ {x}, {y},]" for x, y in pts) + ",]", ""]
    (root / "trajectory-transform/trajectories/toy/s1.toml").write_text("\n".join(lines))
    return tmp_path


@pytest.fixture
def synthetic_bag(tmp_path: Path):
    """Write a small ROS 2 bag with every topic type the converter reads.

    Returns ``(bag_path, expected_boxes)`` where the tracks are written with the
    legacy center convention.
    """
    from rosbags.rosbag2 import Writer

    from acme_dataset.rosbag import get_typestore

    ts = get_typestore()
    T = ts.types
    path = tmp_path / "bags" / "go2nus" / "toy_bag"
    path.parent.mkdir(parents=True)

    def header(t_ns):
        return T["std_msgs/msg/Header"](stamp=T["builtin_interfaces/msg/Time"](sec=t_ns // 10**9, nanosec=t_ns % 10**9),
                                        frame_id="base")

    boxes = [[4.0, 6.0, 10.0, 20.0], [40.0, 10.0, 12.0, 30.0]]
    image = _jpeg(np.full((IMG_H, IMG_W, 3), 128, np.uint8))
    mask = np.zeros((IMG_H, IMG_W), bool)
    for x, y, w, h in boxes:
        mask[int(y):int(y + h), int(x):int(x + w)] = True
    cloud_points = np.array([[1, 2, 3, 10], [4, 5, 6, 20], [np.nan, 0, 0, 0]], np.float32)

    with Writer(path, version=8) as writer:
        conns = {
            "/utlidar/robot_odom": writer.add_connection("/utlidar/robot_odom", "nav_msgs/msg/Odometry", typestore=ts),
            "/camera/camera/color/image_raw/compressed": writer.add_connection(
                "/camera/camera/color/image_raw/compressed", "sensor_msgs/msg/CompressedImage", typestore=ts),
            "/rslidar_points": writer.add_connection("/rslidar_points", "sensor_msgs/msg/PointCloud2", typestore=ts),
            "/tracks_camera_camera_color_image_raw_compressed": writer.add_connection(
                "/tracks_camera_camera_color_image_raw_compressed", "vision_msgs/msg/Detection2DArray", typestore=ts),
            "/masks_camera_camera_color_image_raw_compressed": writer.add_connection(
                "/masks_camera_camera_color_image_raw_compressed", "custom_msgs/msg/BinaryMask", typestore=ts),
            "/speech_command": writer.add_connection("/speech_command", "std_msgs/msg/String", typestore=ts),
        }

        def write(topic, t_ns, msg):
            writer.write(conns[topic], t_ns, ts.serialize_cdr(msg, msg.__msgtype__))

        t0 = 1_700_000_000 * 10**9
        for k in range(50):  # 10 Hz for 5 s
            t = t0 + k * 10**8
            pose = T["geometry_msgs/msg/Pose"](
                position=T["geometry_msgs/msg/Point"](x=0.1 * k, y=0.0, z=0.0),
                orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0))
            twist = T["geometry_msgs/msg/Twist"](linear=T["geometry_msgs/msg/Vector3"](x=1.0, y=0.0, z=0.0),
                                                 angular=T["geometry_msgs/msg/Vector3"](x=0.0, y=0.0, z=0.2))
            write("/utlidar/robot_odom", t, T["nav_msgs/msg/Odometry"](
                header=header(t), child_frame_id="base_link",
                pose=T["geometry_msgs/msg/PoseWithCovariance"](pose=pose, covariance=np.zeros(36)),
                twist=T["geometry_msgs/msg/TwistWithCovariance"](twist=twist, covariance=np.zeros(36))))
            write("/camera/camera/color/image_raw/compressed", t + 1, T["sensor_msgs/msg/CompressedImage"](
                header=header(t), format="jpeg", data=np.frombuffer(image, np.uint8)))
            fields = [T["sensor_msgs/msg/PointField"](name=n, offset=4 * i, datatype=7, count=1)
                      for i, n in enumerate(("x", "y", "z", "intensity"))]
            write("/rslidar_points", t + 2, T["sensor_msgs/msg/PointCloud2"](
                header=header(t), height=1, width=len(cloud_points), fields=fields, is_bigendian=False,
                point_step=16, row_step=16 * len(cloud_points),
                data=np.frombuffer(cloud_points.tobytes(), np.uint8), is_dense=False))
            dets = []
            for i, (x, y, w, h) in enumerate(boxes):
                center = T["vision_msgs/msg/Pose2D"](position=T["vision_msgs/msg/Point2D"](
                    x=(x + w) / 2, y=(y + h) / 2), theta=0.0)  # legacy convention
                dets.append(T["vision_msgs/msg/Detection2D"](
                    header=header(t), results=[], id=str(100 + i),
                    bbox=T["vision_msgs/msg/BoundingBox2D"](center=center, size_x=w, size_y=h)))
            write("/tracks_camera_camera_color_image_raw_compressed", t + 3,
                  T["vision_msgs/msg/Detection2DArray"](header=header(t), detections=dets))
            write("/masks_camera_camera_color_image_raw_compressed", t + 4, T["custom_msgs/msg/BinaryMask"](
                header=header(t), width=IMG_W, height=IMG_H, data=mask.reshape(-1)))
            if k == 20:
                write("/speech_command", t + 5, T["std_msgs/msg/String"](data="excuse me"))
    return path, boxes
