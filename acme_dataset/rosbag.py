"""Read the raw ACME ROS 2 bags without a ROS installation.

Uses the pure-Python `rosbags <https://gitlab.com/ternaris/rosbags>`_ library.
Message definitions that are not part of the ROS 2 Humble core
(``vision_msgs``, the ACME ``custom_msgs`` and ``map_msgs``) are registered
automatically.

Example:
    >>> from acme_dataset.rosbag import Bag, decode_odometry
    >>> with Bag("ACME-dataset/ROS2_Bags/miraikan/<bag>.zip") as bag:
    ...     print(bag.topics)
    ...     for topic, t_ns, msg in bag.messages(["/odom"]):
    ...         print(t_ns, decode_odometry(msg))
"""

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

PathLike = Union[str, Path]

# Definitions of the non-core message types found in ACME bags.
EXTRA_MSG_DEFINITIONS = {
    # vision_msgs 4.x (ROS 2 Humble)
    "vision_msgs/msg/Point2D": "float64 x\nfloat64 y",
    "vision_msgs/msg/Pose2D": "vision_msgs/Point2D position\nfloat64 theta",
    "vision_msgs/msg/BoundingBox2D": "vision_msgs/Pose2D center\nfloat64 size_x\nfloat64 size_y",
    "vision_msgs/msg/ObjectHypothesis": "string class_id\nfloat64 score",
    "vision_msgs/msg/ObjectHypothesisWithPose":
        "vision_msgs/ObjectHypothesis hypothesis\ngeometry_msgs/PoseWithCovariance pose",
    "vision_msgs/msg/Detection2D":
        "std_msgs/Header header\nvision_msgs/ObjectHypothesisWithPose[] results\n"
        "vision_msgs/BoundingBox2D bbox\nstring id",
    "vision_msgs/msg/Detection2DArray": "std_msgs/Header header\nvision_msgs/Detection2D[] detections",
    # ACME anonymization outputs (anonymization-bag/custom_msgs)
    "custom_msgs/msg/BinaryMask": "std_msgs/Header header\nint32 width\nint32 height\nbool[] data",
    "custom_msgs/msg/Keypoint": "float32 x\nfloat32 y\nfloat32 visibility",
    "custom_msgs/msg/Keypoints": "custom_msgs/Keypoint[] keypoints",
    "custom_msgs/msg/KeypointArray": "std_msgs/Header header\ncustom_msgs/Keypoints[] people",
    # map_msgs
    "map_msgs/msg/OccupancyGridUpdate":
        "std_msgs/Header header\nint32 x\nint32 y\nuint32 width\nuint32 height\nint8[] data",
}

BBOX_ENCODINGS = ("center", "legacy")


@lru_cache(maxsize=1)
def get_typestore():
    """ROS 2 Humble typestore with the ACME-specific message types registered."""
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore as _get_typestore

    typestore = _get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    for name, definition in EXTRA_MSG_DEFINITIONS.items():
        types.update(get_types_from_msg(definition, name))
    typestore.register(types)
    return typestore


# ---------------------------------------------------------------------- files
def extract_bag(zip_path: PathLike, out_dir: Optional[PathLike] = None) -> Path:
    """Extract a zipped bag (as distributed on Hugging Face) and return the bag folder.

    The archive is extracted next to the zip file unless ``out_dir`` is given.
    Extraction is skipped if the bag folder already exists.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir) if out_dir is not None else zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        metadata = [n for n in names if n.endswith("metadata.yaml")]
        if not metadata:
            raise ValueError(f"{zip_path} does not contain a ROS 2 bag (no metadata.yaml)")
        bag_dir = out_dir / Path(metadata[0]).parent
        if not (bag_dir / "metadata.yaml").exists():
            root = out_dir.resolve()
            for name in names:
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError(f"unsafe path in archive: {name}")
            zf.extractall(out_dir)
    return bag_dir


def resolve_bag(path: PathLike, extract_dir: Optional[PathLike] = None) -> Path:
    """Return the bag folder for a bag folder, ``.db3``/``.mcap`` file, or ``.zip`` archive."""
    path = Path(path)
    if path.suffix == ".zip":
        return extract_bag(path, extract_dir)
    if path.is_file() and path.suffix in (".db3", ".mcap"):
        return path.parent
    if (path / "metadata.yaml").exists():
        return path
    candidates = [p.parent for p in path.glob("*/metadata.yaml")]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"no ROS 2 bag found at {path}")


def read_metadata(path: PathLike) -> dict:
    """``metadata.yaml`` of a bag folder or zipped bag, without extracting the archive."""
    import yaml

    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.endswith("metadata.yaml")), None)
            if name is None:
                raise ValueError(f"{path} does not contain a ROS 2 bag (no metadata.yaml)")
            text = zf.read(name).decode()
    else:
        text = (resolve_bag(path) / "metadata.yaml").read_text()
    return yaml.safe_load(text)["rosbag2_bagfile_information"]


def metadata_summary(path: PathLike) -> Tuple[float, pd.DataFrame]:
    """``(duration_s, topics)`` of a bag from its metadata; ``topics`` has columns
    ``topic, msgtype, count``."""
    info = read_metadata(path)
    rows = [(t["topic_metadata"]["name"], t["topic_metadata"]["type"], t["message_count"])
            for t in info["topics_with_message_count"]]
    return info["duration"]["nanoseconds"] / 1e9, pd.DataFrame(sorted(rows), columns=["topic", "msgtype", "count"])


def find_bags(root: PathLike) -> List[Path]:
    """Find bag folders (containing ``metadata.yaml``) and zipped bags below ``root``."""
    root = Path(root)
    if root.suffix == ".zip" or (root / "metadata.yaml").exists():
        return [root]
    folders = {p.parent for p in root.rglob("metadata.yaml")}
    zips = {p for p in root.rglob("*.zip") if p.with_suffix("") not in folders}
    return sorted(folders | zips)


class Bag:
    """Context manager around a ``rosbags`` reader for one ACME bag."""

    def __init__(self, path: PathLike, extract_dir: Optional[PathLike] = None):
        self.source = Path(path)
        self.path = resolve_bag(path, extract_dir)
        self.name = self.path.name
        self._reader = None

    def open(self) -> "Bag":
        from rosbags.highlevel import AnyReader

        if self._reader is None:
            self._reader = AnyReader([self.path], default_typestore=get_typestore())
            self._reader.open()
        return self

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "Bag":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def reader(self):
        if self._reader is None:
            raise RuntimeError("bag is not open: use `with Bag(path) as bag:` or `bag = Bag(path).open()`")
        return self._reader

    @property
    def topics(self) -> pd.DataFrame:
        """Topic table with columns ``topic, msgtype, count``."""
        rows = {}
        for conn in self.reader.connections:
            topic, msgtype, count = rows.get(conn.topic, (conn.topic, conn.msgtype, 0))
            rows[conn.topic] = (topic, msgtype, count + conn.msgcount)
        return pd.DataFrame(sorted(rows.values()), columns=["topic", "msgtype", "count"])

    @property
    def start_time(self) -> float:
        return self.reader.start_time / 1e9

    @property
    def end_time(self) -> float:
        return self.reader.end_time / 1e9

    @property
    def duration(self) -> float:
        return (self.reader.end_time - self.reader.start_time) / 1e9

    def connections(self, topics: Optional[Iterable[str]] = None) -> list:
        if topics is None:
            return list(self.reader.connections)
        topics = set(topics)
        return [c for c in self.reader.connections if c.topic in topics]

    def raw_messages(self, topics: Optional[Iterable[str]] = None) -> Iterator[Tuple[object, int, bytes]]:
        """Yield ``(connection, timestamp_ns, raw_bytes)`` in time order."""
        yield from self.reader.messages(connections=self.connections(topics))

    def deserialize(self, raw: bytes, msgtype: str):
        return self.reader.deserialize(raw, msgtype)

    def messages(self, topics: Optional[Iterable[str]] = None) -> Iterator[Tuple[str, int, object]]:
        """Yield ``(topic, timestamp_ns, message)`` in time order."""
        for conn, t_ns, raw in self.raw_messages(topics):
            yield conn.topic, t_ns, self.reader.deserialize(raw, conn.msgtype)

    def read_topic(self, topic: str, limit: Optional[int] = None) -> List[Tuple[int, object]]:
        """All ``(timestamp_ns, message)`` pairs of one topic (optionally the first ``limit``)."""
        out = []
        for _, t_ns, msg in self.messages([topic]):
            out.append((t_ns, msg))
            if limit is not None and len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------ helpers
    def odometry(self, topic: str) -> pd.DataFrame:
        """All odometry messages of ``topic`` as a DataFrame (see :func:`decode_odometry`)."""
        rows = [{"timestamp": t_ns / 1e9, **decode_odometry(msg)} for _, t_ns, msg in self.messages([topic])]
        return pd.DataFrame(rows)

    def static_transforms(self) -> pd.DataFrame:
        """``/tf_static`` as a table: ``parent, child, x, y, z, qx, qy, qz, qw``."""
        rows = {}
        for _, _, msg in self.messages(["/tf_static"]):
            for tf in msg.transforms:
                tr, q = tf.transform.translation, tf.transform.rotation
                rows[(tf.header.frame_id, tf.child_frame_id)] = (
                    tf.header.frame_id, tf.child_frame_id, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)
        return pd.DataFrame(list(rows.values()),
                            columns=["parent", "child", "x", "y", "z", "qx", "qy", "qz", "qw"])

    def camera_info(self) -> pd.DataFrame:
        """First message of every ``sensor_msgs/msg/CameraInfo`` topic (intrinsics)."""
        topics = [c.topic for c in self.reader.connections if c.msgtype == "sensor_msgs/msg/CameraInfo"]
        rows = []
        for topic in sorted(set(topics)):
            for _, msg in self.read_topic(topic, limit=1):
                rows.append({"topic": topic, **decode_camera_info(msg)})
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ decoders
def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rotation about z, counter-clockwise, radians) of a quaternion."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def decode_odometry(msg) -> dict:
    """``nav_msgs/Odometry`` to the columns used by ``traj_data.csv``."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    v, w = msg.twist.twist.linear, msg.twist.twist.angular
    return {
        "position_x": p.x, "position_y": p.y, "yaw": quat_to_yaw(q.x, q.y, q.z, q.w),
        "linear_vel_x": v.x, "linear_vel_y": v.y, "angular_vel": w.z,
    }


def decode_compressed_image(msg) -> np.ndarray:
    """``sensor_msgs/CompressedImage`` to an ``(H, W, 3)`` RGB uint8 array."""
    return np.array(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))


_POINTFIELD_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                      5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def decode_pointcloud2(msg) -> np.ndarray:
    """``sensor_msgs/PointCloud2`` to an ``(N, 4)`` float32 x, y, z, intensity array.

    Non-finite points are dropped. Intensity is zero if the cloud has no
    ``intensity`` field.
    """
    names = [f.name for f in msg.fields]
    dtype = np.dtype({
        "names": names,
        "formats": [_POINTFIELD_DTYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")
    n_points = msg.width * msg.height
    data = np.asarray(msg.data, dtype=np.uint8)
    if msg.row_step != msg.width * msg.point_step:
        data = data.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step].reshape(-1)
    cloud = np.frombuffer(data.tobytes(), dtype=dtype, count=n_points)
    points = np.zeros((n_points, 4), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z")):
        points[:, i] = cloud[name]
    if "intensity" in names:
        points[:, 3] = cloud["intensity"]
    return points[np.isfinite(points).all(axis=1)]


def decode_detections(msg, encoding: str = "center") -> Tuple[List[List[float]], List[str]]:
    """``vision_msgs/Detection2DArray`` to ``(boxes, track_ids)`` with boxes as
    ``[x, y, width, height]`` (top-left corner, pixels).

    ACME bags use one of two box-center conventions, depending on the version
    of the anonymization pipeline that wrote them:

    - ``"center"``: ``bbox.center`` is the true box center (standard vision_msgs).
    - ``"legacy"``: ``bbox.center`` stores ``(top_left + size) / 2``.

    Use :func:`detect_bbox_encoding` to find out which one a bag uses.
    """
    if encoding not in BBOX_ENCODINGS:
        raise ValueError(f"encoding must be one of {BBOX_ENCODINGS}, got {encoding!r}")
    boxes, ids = [], []
    for det in msg.detections:
        cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
        w, h = det.bbox.size_x, det.bbox.size_y
        if encoding == "center":
            x, y = cx - w / 2.0, cy - h / 2.0
        else:
            x, y = 2.0 * cx - w, 2.0 * cy - h
        boxes.append([float(x), float(y), float(w), float(h)])
        ids.append(str(det.id))
    return boxes, ids


def decode_binary_mask(msg) -> np.ndarray:
    """``custom_msgs/BinaryMask`` to an ``(H, W)`` boolean array (True = person)."""
    return np.asarray(msg.data, dtype=bool).reshape(msg.height, msg.width)


def decode_keypoints(msg) -> List[np.ndarray]:
    """``custom_msgs/KeypointArray`` to one ``(K, 3)`` array (x, y, visibility) per person."""
    return [np.array([[k.x, k.y, k.visibility] for k in person.keypoints], dtype=np.float32)
            for person in msg.people]


def decode_camera_info(msg) -> dict:
    """``sensor_msgs/CameraInfo`` to a dict with ``K`` (3x3) and distortion ``D``."""
    return {
        "frame_id": msg.header.frame_id,
        "width": msg.width,
        "height": msg.height,
        "distortion_model": msg.distortion_model,
        "K": np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
        "D": np.asarray(msg.d, dtype=np.float64),
    }


def image_topic_for(tracks_topic: str, topics: Sequence[str]) -> Optional[str]:
    """Image topic that a ``/tracks_*`` (or ``/masks_*``) topic was computed from."""
    suffix = tracks_topic.split("_", 1)[1] if "_" in tracks_topic else ""
    for topic in topics:
        if topic.replace("/", "_").strip("_") == suffix:
            return topic
    return None


def detect_bbox_encoding(bag: Bag, tracks_topic: str, max_messages: int = 300) -> str:
    """Guess the box-center convention (see :func:`decode_detections`) of a bag.

    If the bag has a matching ``/masks_*`` topic, the convention whose boxes
    cover more mask pixels wins. Otherwise the image width is used: with the
    legacy convention ``2 * center_x`` is the right edge of the box and stays
    within the image, while with true centers it often exceeds it.
    """
    topics = bag.topics["topic"].tolist()
    mask_topic = "/masks_" + tracks_topic.split("_", 1)[1]
    if mask_topic in topics:
        tracks, masks = {}, {}
        for topic, _, msg in bag.messages([tracks_topic, mask_topic]):
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            (tracks if topic == tracks_topic else masks)[stamp] = msg
            if min(len(tracks), len(masks)) >= max_messages:
                break
        common = sorted(set(tracks) & set(masks))
        if common:
            scores = {}
            for encoding in BBOX_ENCODINGS:
                covered = total = 0
                for stamp in common:
                    mask = decode_binary_mask(masks[stamp])
                    boxes_mask = np.zeros_like(mask)
                    for x, y, w, h in decode_detections(tracks[stamp], encoding)[0]:
                        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
                        boxes_mask[y0:max(int(round(y + h)), 0), x0:max(int(round(x + w)), 0)] = True
                    covered += int((mask & boxes_mask).sum())
                    total += int(mask.sum())
                scores[encoding] = covered / max(total, 1)
            return max(scores, key=scores.get)

    image_topic = image_topic_for(tracks_topic, topics)
    if image_topic is None:
        raise ValueError(f"cannot find the image topic of {tracks_topic}; pass the encoding explicitly")
    width = decode_compressed_image(bag.read_topic(image_topic, limit=1)[0][1]).shape[1]
    right_edges = []
    for _, _, msg in bag.messages([tracks_topic]):
        right_edges += [2.0 * d.bbox.center.position.x for d in msg.detections]
        if len(right_edges) >= 20 * max_messages:
            break
    if not right_edges:
        return "center"
    return "center" if np.mean(np.asarray(right_edges) > 1.2 * width) > 0.01 else "legacy"
