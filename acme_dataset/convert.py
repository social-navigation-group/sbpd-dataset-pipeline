"""Convert raw ACME ROS 2 bags into the human-readable format (no ROS needed).

The sampling follows the pipeline used to build the released data: messages
are read in time order, the latest message of every required topic is kept,
and a sample is emitted as soon as at least ``1 / rate`` seconds have passed
since the previous sample and every required topic has a new message.

Output (one folder per bag, named ``<bag>_0``)::

    imgs/<i>.jpg         (imgs_<camera>/ for multi-camera robots)
    masked_imgs/<i>.jpg  (masked_imgs_<camera>/ for several mask topics)
    pcd/<i>.npz, pcd/<i>.pcd
    traj_data.csv, traj_data.pkl
    tracks.json, tracks.pkl
    timestamps.txt
    speech.txt           (if the bag has a speech topic)
"""

import io
import json
import pickle
import shutil
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .constants import HUMAN_READABLE_RATE_HZ, INSTITUTIONS
from .pointcloud import write_pcd
from .rosbag import (Bag, decode_binary_mask, decode_detections, decode_odometry,
                     decode_pointcloud2, detect_bbox_encoding)

PathLike = Union[str, Path]
CONFIG_PATH = Path(__file__).parent / "configs" / "topics.yaml"
PCD_FORMATS = ("npz", "pcd", "both")


@lru_cache(maxsize=None)
def load_topic_config(path: PathLike = CONFIG_PATH) -> Dict[str, dict]:
    """Per-institution topic configuration (see ``configs/topics.yaml``)."""
    with open(path) as f:
        return yaml.safe_load(f)


def guess_institution(path: PathLike) -> Optional[str]:
    """Institution name found in the path components, e.g. ``ros2 bags/miraikan/x.zip``."""
    for part in reversed(Path(path).resolve().parts):
        if part in INSTITUTIONS:
            return part
    return None


def _folder_name(prefix: str, topic: str, n_topics: int, strip: str = "") -> str:
    if n_topics == 1:
        return prefix
    name = topic.replace("/", "_").replace("compressed", "").replace(strip, "").strip("_")
    return f"{prefix}_{name}"


def _write_image(raw_msg, path: Path) -> None:
    data = bytes(raw_msg.data)
    if raw_msg.format.lower().startswith(("jpeg", "jpg")) or data[:2] == b"\xff\xd8":
        path.write_bytes(data)
    else:
        Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=95)


def convert_bag(
    bag_path: PathLike,
    output_dir: PathLike,
    institution: Optional[str] = None,
    config: Optional[dict] = None,
    rate: float = HUMAN_READABLE_RATE_HZ,
    pcd_format: str = "both",
    bbox_encoding: str = "auto",
    overwrite: bool = False,
    extract_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """Convert one bag (folder, ``.db3`` or ``.zip``) into a human-readable trajectory folder.

    Args:
        bag_path: The bag to convert.
        output_dir: Parent folder of the trajectory folder that is created.
        institution: Key of ``configs/topics.yaml``; guessed from the path if omitted.
        config: Topic configuration overriding the built-in one for ``institution``.
        rate: Sampling rate in Hz (the released data uses 4 Hz).
        pcd_format: ``"npz"``, ``"pcd"`` (ASCII) or ``"both"``.
        bbox_encoding: ``"auto"``, ``"center"`` or ``"legacy"``
            (see :func:`acme_dataset.rosbag.decode_detections`).
        overwrite: Replace an existing output folder instead of skipping the bag.
        extract_dir: Where to extract zipped bags (default: next to the zip).

    Returns:
        The trajectory folder, or None if the bag was skipped.
    """
    if pcd_format not in PCD_FORMATS:
        raise ValueError(f"pcd_format must be one of {PCD_FORMATS}")
    if config is None:
        institution = institution or guess_institution(bag_path)
        if institution is None:
            raise ValueError(f"cannot guess the institution of {bag_path}; pass institution=...")
        config = load_topic_config()[institution]

    with Bag(bag_path, extract_dir) as bag:
        out = Path(output_dir) / f"{bag.name}_0"
        if out.exists():
            if not overwrite:
                print(f"{out} exists, skipping (pass --overwrite / overwrite=True to replace)")
                return None
            shutil.rmtree(out)

        counts = dict(zip(bag.topics["topic"], bag.topics["count"]))
        present = [t for t in counts if counts[t] > 0]

        def first(topics):
            return next((t for t in topics or [] if t in present), None)

        image_topics = [t for t in config.get("image_topics") or [] if t in present]
        mask_topics = [t for t in config.get("mask_topics") or [] if t in present]
        odom_topic = first(config.get("odom_topics"))
        pcl_topic = first(config.get("pointcloud_topics"))
        tracking_topic = first(config.get("tracking_topics"))
        speech_topic = config.get("speech_topic") if config.get("speech_topic") in present else None
        if odom_topic is None or not (image_topics or mask_topics):
            print(f"{bag.name}: no odometry or image topic found, skipping")
            return None

        if tracking_topic and bbox_encoding == "auto":
            bbox_encoding = detect_bbox_encoding(bag, tracking_topic)

        required = image_topics + mask_topics + [odom_topic] + [t for t in (pcl_topic, tracking_topic) if t]
        image_dirs = {t: _folder_name("imgs", t, len(image_topics)) for t in image_topics}
        mask_dirs = {t: _folder_name("masked_imgs", t, len(mask_topics), strip="masks_") for t in mask_topics}
        for folder in list(image_dirs.values()) + list(mask_dirs.values()) + (["pcd"] if pcl_topic else []):
            (out / folder).mkdir(parents=True, exist_ok=True)

        odometry, tracks, timestamps, speech = [], [], [], []
        start_slack, end_slack = int(config.get("start_slack", 0)), int(config.get("end_slack", 0))
        pending = deque()

        def write_sample(stamp: float, sample: dict) -> None:
            n = len(timestamps)
            for topic, folder in image_dirs.items():
                conn, raw = sample[topic]
                _write_image(bag.deserialize(raw, conn.msgtype), out / folder / f"{n}.jpg")
            for topic, folder in mask_dirs.items():
                conn, raw = sample[topic]
                mask = decode_binary_mask(bag.deserialize(raw, conn.msgtype))
                Image.fromarray(mask.astype(np.uint8) * 255).save(out / folder / f"{n}.jpg")
            if pcl_topic:
                conn, raw = sample[pcl_topic]
                points = decode_pointcloud2(bag.deserialize(raw, conn.msgtype))
                if pcd_format in ("npz", "both"):
                    np.savez(out / "pcd" / f"{n}.npz", points)
                if pcd_format in ("pcd", "both"):
                    write_pcd(points, out / "pcd" / f"{n}.pcd")
            if tracking_topic:
                conn, raw = sample[tracking_topic]
                tracks.append(decode_detections(bag.deserialize(raw, conn.msgtype), bbox_encoding))
            conn, raw = sample[odom_topic]
            odometry.append(decode_odometry(bag.deserialize(raw, conn.msgtype)))
            timestamps.append(stamp)

        period = 1.0 / rate
        latest = {t: None for t in required}
        last_time = None
        n_emitted = 0
        topics = required + ([speech_topic] if speech_topic else [])
        for conn, t_ns, raw in bag.raw_messages(topics):
            t = t_ns / 1e9
            if last_time is None:
                last_time = t
            if conn.topic == speech_topic:
                if speech is not None:
                    try:
                        speech.append((t, bag.deserialize(raw, conn.msgtype).data))
                    except (KeyError, AttributeError):
                        speech = None  # unknown or non-string message type
                continue
            latest[conn.topic] = (conn, raw)
            if t - last_time >= period and all(v is not None for v in latest.values()):
                if n_emitted >= start_slack:
                    pending.append((t, dict(latest)))
                    if len(pending) > end_slack:
                        write_sample(*pending.popleft())
                n_emitted += 1
                latest = {k: None for k in latest}
                last_time = t

    _write_tables(out, odometry, tracks if tracking_topic else None, timestamps, speech or [])
    return out


def _write_tables(out: Path, odometry: List[dict], tracks: Optional[list],
                  timestamps: List[float], speech: list) -> None:
    odom = pd.DataFrame(odometry, columns=["position_x", "position_y", "yaw",
                                           "linear_vel_x", "linear_vel_y", "angular_vel"])
    odom.insert(0, "frame_idx", range(len(odom)))
    odom.to_csv(out / "traj_data.csv", index=False)
    with open(out / "traj_data.pkl", "wb") as f:
        pickle.dump({
            "position": odom[["position_x", "position_y"]].to_numpy(),
            "yaw": odom["yaw"].to_numpy(),
            "linear_vel": odom[["linear_vel_x", "linear_vel_y"]].to_numpy(),
            "angular_vel": odom["angular_vel"].to_numpy(),
        }, f)
    with open(out / "timestamps.txt", "w") as f:
        f.writelines(f"{t}\n" for t in timestamps)
    if tracks is not None:
        with open(out / "tracks.pkl", "wb") as f:
            pickle.dump(tracks, f)
        frames = [{
            "frame_idx": i,
            "detections": [{"track_id": tid, "bbox": dict(zip(("x", "y", "width", "height"), box))}
                           for box, tid in zip(boxes, ids)],
        } for i, (boxes, ids) in enumerate(tracks)]
        with open(out / "tracks.json", "w") as f:
            json.dump(frames, f, indent=2)
    if speech:
        with open(out / "speech.txt", "w") as f:
            f.writelines(f"{t}: {text}\n" for t, text in speech)
