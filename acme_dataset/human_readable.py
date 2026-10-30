"""Load trajectories from the human-readable (4 Hz resampled) part of ACME.

Each trajectory folder looks like::

    human_readable_sampled/<institution>/<trajectory>/
        timestamps.txt      one UNIX timestamp (s) per sample
        traj_data.csv       robot odometry per sample (odom frame)
        tracks.json         pedestrian 2D boxes + track ids per sample
        imgs/<i>.jpg        egocentric RGB (imgs_<camera>/ for multi-camera robots)
        masked_imgs/<i>.jpg binary person segmentation masks (255 = person)
        pcd/<i>.pcd         LiDAR point cloud, x y z intensity (and/or <i>.npz)
        speech.txt          robot utterances "<timestamp>: <text>" (some institutions)

Sample ``i`` of every modality refers to the same instant ``timestamps[i]``.
"""

import json
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from .constants import HUMAN_READABLE_DIR
from .pointcloud import read_npz, read_pcd

PathLike = Union[str, Path]

ODOMETRY_COLUMNS = ["frame_idx", "position_x", "position_y", "yaw",
                    "linear_vel_x", "linear_vel_y", "angular_vel"]
DETECTION_COLUMNS = ["frame_idx", "timestamp", "track_id", "x", "y", "width", "height"]


@dataclass
class Frame:
    """All modalities of one synchronized sample."""
    index: int
    timestamp: float
    odometry: pd.Series
    detections: pd.DataFrame
    image: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    pointcloud: Optional[np.ndarray]


class Trajectory:
    """One trajectory folder of the human-readable data.

    Example:
        >>> traj = Trajectory("ACME-dataset/human_readable_sampled/go2nus/<name>")
        >>> traj.odometry.head()
        >>> img = traj.image(0)
        >>> boxes = traj.detections_at(0)
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        if not (self.path / "timestamps.txt").exists():
            raise FileNotFoundError(f"{self.path} is not a trajectory folder (no timestamps.txt)")
        self.name = self.path.name
        self.institution = self.path.parent.name

    def __repr__(self) -> str:
        return f"Trajectory({self.institution}/{self.name}, {len(self)} frames)"

    def __len__(self) -> int:
        return len(self.timestamps)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.frame(i)

    # ------------------------------------------------------------------ tables
    @cached_property
    def timestamps(self) -> np.ndarray:
        """UNIX timestamps in seconds, one per sample."""
        return np.loadtxt(self.path / "timestamps.txt", dtype=np.float64, ndmin=1)

    @cached_property
    def odometry(self) -> pd.DataFrame:
        """Robot odometry with columns ``timestamp, position_x, position_y, yaw,
        linear_vel_x, linear_vel_y, angular_vel`` (odom frame, SI units)."""
        csv_path = self.path / "traj_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            with open(self.path / "traj_data.pkl", "rb") as f:
                data = pickle.load(f)
            df = pd.DataFrame({
                "frame_idx": np.arange(len(data["position"])),
                "position_x": data["position"][:, 0],
                "position_y": data["position"][:, 1],
                "yaw": data["yaw"],
                "linear_vel_x": data["linear_vel"][:, 0],
                "linear_vel_y": data["linear_vel"][:, 1],
                "angular_vel": data["angular_vel"],
            })
        n = min(len(df), len(self.timestamps))
        df = df.iloc[:n].copy()
        df.insert(1, "timestamp", self.timestamps[:n])
        return df

    @cached_property
    def detections(self) -> pd.DataFrame:
        """Pedestrian boxes, one row per (frame, pedestrian).

        Columns: ``frame_idx, timestamp, track_id, x, y, width, height`` where
        ``(x, y)`` is the top-left corner in image pixels. ``track_id`` is
        consistent across frames of the same trajectory.
        """
        rows = []
        json_path = self.path / "tracks.json"
        if json_path.exists():
            with open(json_path) as f:
                for frame in json.load(f):
                    for det in frame["detections"]:
                        b = det["bbox"]
                        rows.append((frame["frame_idx"], str(det["track_id"]),
                                     b["x"], b["y"], b["width"], b["height"]))
        elif (self.path / "tracks.pkl").exists():
            with open(self.path / "tracks.pkl", "rb") as f:
                for i, (boxes, ids) in enumerate(pickle.load(f)):
                    for box, track_id in zip(boxes or [], ids or []):
                        rows.append((i, str(track_id), *box))
        df = pd.DataFrame(rows, columns=["frame_idx", "track_id", "x", "y", "width", "height"])
        df["frame_idx"] = df["frame_idx"].astype(int)
        ts = self.timestamps
        df.insert(1, "timestamp", ts[df["frame_idx"].clip(upper=len(ts) - 1)] if len(df) else [])
        return df[DETECTION_COLUMNS]

    @cached_property
    def speech(self) -> pd.DataFrame:
        """Robot utterances with columns ``timestamp, text`` (empty if none)."""
        txt_path, csv_path = self.path / "speech.txt", self.path / "speech.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = ["timestamp", "text"] + list(df.columns[2:])
            return df
        rows = []
        if txt_path.exists():
            for line in txt_path.read_text().splitlines():
                stamp, sep, text = line.partition(": ")
                if sep:
                    rows.append((float(stamp), text))
        return pd.DataFrame(rows, columns=["timestamp", "text"])

    def detections_at(self, index: int) -> pd.DataFrame:
        """Pedestrian boxes of sample ``index``."""
        return self.detections[self.detections["frame_idx"] == index]

    # ------------------------------------------------------------- per-sample
    @cached_property
    def cameras(self) -> List[str]:
        """Image folder names, e.g. ``['imgs']`` or ``['imgs_zed2_...', 'imgs_insta360_...']``."""
        return sorted(p.name for p in self.path.iterdir()
                      if p.is_dir() and (p.name == "imgs" or p.name.startswith("imgs_")))

    @cached_property
    def mask_folders(self) -> List[str]:
        """Person-mask folder names (``masked_imgs`` or ``masks``, plus per-camera variants)."""
        return sorted(p.name for p in self.path.iterdir()
                      if p.is_dir() and p.name.startswith(("masked_imgs", "masks")))

    def image_path(self, index: int, camera: Optional[str] = None) -> Path:
        camera = camera or (self.cameras[0] if self.cameras else "imgs")
        return self.path / camera / f"{index}.jpg"

    def image(self, index: int, camera: Optional[str] = None) -> Optional[np.ndarray]:
        """RGB image of sample ``index`` as an ``(H, W, 3)`` uint8 array, or None."""
        path = self.image_path(index, camera)
        return np.array(Image.open(path).convert("RGB")) if path.exists() else None

    def mask(self, index: int, folder: Optional[str] = None) -> Optional[np.ndarray]:
        """Boolean person mask of sample ``index`` (JPEG-compressed, thresholded at 128)."""
        if folder is None:
            if not self.mask_folders:
                return None
            folder = self.mask_folders[0]
        path = self.path / folder / f"{index}.jpg"
        if not path.exists():
            return None
        return np.array(Image.open(path).convert("L")) > 128

    def pointcloud_path(self, index: int) -> Optional[Path]:
        for suffix in (".npz", ".pcd"):
            path = self.path / "pcd" / f"{index}{suffix}"
            if path.exists():
                return path
        return None

    def pointcloud(self, index: int) -> Optional[np.ndarray]:
        """LiDAR points of sample ``index`` as an ``(N, 4)`` x, y, z, intensity array
        in the LiDAR frame, or None."""
        path = self.pointcloud_path(index)
        if path is None:
            return None
        return read_npz(path) if path.suffix == ".npz" else read_pcd(path)

    def frame(self, index: int, load_pointcloud: bool = True) -> Frame:
        if not 0 <= index < len(self):
            raise IndexError(f"frame {index} out of range for {len(self)} frames")
        odom = self.odometry
        return Frame(
            index=index,
            timestamp=float(self.timestamps[index]),
            odometry=odom.iloc[index] if index < len(odom) else pd.Series(dtype=float),
            detections=self.detections_at(index),
            image=self.image(index),
            mask=self.mask(index),
            pointcloud=self.pointcloud(index) if load_pointcloud else None,
        )

    def summary(self) -> dict:
        """Basic facts about the trajectory (no image or point cloud loading)."""
        pcd_dir = self.path / "pcd"
        return {
            "institution": self.institution,
            "trajectory": self.name,
            "frames": len(self),
            "duration_s": float(self.timestamps[-1] - self.timestamps[0]) if len(self) > 1 else 0.0,
            "cameras": self.cameras,
            "mask_folders": self.mask_folders,
            "pointclouds": len({p.stem for p in pcd_dir.iterdir()}) if pcd_dir.exists() else 0,
            "pedestrian_boxes": len(self.detections),
            "unique_pedestrians": int(self.detections["track_id"].nunique()),
            "speech_events": len(self.speech),
        }


def find_trajectory_dirs(root: PathLike, institution: Optional[str] = None) -> List[Path]:
    """Return trajectory folders below ``root``.

    ``root`` may be the dataset root, its ``human_readable_sampled`` folder,
    an institution folder, or a single trajectory folder.
    """
    root = Path(root)
    if (root / "timestamps.txt").exists():
        return [root]
    if (root / HUMAN_READABLE_DIR).is_dir():
        root = root / HUMAN_READABLE_DIR
    if institution is not None:
        root = root / institution
    return sorted((p.parent for p in root.rglob("timestamps.txt")), key=lambda p: (p.parent.name, p.name))


def list_trajectories(root: PathLike, institution: Optional[str] = None) -> List[Trajectory]:
    """Like :func:`find_trajectory_dirs` but returns :class:`Trajectory` objects."""
    return [Trajectory(p) for p in find_trajectory_dirs(root, institution)]
