"""Sensor calibration files of the human-readable data.

``sampled_data/extrinsics_intrinsics/<institution>/`` holds, depending on the
institution:

- ``_tf_static.txt``: all static transforms of the robot (sensor extrinsics)
- ``_<camera>_camera_info.txt`` or ``intrinsics.txt``: camera intrinsics
- ``static_transforms.yaml`` and ``body_to_camera_optical_frame/<trajectory>/extrinsics.txt``
  (NUS): robot transforms and a per-trajectory 4x4 body-to-camera transform
"""

import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .constants import CALIBRATION_DIR, INSTITUTION_ALIASES

PathLike = Union[str, Path]
TRANSFORM_COLUMNS = ["parent", "child", "x", "y", "z", "qx", "qy", "qz", "qw"]
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _floats(text: str) -> List[float]:
    return [float(v) for v in re.findall(_NUMBER, text)]


def calibration_dir(root: PathLike, institution: str) -> Optional[Path]:
    """Calibration folder of an institution (``root`` is the dataset root)."""
    base = Path(root) / CALIBRATION_DIR
    names = [institution] + [alias for alias, name in INSTITUTION_ALIASES.items() if name == institution]
    names += [INSTITUTION_ALIASES[institution]] if institution in INSTITUTION_ALIASES else []
    return next((base / n for n in names if (base / n).is_dir()), None)


def calibration_files(root: PathLike, institution: str, trajectory: Optional[str] = None) -> List[Path]:
    """Calibration files of an institution, plus the per-trajectory ones if ``trajectory`` is given.

    Files directly inside sub-folders (e.g. UBonn's ``<session>_tfstatic_0/_tf_static.txt``)
    are included; per-trajectory folders one level deeper (NUS) only for ``trajectory``.
    """
    folder = calibration_dir(root, institution)
    if folder is None:
        return []
    files = sorted(p for p in folder.iterdir() if p.is_file())
    files += sorted(p for p in folder.glob("*/*") if p.is_file())
    if trajectory is not None:
        files += sorted(folder.glob(f"*/{trajectory}/*"))
    return files


def read_camera_info_txt(path: PathLike) -> dict:
    """Read ``*_camera_info.txt`` (dump of a CameraInfo message) or ``intrinsics.txt``
    (9 comma-separated values of K).

    Returns a dict with ``K`` (3x3) and, when available, ``D``, ``width``,
    ``height``, ``distortion_model`` and ``frame_id``.
    """
    text = Path(path).read_text()
    if "K (intrinsic" not in text:
        return {"K": np.array(_floats(text), dtype=np.float64).reshape(3, 3)}
    info = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "frame_id":
            info["frame_id"] = value.strip()
        elif key.strip() == "size":
            info["width"], info["height"] = (int(v) for v in re.findall(r"\d+", value)[:2])
        elif key.strip() == "distortion_model":
            info["distortion_model"] = value.strip()
        elif key.strip().startswith("D ("):
            info["D"] = np.array(_floats(value), dtype=np.float64)
    k_block = text.split("K (intrinsic 3x3):", 1)[1].split("R (", 1)[0]
    info["K"] = np.array(_floats(k_block), dtype=np.float64).reshape(3, 3)
    return info


def read_tf_static_txt(path: PathLike) -> pd.DataFrame:
    """Read ``_tf_static.txt`` (``• parent -> child`` blocks with Translation and
    Rotation (Quat) lines) into ``parent, child, x, y, z, qx, qy, qz, qw`` rows."""
    rows, current = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip().lstrip("•").strip()
        if "->" in line:
            parent, child = (s.strip() for s in line.split("->", 1))
            current = [parent, child]
        elif current is not None and line.startswith("Translation"):
            current += _floats(line)[:3]
        elif current is not None and line.startswith("Rotation"):
            current += _floats(line.split(":", 1)[1])[:4]
            if len(current) == len(TRANSFORM_COLUMNS):
                rows.append(current)
            current = None
    return pd.DataFrame(rows, columns=TRANSFORM_COLUMNS)


def read_static_transforms_yaml(path: PathLike) -> pd.DataFrame:
    """Read ``static_transforms.yaml`` into ``parent, child, x, y, z, qx, qy, qz, qw`` rows."""
    with open(path) as f:
        entries = yaml.safe_load(f)["static_transforms"]
    return pd.DataFrame([[e["parent"], e["child"], *e["translation"], *e["rotation"]] for e in entries],
                        columns=TRANSFORM_COLUMNS)


def read_extrinsics_txt(path: PathLike) -> np.ndarray:
    """Read a 4x4 homogeneous transform stored as 16 numbers (row-major)."""
    return np.array(_floats(Path(path).read_text()), dtype=np.float64).reshape(4, 4)


def transform_matrix(row: pd.Series) -> np.ndarray:
    """4x4 homogeneous matrix (child frame to parent frame) of one transform row."""
    x, y, z, w = row["qx"], row["qy"], row["qz"], row["qw"]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [row["x"], row["y"], row["z"]]
    return T
