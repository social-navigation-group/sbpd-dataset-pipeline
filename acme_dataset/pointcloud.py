"""Read and write point clouds in the formats used by the human-readable data."""

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

_PCD_TYPES = {("F", 4): np.float32, ("F", 8): np.float64,
              ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32,
              ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32}


def read_npz(path: PathLike) -> np.ndarray:
    """Load an ``(N, 4)`` float32 x, y, z, intensity array from a ``.npz`` file."""
    with np.load(path) as data:
        return data[data.files[0]]


def read_pcd(path: PathLike) -> np.ndarray:
    """Load an ASCII or binary ``.pcd`` file into an ``(N, C)`` float32 array.

    Columns follow the ``FIELDS`` line of the header (``x y z intensity`` for
    ACME data). Compressed binary PCD files are not supported.
    """
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing DATA line in PCD header")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.split()
            if key.upper() == "DATA":
                break
        body = f.read()

    fields = header["FIELDS"]
    num_points = int(header["POINTS"][0])
    data_format = header["DATA"][0]
    if data_format == "ascii":
        points = np.loadtxt(body.decode("ascii").splitlines(), dtype=np.float32, ndmin=2)
        return points.reshape(-1, len(fields))
    if data_format == "binary":
        dtype = np.dtype([(name, _PCD_TYPES[(t, int(s))])
                          for name, t, s in zip(fields, header["TYPE"], header["SIZE"])])
        records = np.frombuffer(body, dtype=dtype, count=num_points)
        return np.stack([records[name].astype(np.float32) for name in fields], axis=1)
    raise ValueError(f"{path}: unsupported PCD DATA format '{data_format}'")


def write_pcd(points: np.ndarray, path: PathLike, binary: bool = False) -> Path:
    """Write an ``(N, 3)`` or ``(N, 4)`` array as a PCD v0.7 file."""
    points = np.asarray(points, dtype=np.float32)
    fields = "x y z intensity" if points.shape[1] >= 4 else "x y z"
    points = points[:, :4] if points.shape[1] >= 4 else points[:, :3]
    n_cols = points.shape[1]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {' '.join(['4'] * n_cols)}\n"
        f"TYPE {' '.join(['F'] * n_cols)}\n"
        f"COUNT {' '.join(['1'] * n_cols)}\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )
    path = Path(path)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if binary:
            f.write(points.tobytes())
        else:
            np.savetxt(f, points, fmt="%.9g")
    return path
