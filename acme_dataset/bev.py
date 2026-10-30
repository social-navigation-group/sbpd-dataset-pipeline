"""Load BEV (overhead camera) pedestrian annotations.

A BEV *session* is one overhead video annotated with pedestrian tracks. It is
identified by a dataset name (e.g. ``bonn``, ``nus``, ``cmu``) and a session
id (the stem of its trajectory file, e.g. ``20`` or ``GX010036_BEV1_DOW``).
Its files live under ``BEV/BEV_annotations``::

    frames/<stem>.jpg                 BEV image the semantic tags were drawn on
    traj_frames/<stem>.jpg            same view with annotated tracks drawn in
    annotations/<stem>.json           tag polygons [{tag_id, tag_name, points}]
    masks/<stem>/tag_<id>_<name>.jpg  one binary mask per tag (255 inside)
    trajectory-transform/
        trajectories/<dataset>/<session>.toml           pixel tracks at 10 FPS
        homographies/<dataset>/<session>_homography.txt  3x3 pixel -> metre
        trajectories/<dataset>/intrinsics-<dataset>.txt  lens model (CMU only)

where ``<stem>`` is usually ``<dataset>_<session>_transformed``. Files are
located by name anywhere below the given root, so per-dataset sub-folders work
as well.
"""

import json
import os
import sys
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .bev_filter import filter_and_downsample, filter_trajectory, LABEL_DYNAMIC
from .constants import BEV_ANNOTATIONS_DIR, BEV_FPS, BEV_IGNORE_TAG, BEV_TAGS

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PathLike = Union[str, Path]
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------- homography
def read_homography(path: PathLike) -> np.ndarray:
    """Read a 3x3 pixel-to-metre homography (three rows of three floats)."""
    rows = [[float(v) for v in line.split()] for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if len(r) == 3]
    if len(rows) != 3:
        raise ValueError(f"{path}: expected 3 rows of 3 values")
    return np.array(rows, dtype=np.float64)


def read_intrinsics(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read ``(camera_matrix, dist_coeffs)``: three matrix rows then one line of
    distortion coefficients (k1 k2 p1 p2 k3), brackets optional."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    parse = lambda line: [float(v) for v in line.replace("[", " ").replace("]", " ").replace(",", " ").split()]
    camera_matrix = np.array([parse(line) for line in lines[:3]], dtype=np.float64)
    return camera_matrix, np.array(parse(lines[3]), dtype=np.float64)


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` points through a 3x3 homography."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ H.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Remove lens distortion from pixel coordinates (result stays in pixels)."""
    import cv2

    src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(src, camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 2)


def distort_points(points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Apply the plumb-bob lens model to undistorted pixel coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs[:5]
    x, y = (points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([xd * fx + cx, yd * fy + cy], axis=1)


# -------------------------------------------------------------------- index
class BEVIndex:
    """File index of a BEV annotation tree, built with a single walk."""

    def __init__(self, root: PathLike):
        self.root = Path(root)
        for sub in (BEV_ANNOTATIONS_DIR, "BEV_annotations", "bev/bev_annotations", "bev_annotations"):
            if (self.root / sub).is_dir():
                self.root = self.root / sub
                break
        self.frames: Dict[str, Path] = {}
        self.traj_frames: Dict[str, Path] = {}
        self.annotations: Dict[str, Path] = {}
        self.masks: Dict[str, Path] = {}
        self.trajectories: Dict[Tuple[str, str], Path] = {}
        self.homographies: Dict[str, List[Path]] = defaultdict(list)
        self.intrinsics: Dict[str, Path] = {}
        self.videos: List[Path] = []
        self._build()

    def _build(self) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
            here = Path(dirpath)
            if here.name == "masks":
                for d in dirnames:
                    self.masks[d] = here / d
            for name in filenames:
                path = here / name
                suffix = path.suffix.lower()
                if here.name == "frames" and suffix in (".jpg", ".png"):
                    self.frames[path.stem] = path
                elif here.name == "traj_frames" and suffix in (".jpg", ".png"):
                    self.traj_frames[path.stem] = path
                elif here.name == "annotations" and suffix == ".json":
                    self.annotations[path.stem] = path
                elif name.endswith("_homography.txt"):
                    self.homographies[name[: -len("_homography.txt")]].append(path)
                elif name.startswith("intrinsics-") and suffix == ".txt":
                    self.intrinsics[path.stem[len("intrinsics-"):]] = path
                elif suffix == ".toml" and "trajectories" in path.relative_to(self.root).parts:
                    parts = path.relative_to(self.root).parts
                    after = parts[parts.index("trajectories") + 1:]
                    dataset = after[0] if len(after) >= 2 else path.parent.parent.name
                    self.trajectories[(dataset, path.stem)] = path
                elif suffix in _VIDEO_SUFFIXES:
                    self.videos.append(path)

    def sessions(self) -> pd.DataFrame:
        """One row per annotated session: ``dataset, session, toml``."""
        rows = [(d, s, p) for (d, s), p in sorted(self.trajectories.items())]
        return pd.DataFrame(rows, columns=["dataset", "session", "toml"])

    def find_stem(self, dataset: str, session: str) -> Optional[str]:
        """Image/annotation stem of a session (handles the naming variants used per dataset)."""
        stems = set(self.frames) | set(self.annotations) | set(self.masks)
        exact = f"{dataset}_{session}_transformed"
        if exact in stems:
            return exact
        matches = sorted(s for s in stems if s.startswith(f"{dataset}_") and s.endswith("_transformed")
                         and (s.startswith(f"{dataset}_{session}_") or s.endswith(f"_{session}_transformed")))
        return matches[0] if matches else None

    def find_homography(self, dataset: str, session: str) -> Optional[Path]:
        paths = self.homographies.get(session, [])
        in_dataset = [p for p in paths if dataset in p.parts]
        return (in_dataset or paths or [None])[0]


# ------------------------------------------------------------------ session
class BEVSession:
    """Annotations of one BEV session.

    Example:
        >>> session = BEVSession("ACME-dataset/BEV", "bonn", "20")
        >>> df = session.trajectories()               # metric, filtered, 10 FPS
        >>> df = session.trajectories(target_fps=2.5) # transform_trajectories.py output
        >>> tags = session.tag_segments(df)           # split by semantic region
    """

    def __init__(self, root: Union[PathLike, BEVIndex], dataset: str, session: str):
        self.index = root if isinstance(root, BEVIndex) else BEVIndex(root)
        self.dataset, self.session = dataset, str(session)
        if (dataset, self.session) not in self.index.trajectories:
            raise KeyError(f"no trajectory file for dataset={dataset!r} session={session!r}")
        self.toml_path = self.index.trajectories[(dataset, self.session)]
        self.stem = self.index.find_stem(dataset, self.session)
        self.homography_path = self.index.find_homography(dataset, self.session)
        self.intrinsics_path = self.index.intrinsics.get(dataset)

    def __repr__(self) -> str:
        return f"BEVSession({self.dataset}/{self.session})"

    # -------------------------------------------------------------- images
    def _image(self, table: Dict[str, Path]) -> Optional[np.ndarray]:
        path = table.get(self.stem) if self.stem else None
        return np.array(Image.open(path).convert("RGB")) if path else None

    def frame(self) -> Optional[np.ndarray]:
        """BEV image (RGB) on which the semantic tags were drawn."""
        return self._image(self.index.frames)

    def traj_frame(self) -> Optional[np.ndarray]:
        """BEV image with the annotated trajectories drawn in."""
        return self._image(self.index.traj_frames)

    def annotations(self) -> List[dict]:
        """Semantic tag polygons ``[{tag_id, tag_name, points: [[x, y], ...]}]`` in pixels."""
        path = self.index.annotations.get(self.stem) if self.stem else None
        if path is None:
            return []
        with open(path) as f:
            return json.load(f)

    def masks(self) -> Dict[int, np.ndarray]:
        """``{tag_id: boolean mask}`` (JPEG masks thresholded at 128)."""
        folder = self.index.masks.get(self.stem) if self.stem else None
        if folder is None:
            return {}
        out = {}
        for path in sorted(folder.glob("tag_*.*")):
            out[int(path.stem.split("_", 2)[1])] = np.array(Image.open(path).convert("L")) > 128
        return out

    def tag_names(self) -> Dict[int, str]:
        names = dict(BEV_TAGS)
        names.update({a["tag_id"]: a["tag_name"] for a in self.annotations()})
        return names

    # ------------------------------------------------------------ geometry
    @cached_property
    def homography(self) -> np.ndarray:
        if self.homography_path is None:
            raise FileNotFoundError(f"no homography for {self}")
        return read_homography(self.homography_path)

    @cached_property
    def intrinsics(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(camera_matrix, dist_coeffs)`` if the BEV camera has a lens model (CMU)."""
        return read_intrinsics(self.intrinsics_path) if self.intrinsics_path else None

    def pixel_to_metric(self, points: np.ndarray) -> np.ndarray:
        """Pixel ``(N, 2)`` to metres (undistorting first when intrinsics exist)."""
        if self.intrinsics is not None:
            points = undistort_points(points, *self.intrinsics)
        return apply_homography(points, self.homography)

    def metric_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Metres ``(N, 2)`` to pixels of the (distorted) BEV image."""
        pixels = apply_homography(points, np.linalg.inv(self.homography))
        if self.intrinsics is not None:
            pixels = distort_points(pixels, *self.intrinsics)
        return pixels

    # -------------------------------------------------------- trajectories
    @cached_property
    def raw(self) -> dict:
        """Parsed trajectory TOML (``human<N>`` keys with ``traj_start`` and ``trajectories``)."""
        with open(self.toml_path, "rb") as f:
            return tomllib.load(f)

    def pixel_trajectories(self) -> Dict[str, Tuple[int, np.ndarray]]:
        """``{pedestrian_id: (start_frame, (N, 2) pixel array)}`` at 10 FPS."""
        out = {}
        for key, value in self.raw.items():
            if not key.startswith("human") or "trajectories" not in value:
                continue
            points = np.asarray(value["trajectories"], dtype=np.float64).reshape(-1, 2)
            out[key] = (int(value.get("traj_start", 0)), points)
        return out

    def trajectories(
        self,
        filtered: bool = True,
        target_fps: Optional[float] = None,
        fps: float = BEV_FPS,
        **filter_kwargs,
    ) -> pd.DataFrame:
        """Metric pedestrian trajectories, one row per (pedestrian, frame).

        Args:
            filtered: Apply :func:`acme_dataset.bev_filter.filter_trajectory`
                (drop short/static tracks, fix jumps, smooth, label points).
            target_fps: If given, downsample with
                :func:`acme_dataset.bev_filter.filter_and_downsample` instead.
            fps: Annotation frame rate.
            **filter_kwargs: Forwarded to the filter (e.g. ``coordinate_bounds=(-25, 25)``).

        Returns:
            DataFrame with columns ``ped_id, frame, time_s, x, y, px, py, label``
            where ``x, y`` are metres, ``px, py`` the matching BEV pixel
            coordinates and ``label`` one of dynamic / stationary / out_of_bounds.
        """
        frames = []
        for ped_id, (start, pixels) in self.pixel_trajectories().items():
            if len(pixels) < 2:
                continue
            metric = self.pixel_to_metric(pixels)
            offsets = np.arange(len(metric))
            labels = np.full(len(metric), LABEL_DYNAMIC, dtype=object)
            if target_fps is not None:
                result = filter_and_downsample(metric, fps, target_fps, **filter_kwargs)
                if result is None:
                    continue
                metric, labels, offsets = result
            elif filtered:
                result = filter_trajectory(metric, fps, **filter_kwargs)
                if result is None:
                    continue
                metric, labels = result
            pixels = self.metric_to_pixel(metric) if (filtered or target_fps) else pixels
            frame = start + offsets
            frames.append(pd.DataFrame({
                "ped_id": ped_id, "frame": frame, "time_s": frame / fps,
                "x": metric[:, 0], "y": metric[:, 1], "px": pixels[:, 0], "py": pixels[:, 1],
                "label": labels,
            }))
        columns = ["ped_id", "frame", "time_s", "x", "y", "px", "py", "label"]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    def tag_segments(self, trajectories: Optional[pd.DataFrame] = None, min_points: int = 2) -> pd.DataFrame:
        """Split trajectories into contiguous runs inside each semantic tag region.

        Tag regions may overlap, so a point can belong to segments of several
        tags. The Ignore tag is skipped.

        Returns ``trajectories`` rows (repeated per tag) with extra columns
        ``tag_id, tag_name, segment``; ``segment`` is unique within the session.
        """
        df = self.trajectories() if trajectories is None else trajectories
        masks = self.masks()
        names = self.tag_names()
        columns = list(df.columns) + ["tag_id", "tag_name", "segment"]
        if df.empty or not masks:
            return pd.DataFrame(columns=columns)
        h, w = next(iter(masks.values())).shape
        cols = np.clip(np.round(df["px"].to_numpy()).astype(int), 0, w - 1)
        rows = np.clip(np.round(df["py"].to_numpy()).astype(int), 0, h - 1)
        ped = df["ped_id"].to_numpy()
        out, segment = [], 0
        for tag_id, mask in sorted(masks.items()):
            if tag_id == BEV_IGNORE_TAG:
                continue
            inside = mask[rows, cols]
            # a new run starts where inside turns on or the pedestrian changes
            starts = inside & ~np.concatenate([[False], inside[:-1] & (ped[1:] == ped[:-1])])
            run_id = np.cumsum(starts)
            for run in np.unique(run_id[inside]):
                idx = np.nonzero(inside & (run_id == run))[0]
                if len(idx) < min_points:
                    continue
                part = df.iloc[idx].copy()
                part["tag_id"], part["tag_name"], part["segment"] = tag_id, names.get(tag_id, str(tag_id)), segment
                out.append(part)
                segment += 1
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def list_bev_sessions(root: PathLike) -> pd.DataFrame:
    """All annotated BEV sessions below ``root`` (dataset root, ``BEV/`` or ``BEV/BEV_annotations/``)."""
    return BEVIndex(root).sessions()


def find_bev_videos(root: PathLike) -> List[Path]:
    """BEV videos below ``root`` (e.g. ``BEV/BEV_videos/<institution>/*.MP4``)."""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _VIDEO_SUFFIXES)
