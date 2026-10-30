#!/usr/bin/env python3
"""Top-down comparison of the raw accumulated map and the ERASOR2 static map.

Raw map points with no static-map point within 10 cm (what ERASOR2 removed: mostly moving people)
are drawn in red. Only points 0.15-2 m above the floor and within 15 m of the robot path are shown. Run from the erasor2/ dir:  python plot_maps.py [output_dir] [--out static_map.png]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

Z_BAND = (0.15, 2.0)   # m above the floor: skip floor and ceiling
MARGIN = 15.0          # m around the robot path


def load(path, voxel=0.05):
    return np.asarray(o3d.io.read_point_cloud(str(path)).voxel_down_sample(voxel).points)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", nargs="?", default="output")
    ap.add_argument("--out", default="static_map.png")
    a = ap.parse_args()
    raw = load(Path("..") / "map.pcd")
    static = load(Path(a.output_dir) / "static_map_slam_frame.pcd")
    poses = np.loadtxt(Path("..") / "poses_base_link_tum.txt")[:, 1:4]
    lower = raw[raw[:, 2] < poses[:, 2].max(), 2]
    hist, edges = np.histogram(lower, bins=np.arange(lower.min(), lower.max() + 0.05, 0.05))
    floor = edges[hist.argmax()]   # the ground is the densest height level below the robot
    band = lambda p: p[(p[:, 2] > floor + Z_BAND[0]) & (p[:, 2] < floor + Z_BAND[1])]
    raw, static = band(raw), band(static)
    removed = raw[np.isinf(cKDTree(static).query(raw, distance_upper_bound=0.1)[0])]

    lo, hi = poses[:, :2].min(0) - MARGIN, poses[:, :2].max(0) + MARGIN
    keep = lambda p: p[((p[:, :2] >= lo) & (p[:, :2] <= hi)).all(axis=1)]
    raw, static, removed = keep(raw), keep(static), keep(removed)
    w, h = hi - lo
    fig, axes = plt.subplots(1, 2, figsize=(2 * 6 * w / max(w, h) + 1, 6 * h / max(w, h) + 1),
                             sharex=True, sharey=True)
    for ax, title in zip(axes, ("raw accumulated map (people included)", "static map (ERASOR2)")):
        ax.set_facecolor("white")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel("x (m)")
    axes[0].scatter(raw[:, 0], raw[:, 1], s=0.2, c="0.35", lw=0)
    axes[0].scatter(removed[:, 0], removed[:, 1], s=0.4, c="tab:red", lw=0, label="removed by ERASOR2")
    axes[1].scatter(static[:, 0], static[:, 1], s=0.2, c="0.35", lw=0)
    for ax in axes:
        ax.plot(poses[:, 0], poses[:, 1], "-", c="tab:blue", lw=1.5, label="robot path (SLAM)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="upper right", markerscale=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}: {len(raw)} raw / {len(static)} static points in the "
          f"{Z_BAND[0]}-{Z_BAND[1]} m band within {MARGIN:.0f} m of the path, {len(removed)} removed")


if __name__ == "__main__":
    main()
