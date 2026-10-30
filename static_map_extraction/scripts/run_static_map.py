#!/usr/bin/env python3
"""Static point-cloud map of one ACME recording: offline LiDAR SLAM + ERASOR2 dynamic-object removal.

Per bag (folder with metadata.yaml + .db3, or the .zip from ROS2_Bags/<institution>/):
  1. run_slam.py         KISS-ICP + GTSAM offline SLAM -> poses, raw accumulated map   (SLAM env)
  2. export_scans.py     scans + poses in ERASOR2's layout                            (SLAM env)
  3. gen_labels.py       Patchwork++ ground + HDBSCAN instances per scan              (ERASOR2 env)
  4. run_erasor2         ERASOR2 dynamic-object removal                               (C++ binary)
  5. finalize_map.py     static map back in the SLAM frame + stats                    (ERASOR2 env)
  6. plot_maps.py        top-down figure: raw map vs static map                       (ERASOR2 env)

Outputs in <out>/<bag>/slam_output/:
  map.pcd, map_full_res.pcd              raw accumulated map (moving people included), SLAM `map` frame
  poses_*_tum.txt, slam_qc.json          trajectories, SLAM-vs-odometry check
  erasor2/output/static_map_slam_frame.pcd   static map, SLAM `map` frame
  erasor2/summary.json, static_map.png   stats and figure

Usage:
  python run_static_map.py ACME-dataset/ROS2_Bags/miraikan/<bag>.zip --profile miraikan --out maps/ \
      --erasor2 ~/ERASOR2/build/run_erasor2 [--slam-python PY] [--labels-python PY] [--workers 8]
Every step is skipped when its output exists (rerun after a failure to resume).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_profile as DP  # noqa: E402

SLAM_SCRIPTS = ("dataset_profile.py", "run_slam.py")
ERASOR2_SCRIPTS = ("export_scans.py", "gen_labels.py", "finalize_map.py", "plot_maps.py")
MIN_SCANS = 30


def resolve_bag(path: Path, out: Path) -> Path:
    """Bag folder for a folder or a zip (extracted under <out>/bags/)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            meta = next(n for n in zf.namelist() if n.endswith("metadata.yaml"))
            bag = out / "bags" / Path(meta).parent
            if not (bag / "metadata.yaml").exists():
                zf.extractall(out / "bags")
        return bag
    if (path / "metadata.yaml").exists():
        return path
    raise SystemExit(f"no ROS 2 bag at {path}")


def write_erasor2_yaml(e: Path, n_scans: int, prof: dict) -> Path:
    """Fill the ERASOR2 template for this bag (frame range, paths, heights from the profile)."""
    t = (HERE / "erasor2_template.yaml").read_text()
    h = prof["lidar_height"]   # heights in the yaml are in the lidar frame
    t = re.sub(r"end_frame: \d+.*", f"end_frame: {n_scans - 3}", t)
    t = re.sub(r"min_z_voi: [-\d.]+", f"min_z_voi: {-(h + 0.62):.3f}", t)   # ground must lie inside the VoI
    t = re.sub(r"max_z_voi: [-\d.]+", f"max_z_voi: {DP.CEIL_Z - h:.3f}", t)
    t = re.sub(r"sensor_height: [-\d.]+", f"sensor_height: {h}", t)
    t = t.replace("rerun:\n  enabled: true", "rerun:\n  enabled: false")
    t = t.replace("<slam_output>/erasor2", str(e))
    path = e / "erasor2.yaml"
    path.write_text(t)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", type=Path, help="bag folder or .zip")
    ap.add_argument("--profile", required=True, choices=sorted(DP.PROFILES), help="robot/sensor setup")
    ap.add_argument("--out", type=Path, required=True, help="output root; results in <out>/<bag>/slam_output/")
    ap.add_argument("--erasor2", default=os.environ.get("ERASOR2_BIN", "run_erasor2"),
                    help="path to the run_erasor2 binary (default: $ERASOR2_BIN or run_erasor2 on PATH)")
    ap.add_argument("--slam-python", default=sys.executable,
                    help="python with kiss-icp, gtsam, open3d, rosbags (default: this python)")
    ap.add_argument("--labels-python", default=sys.executable,
                    help="python with pypatchworkpp, hdbscan, open3d, matplotlib (default: this python)")
    ap.add_argument("--workers", type=int, default=8, help="processes for gen_labels.py")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    bag = resolve_bag(a.bag.resolve(), a.out.resolve())
    out = a.out.resolve() / bag.name / "slam_output"
    e = out / "erasor2"
    e.mkdir(parents=True, exist_ok=True)
    prof = DP.write(out, a.profile)
    for name in SLAM_SCRIPTS:
        shutil.copy(HERE / name, out / name)
    for name in ERASOR2_SCRIPTS:
        shutil.copy(HERE / name, e / name)

    def run(cmd, cwd, **env):
        print(f"$ {' '.join(map(str, cmd))}", flush=True)
        subprocess.run([str(c) for c in cmd], cwd=cwd, check=True,
                       env={**os.environ, "PYTHONPATH": str(cwd), **env})

    if not (out / "poses_rslidar_tum.txt").exists():
        run([a.slam_python, "run_slam.py", bag], out)
    if not (e / "data/Merged/poses.txt").exists():
        run([a.slam_python, "export_scans.py", bag], e)
    n = len(list((e / "data/Merged/velodyne").glob("*.bin")))
    if n < MIN_SCANS:
        raise SystemExit(f"only {n} scans: too short for ERASOR2")
    if len(list((e / "data/Merged/hdbscan").glob("*.label"))) != n:
        run([a.labels_python, "gen_labels.py", "--workers", a.workers], e, OMP_NUM_THREADS="1")
    if not (e / "output/static_map_slam_frame.pcd").exists():
        (e / "output").mkdir(exist_ok=True)
        run([a.erasor2, write_erasor2_yaml(e, n, prof)], e)
        run([a.labels_python, "finalize_map.py", "output", "--rm-lifted"], e)
    run([a.labels_python, "plot_maps.py"], e)
    print(f"static map: {e / 'output/static_map_slam_frame.pcd'}")


if __name__ == "__main__":
    main()
