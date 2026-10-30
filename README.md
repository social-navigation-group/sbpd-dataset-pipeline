# ACME dataset tools

Code for **ACME**, a multi-cultural, multi-embodiment social navigation and pedestrian
trajectory dataset collected at 8 institutions in 5 countries with 7 robot platforms.

This repository contains:

- **`acme_dataset`**: a Python package and command line tool to download, load, explore,
  convert and analyse the dataset. ROS is not required.
- **Example notebooks** for each part of the dataset.
- **The pipeline that built the dataset**: synchronization, bag splitting, anonymization,
  trajectory labelling, scenario tagging and BEV region annotation tools.

The dataset is hosted on Hugging Face:
[`acmesocnavdataset/ACME-dataset`](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset).

## The dataset

ACME has three parts:

| Part | Folder | Content |
|---|---|---|
| Raw ROS 2 bags | `ROS2_Bags/<institution>/<bag>.zip` | One zipped ROS 2 Humble bag (sqlite3) per robot trajectory, with every recorded topic plus the anonymization outputs: pedestrian boxes `/tracks_*`, person masks `/masks_*` and body keypoints `/keypoints_*`. |
| Human-readable data | `sampled_data/<institution>/<trajectory>.zip` | The same trajectories resampled at 4 Hz into JPEG, PCD, CSV and JSON files, plus sensor calibration and robot utterances. |
| BEV | `BEV/` | Overhead-camera videos, and human-verified pedestrian tracks with pixel-to-metre homographies and semantic region tags. |

`acme_tags.xlsx` labels every robot trajectory with scenario tags (one sheet per institution).
`acme-dataset list <component>` shows which institutions are currently on Hugging Face.

| Institution folder | Institution | Robot |
|---|---|---|
| `go2nus` (`nus` in calibration and BEV) | NUS | Unitree GO2 |
| `miraikan` | Miraikan | Suitcase robot |
| `cmu` | CMU | Suitcase robot |
| `keio` | Keio | Suitcase robot |
| `umich` | UMich | Stretch 2 |
| `uex` | UEx | Shadow robot |
| `gmu` | GMU | Scout Mini |
| `ubonn` (`bonn` in BEV) | UBonn | Unitree GO1 |

<details>
<summary>Full folder layout</summary>

```
ACME-dataset/
├── acme_tags.xlsx
├── explore_dataset_human_readable.ipynb
├── ROS2_Bags/
│   └── <institution>/<bag>.zip                  -> <bag>/metadata.yaml, <bag>/*.db3
├── sampled_data/
│   ├── <institution>/<trajectory>.zip            -> <trajectory>/
│   │       ├── timestamps.txt                    UNIX time (s) of each sample
│   │       ├── traj_data.csv                     robot odometry per sample
│   │       ├── tracks.json                       pedestrian boxes + track ids per sample
│   │       ├── imgs/<i>.jpg                      RGB (imgs_<camera>/ on multi-camera robots)
│   │       ├── masks/<i>.jpg                     person masks, 255 = person (some institutions)
│   │       ├── pcd/<i>.pcd                       LiDAR points: x y z intensity
│   │       └── speech.txt                        robot speech / UI events (some institutions)
│   ├── extrinsics_intrinsics/<institution>/      _tf_static.txt, *_camera_info.txt / intrinsics.txt, ...
│   └── utterances/<institution>_utterances.csv
└── BEV/
    ├── BEV_videos/<institution>/*.MP4 | *.avi
    └── BEV_annotations/
        ├── explore_bev.ipynb
        ├── frames/<stem>.jpg                     BEV image the tags were drawn on
        ├── traj_frames/<stem>.jpg                same image with the tracks drawn in
        ├── annotations/<stem>.json               tag polygons
        ├── masks/<stem>/tag_<id>_<name>.jpg
        └── trajectory-transform/
            ├── trajectories/<dataset>/<session>.toml           pixel tracks, 10 FPS
            └── homographies/<dataset>/<session>_homography.txt  pixel -> metre
```

</details>

## Installation

```bash
git clone https://github.com/social-navigation-group/sbpd-dataset-pipeline.git
cd sbpd-dataset-pipeline
pip install -e .                 # add ".[notebooks]" for Jupyter
```

Python 3.9 or newer. The package installs the `acme-dataset` command.

## Quick start

### Download

Download one component, optionally restricted to an institution or to named items.
The Hugging Face folder layout is kept, so the loaders work directly on `--local-dir`
(default `ACME-dataset`). Human-readable zips are extracted after download.

| Component | Content | Size |
|---|---|---|
| `raw` | `ROS2_Bags/` | ~1.1 TB |
| `human_readable` | `sampled_data/` (includes calibration and utterances) | ~200 GB |
| `calibration`, `utterances` | `sampled_data/extrinsics_intrinsics/`, `sampled_data/utterances/` | < 10 MB |
| `bev_annotations` | `BEV/BEV_annotations/` | ~1 GB |
| `bev_videos` | `BEV/BEV_videos/` | ~180 GB |
| `bev` | both BEV folders | ~180 GB |
| `tags` | `acme_tags.xlsx` | 0.1 MB |

```bash
acme-dataset list human_readable                       # institutions
acme-dataset list human_readable -i miraikan           # trajectories of one institution
acme-dataset download tags
acme-dataset download human_readable -i keio           # one institution
acme-dataset download human_readable -i miraikan -n 03-08-2025-1-3F_Geo_Cosmos_0_0_10_0
acme-dataset download raw -i miraikan -n 03-08-2025-1-3F_Geo_Cosmos_0_0_10
acme-dataset download calibration
acme-dataset download bev_annotations
```

```python
from acme_dataset import download, list_remote
names = list_remote("human_readable", institution="miraikan")
download("human_readable", "miraikan", names[:5], local_dir="ACME-dataset")
```

### Human-readable data

```python
from acme_dataset import list_trajectories

traj = list_trajectories("ACME-dataset", institution="miraikan")[0]
traj.odometry          # DataFrame: timestamp, position_x, position_y, yaw, linear_vel_x, linear_vel_y, angular_vel
traj.detections        # DataFrame: frame_idx, timestamp, track_id, x, y, width, height
traj.speech            # DataFrame: timestamp, text
traj.image(0)          # (H, W, 3) uint8 RGB
traj.mask(0)           # (H, W) bool person mask
traj.pointcloud(0)     # (N, 4) float32 x, y, z, intensity
for frame in traj:     # all modalities of one sample
    ...
```

Calibration and utterances:

```python
from acme_dataset.calibration import calibration_files, read_camera_info_txt, read_tf_static_txt
from acme_dataset.human_readable import load_utterances

calibration_files("ACME-dataset", "cmu")                  # _tf_static.txt, *_camera_info.txt, ...
calib = "ACME-dataset/sampled_data/extrinsics_intrinsics/cmu"
K = read_camera_info_txt(f"{calib}/_zed_zed_node_left_camera_info.txt")["K"]
extrinsics = read_tf_static_txt(f"{calib}/_tf_static.txt")  # parent, child, x, y, z, qx, qy, qz, qw
utterances = load_utterances("ACME-dataset")              # institution, trajectory, utterance, timestamp
```

### Raw ROS 2 bags (no ROS installation needed)

```python
from acme_dataset.rosbag import Bag, decode_detections, detect_bbox_encoding

with Bag("ACME-dataset/ROS2_Bags/miraikan/<bag>.zip") as bag:   # extracts the zip on first use
    print(bag.topics)                                            # topic, msgtype, count
    odom = bag.odometry("/odom")
    encoding = detect_bbox_encoding(bag, "/tracks_d455_rs1_color_image_raw_compressed")
    for topic, t_ns, msg in bag.messages(["/tracks_d455_rs1_color_image_raw_compressed"]):
        boxes, track_ids = decode_detections(msg, encoding)     # top-left x, y, w, h
```

The `vision_msgs`, ACME `custom_msgs` (`BinaryMask`, `KeypointArray`) and `map_msgs`
definitions are registered automatically.

> **Box centers in bags.** Depending on the version of the anonymization pipeline that
> wrote a bag, `bbox.center` of the `/tracks_*` messages is either the true box center
> or `(top_left + size) / 2`. Use `detect_bbox_encoding` (it checks the boxes against the
> person masks, or against the image width) and pass the result to `decode_detections`.
> The human-readable `tracks.json` files store top-left corners.

Re-sample bags into the human-readable format, at any rate:

```bash
acme-dataset convert ACME-dataset/ROS2_Bags/miraikan --out converted/miraikan --rate 4
```

The output has the layout of the extracted `sampled_data` zips. The topics used per
institution are in [`acme_dataset/configs/topics.yaml`](acme_dataset/configs/topics.yaml).

### BEV pedestrian trajectories

```python
from acme_dataset import BEVSession, list_bev_sessions

print(list_bev_sessions("ACME-dataset/BEV"))           # dataset, session
session = BEVSession("ACME-dataset/BEV", "bonn", "20")
df = session.trajectories()                            # metres, filtered, 10 FPS
df = session.trajectories(target_fps=2.5)              # as transform_trajectories.py
segments = session.tag_segments(df)                    # split by semantic region
```

`trajectories()` returns one row per pedestrian and frame, with columns `ped_id, frame, time_s,
x, y` (metres), `px, py` (BEV pixels) and `label` (`dynamic`, `stationary` or
`out_of_bounds`). The filter is the one shipped in `BEV_annotations/trajectory-transform/trajectory_filter.py`:
it drops tracks shorter than 3.2 s, interpolates annotation jumps, smooths over 5 frames,
labels points whose local speed is below 0.5 m/s as stationary and points more than 25 m from
the homography origin as out of bounds. BEV dataset names are `bonn`, `cmu`, `gmu`, `keio`, `nus`,
`short-term` (Miraikan) and `umich`. Export every session to CSV with:

```bash
acme-dataset bev-export ACME-dataset/BEV --out bev_csv --target-fps 2.5
acme-dataset bev-export ACME-dataset/BEV --out bev_csv_tags --tags
```

### Scenario tags and statistics

```python
from acme_dataset import load_scenario_tags, filter_by_tags, tags_for, dataset_stats, summarize

tags = load_scenario_tags("ACME-dataset/acme_tags.xlsx")
narrow = filter_by_tags(tags, all_of=["narrow_corridors"], any_of=["against_traffic", "large_crowds"])
tags_for(tags, traj.name)                  # accepts tag-sheet, bag or sampled_data names
stats = dataset_stats("ACME-dataset")      # per trajectory: duration, distance, speed, pedestrians, ...
summarize(stats)                           # per institution
```

```bash
acme-dataset info ACME-dataset                       # what is on disk
acme-dataset info ACME-dataset/ROS2_Bags/miraikan/<bag>.zip   # topics of a bag, read from metadata.yaml
acme-dataset stats ACME-dataset --out stats.csv
```

## Notebooks

| Notebook | Content |
|---|---|
| [`01_human_readable_data.ipynb`](notebooks/01_human_readable_data.ipynb) | odometry, images with tracks, person masks, point clouds, speech, calibration, scenario tags, statistics |
| [`02_raw_ros2_bags.ipynb`](notebooks/02_raw_ros2_bags.ipynb) | topics, odometry, images, boxes, masks, keypoints, LiDAR, calibration, conversion |
| [`03_bev_trajectories.ipynb`](notebooks/03_bev_trajectories.ipynb) | metric BEV tracks, filtering, semantic regions, walking speed, closest approach |

Set `ACME_ROOT` to the dataset folder (default `../ACME-dataset`) before starting Jupyter.

## Data format reference

| File | Format |
|---|---|
| `timestamps.txt` | One UNIX timestamp (seconds) per sample. Sample `i` of every file below refers to this time. |
| `traj_data.csv` | `frame_idx, position_x, position_y, yaw, linear_vel_x, linear_vel_y, angular_vel` from the robot odometry. Positions in metres in the odometry frame, yaw in radians, velocities from the odometry twist. |
| `tracks.json` | `[{frame_idx, detections: [{track_id, bbox: {x, y, width, height}}]}]`. Boxes in image pixels, `(x, y)` is the top-left corner (it can be negative for people at the image border). Track ids are consistent within a trajectory. |
| `imgs/<i>.jpg` | Egocentric RGB image (anonymized). |
| `masks/<i>.jpg` | Person segmentation mask, 255 = person. JPEG compressed, so threshold at 128. Only for some institutions. |
| `pcd/<i>.pcd` | LiDAR point cloud in the LiDAR frame, ASCII PCD with fields `x y z intensity`. |
| `speech.txt` | `<UNIX time>: <text>` per line: robot utterances, or robot UI events for the suitcase robots. |
| `extrinsics_intrinsics/<institution>/_tf_static.txt` | Static transforms: `• parent -> child` blocks with `Translation: [x, y, z]` and `Rotation (Quat): [qx, qy, qz, qw]`. |
| `extrinsics_intrinsics/<institution>/*_camera_info.txt`, `intrinsics.txt` | Camera intrinsics: a dump of the `CameraInfo` message (size, D, K, R, P), or the 9 values of K. |
| `utterances/<institution>_utterances.csv` | `utterance, trajectory, timestamps` (list of UNIX times). |
| BEV `<session>.toml` | `[human<N>]` tables with `traj_start` (first frame) and `trajectories` (pixel `[x, y]` per frame at 10 FPS). |
| BEV `<session>_homography.txt` | 3x3 matrix `H`, `[x_m, y_m, w] = H [x_px, y_px, 1]`. |
| BEV `annotations/<stem>.json` | `[{tag_id, tag_name, points: [[x, y], ...]}]` polygons. Tag ids: 1 Open Area, 2 Wide Corridor, 3 Narrow Corridor, 4 Intersection, 5 Entry/Exit, 6 Blind Corner, 7 Ignore. |

## Repository layout

| Folder | Purpose |
|---|---|
| [`acme_dataset/`](acme_dataset) | Python package and `acme-dataset` CLI |
| [`notebooks/`](notebooks) | Example notebooks |
| [`tests/`](tests) | Unit tests on synthetic data (`pytest`); `ACME_ROOT=... pytest tests/test_sample_data.py` checks a real download |
| [`static_map_extraction/`](static_map_extraction) | Static 3D maps with moving people removed, from a raw bag: offline LiDAR SLAM (KISS-ICP + GTSAM) + ERASOR2 |

Tools used to build the dataset, in pipeline order:

| Folder | Purpose |
|---|---|
| [`synchronization/`](synchronization) | Time and position synchronization between robot bags and BEV videos (QR-code / AprilTag based) |
| [`split-bag/`](split-bag) | Split recordings into trajectories using the start / end / discard signals |
| [`anonymization-bag/`](anonymization-bag) | Anonymize camera topics in bags and add pedestrian tracks, masks and keypoints |
| [`anonymization/`](anonymization) | Anonymize videos (FPV or BEV) and track pedestrians |
| [`labelling-tool/`](labelling-tool) | GUI to verify and correct pedestrian trajectories |
| [`robot-action-annotation-tool/`](robot-action-annotation-tool) | Robot action annotation and scenario tagging (VIA based) |
| [`bev_semantic_annotation_tool/`](bev_semantic_annotation_tool) | GUI to draw semantic regions on BEV images |
| [`benchmarks/move_base_experiments/`](benchmarks/move_base_experiments) | Replay bags through a geometric planner (move_base) to measure planner failures |

Each folder has its own README with setup instructions.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md): commits must be signed off (`git commit -s`).

## Citation

If you use ACME, please cite *ACME: A Multi-Cultural Multi-Embodiment Social Navigation Dataset*
(ACME Collaboration, under review).

## License

The code in this repository is released under the [Apache 2.0 License](LICENSE).
See the Hugging Face dataset card for the license of the data.
