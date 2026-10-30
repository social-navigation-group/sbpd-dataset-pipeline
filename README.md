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
| Raw ROS 2 bags | `ros2 bags/<institution>/<bag>.zip` | One zipped ROS 2 Humble bag (sqlite3) per robot trajectory, with every recorded topic plus the anonymization outputs: pedestrian boxes `/tracks_*`, person masks `/masks_*` and body keypoints `/keypoints_*`. |
| Human-readable data | `human_readable_sampled/<institution>/<trajectory>/` | The same trajectories resampled at 4 Hz into JPEG, PCD, CSV and JSON files. |
| BEV annotations | `bev/` | Overhead-camera videos with human-verified pedestrian tracks, pixel-to-metre homographies and semantic region tags. |

`dataset_tags.xlsx` labels every robot trajectory with scenario tags (one sheet per institution).

| Institution folder | Institution | Robot |
|---|---|---|
| `go2nus` | NUS | Unitree GO2 |
| `miraikan` | Miraikan | Suitcase robot |
| `cmu` | CMU | Suitcase robot |
| `keio` | Keio | Suitcase robot |
| `umich` | UMich | Stretch 2 |
| `uex` | UEx | Shadow robot |
| `gmu` | GMU | Scout Mini |
| `ubonn` | UBonn | Unitree GO1 |

<details>
<summary>Full folder layout</summary>

```
ACME-dataset/
├── dataset_tags.xlsx
├── ros2 bags/
│   └── <institution>/<bag>.zip            -> <bag>/metadata.yaml, <bag>/*.db3
├── human_readable_sampled/
│   └── <institution>/<trajectory>/
│       ├── timestamps.txt                  UNIX time (s) of each sample
│       ├── traj_data.csv                   robot odometry per sample
│       ├── tracks.json                     pedestrian boxes + track ids per sample
│       ├── imgs/<i>.jpg                    RGB (imgs_<camera>/ on multi-camera robots)
│       ├── masked_imgs/<i>.jpg             person masks (255 = person)
│       ├── pcd/<i>.pcd, pcd/<i>.npz        LiDAR points: x y z intensity
│       ├── speech.txt                      robot speech "<time>: <text>" (some institutions)
│       └── traj_data.pkl, tracks.pkl       pickled copies of the CSV/JSON files
└── bev/
    ├── explore_bev.ipynb
    ├── bev_videos/<institution>/...
    └── bev_annotations/
        ├── frames/<stem>.jpg               BEV image the tags were drawn on
        ├── traj_frames/<stem>.jpg          same image with the tracks drawn in
        ├── annotations/<stem>.json         tag polygons
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
The Hugging Face folder layout is kept, so the loaders work directly on `--local-dir`.

```bash
acme-dataset list human_readable                       # institutions
acme-dataset list human_readable -i go2nus             # trajectories of one institution
acme-dataset download tags
acme-dataset download human_readable -i go2nus
acme-dataset download human_readable -i go2nus -n deck_1_without_overhead_2025-03-26_12-05-52_3_corrected_merged_0
acme-dataset download raw -i miraikan --extract        # unzip the bags after download
acme-dataset download bev
```

```python
from acme_dataset import download
download("human_readable", institution="go2nus", local_dir="ACME-dataset")
```

### Human-readable data

```python
from acme_dataset import list_trajectories

traj = list_trajectories("ACME-dataset", institution="go2nus")[0]
traj.odometry          # DataFrame: timestamp, position_x, position_y, yaw, linear_vel_x, linear_vel_y, angular_vel
traj.detections        # DataFrame: frame_idx, timestamp, track_id, x, y, width, height
traj.speech            # DataFrame: timestamp, text
traj.image(0)          # (H, W, 3) uint8 RGB
traj.mask(0)           # (H, W) bool person mask
traj.pointcloud(0)     # (N, 4) float32 x, y, z, intensity
for frame in traj:     # all modalities of one sample
    ...
```

### Raw ROS 2 bags (no ROS installation needed)

```python
from acme_dataset.rosbag import Bag, decode_detections, detect_bbox_encoding

with Bag("ACME-dataset/ros2 bags/miraikan/<bag>.zip") as bag:   # extracts the zip on first use
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
acme-dataset convert "ACME-dataset/ros2 bags/miraikan" --out converted/miraikan --rate 4
```

The topics used per institution are in [`acme_dataset/configs/topics.yaml`](acme_dataset/configs/topics.yaml).

### BEV pedestrian trajectories

```python
from acme_dataset import BEVSession, list_bev_sessions

print(list_bev_sessions("ACME-dataset/bev"))           # dataset, session
session = BEVSession("ACME-dataset/bev", "bonn", "20")
df = session.trajectories()                            # metres, filtered, 10 FPS
df = session.trajectories(target_fps=2.5)              # as transform_trajectories.py
segments = session.tag_segments(df)                    # split by semantic region
```

`trajectories()` returns one row per pedestrian and frame, with columns `ped_id, frame, time_s,
x, y` (metres), `px, py` (BEV pixels) and `label` (`dynamic`, `stationary` or
`out_of_bounds`). The filter drops tracks shorter than 3.2 s, interpolates annotation jumps,
smooths over 5 frames and labels points whose local speed is below 0.5 m/s as stationary.
Export every session to CSV with:

```bash
acme-dataset bev-export ACME-dataset/bev --out bev_csv --target-fps 2.5
acme-dataset bev-export ACME-dataset/bev --out bev_csv_tags --tags
```

### Scenario tags and statistics

```python
from acme_dataset import load_scenario_tags, filter_by_tags, tags_for, dataset_stats, summarize

tags = load_scenario_tags("ACME-dataset/dataset_tags.xlsx")
narrow = filter_by_tags(tags, all_of=["narrow_corridors"], any_of=["against_traffic", "large_crowds"])
stats = dataset_stats("ACME-dataset")      # per trajectory: duration, distance, speed, pedestrians, ...
summarize(stats)                           # per institution
```

```bash
acme-dataset info ACME-dataset                       # what is on disk
acme-dataset info "ACME-dataset/ros2 bags/miraikan/<bag>.zip"   # topics of a bag, read from metadata.yaml
acme-dataset stats ACME-dataset --out stats.csv
```

## Notebooks

| Notebook | Content |
|---|---|
| [`01_human_readable_data.ipynb`](notebooks/01_human_readable_data.ipynb) | odometry, images with tracks, person masks, point clouds, speech, scenario tags, statistics |
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
| `masked_imgs/<i>.jpg` | Person segmentation mask, 255 = person. JPEG compressed, so threshold at 128. |
| `pcd/<i>.pcd`, `pcd/<i>.npz` | LiDAR point cloud in the LiDAR frame, fields `x y z intensity`. The `.npz` holds the same `(N, 4)` float32 array under the key `arr_0`. |
| `speech.txt` | `<UNIX time>: <text>` per line: robot utterances, or robot UI events for the suitcase robots. |
| BEV `<session>.toml` | `[human<N>]` tables with `traj_start` (first frame) and `trajectories` (pixel `[x, y]` per frame at 10 FPS). |
| BEV `<session>_homography.txt` | 3x3 matrix `H`, `[x_m, y_m, w] = H [x_px, y_px, 1]`. |
| BEV `annotations/<stem>.json` | `[{tag_id, tag_name, points: [[x, y], ...]}]` polygons. Tag ids: 1 Open Area, 2 Wide Corridor, 3 Narrow Corridor, 4 Intersection, 5 Entry/Exit, 6 Blind Corner, 7 Ignore. |

## Repository layout

| Folder | Purpose |
|---|---|
| [`acme_dataset/`](acme_dataset) | Python package and `acme-dataset` CLI |
| [`notebooks/`](notebooks) | Example notebooks |
| [`tests/`](tests) | Unit tests on synthetic data (`pytest`); `ACME_ROOT=... pytest tests/test_sample_data.py` checks a real download |

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
