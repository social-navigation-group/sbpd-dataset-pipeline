# Anonymization and Tracking

## Overview
The codes in this folder are for anonymizing videos (Both FPV and BEV videos) and producing automatically tracked trajectories for pedestrians.

To use these codes:
1. Place all the raw videos in the "videos" folder.
2. If you would like to define an area within the videos for producing trajectories individually, run `define_area.py`
3. Run `run_all_videos.py`.

## Dependencies
Create a virtual environment or activate your own virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Follow the official instructions to install ByteTrack: [ByteTrack GitHub Repository](https://github.com/ifzhang/ByteTrack)
**Make sure these paths are valid in your setup:**
```
- Model checkpoint:
../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar

- Config:
../ByteTrack/exps/example/mot/yolox_x_mix_det.py
```

Install the following packages
```
pip install numpy toml pyyaml opencv-python 
```
Create the following folders
```
mkdir videos areas trajectories videos_anonymized
```

## Detailed Explanations
The pipeline includes two main scripts:
 - `anonymize_and_track.py` processes a single video to perform pedestrian anonymization and tracking at the same time (but either anonymization or tracking can be turned off). It provides various options to customize how these tasks are executed.
 - `run_all_videos.py` is a wrapper script that automatically applies `anonymize_and_track.py` to all videos in a specified folder (default: `./videos`). Each video is processed in a separate subprocess to ensure that tracking IDs are reset between videos and do not carry over, which could lead to ID conflicts or tracking inconsistencies across videos.
   
> To process all videos in a folder using full-frame anonymization and restricted area tracking:
```
python3 run_all_videos.py --video-folder ./videos2 --blur-all --restrict-area
```

By default, the code will: 
1. It reads video files (e.g., `.mp4`, `.avi`, `.mov`) from a folder.
2. Outputs anonymized versions to the folder specified by `--output` (default: `./videos_anonymized`).
3. Outputs trajectories and annotated trajectory videos to the folder specified by `--trajectory-output` (default: `./trajectories`). These videos have the same fps as the trajectories and display the tracking area if an area is defined for tracking the trajectories. The data in this folder can be directly copied to the labeling tool for processing.
4. If area-based tracking is enabled (`--restrict-area`), area definitions are read from or saved to the folder specified by `--area-path` (default: `./areas`).

Notable Options:
- `no-blur`: Disable anonymization
- `blur-all`: Blur the entire video instead of bounding boxes picked up by tracking
- `no-track`: Disable tracking
- `restrict-area`: Enable tracking area for tracking to be registered only inside the defined areas.

If `restrict-area` is turned on, it will attempt to read the area yaml file saved in the "areas" folder. For each video file, it will look for the yaml file with the same name. If it doesn't exist, it will call the `select_area` function in `define_area.py` and ask you to select the points to define the area.

You can also use `define_area.py` to define the tracking area for each video file individually.

## Tips
- In both python files, scale can be helpful if the video resolution is marger than the screen resolution.
- For `restrict-area`, area selection should be focused on pedestrians' footing area.

## Connect to labelling-tool

Go to the "trajectories" folder. Copy the toml files to `<path to labelling-tool>/resources/config/original_data` and the video files to `<path to labelling-tool>/resources/videos`. Then please refer to labeling-tool's readme for further instructions.
