# Anonymization and Tracking

## Overview
The codes in this folder are for anonymizing videos (Both FPV and BEV videos in video formats such as mp4, avi) and optionally producing automatically tracked trajectories for pedestrians. The files output from the trajectory tracking can be directly moved to the labeling tool for manual annotation checks.

To use these codes by default:
1. Place all the raw videos in the "videos" folder.
2. If you would like to define an area within the videos for producing trajectories individually, run `define_area.py`
3. Run `run_videos.py`.

## Dependencies
Create a virtual environment or activate your own virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Follow the instructions to install the slightly modified ByteTrack: [ByteTrack GitHub Repository](https://github.com/allanwangliqian/ByteTrack/tree/main)

Also download the `bytetrack_x_mot17` checkpoint.

Install the following packages
```
pip install numpy toml pyyaml opencv-python 
```
Create the following folders
```
mkdir videos areas trajectories videos_anonymized
```

In `anonymize_and_track.py`, modify the default arguments of `--model` and `--experiment-config` to be the path to the checkpoint file and the config file respectively. 

For example:
```
parser.add_argument("--model", default = "../../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", type = str, help = "Path to the YOLOX model")
parser.add_argument("--experiment-config", default = "../../ByteTrack/exps/example/mot/yolox_x_mix_det.py", type = str, help = "ByteTrack experiment config file")
```

## Detailed Explanations
The pipeline includes two main scripts:
 - `anonymize_and_track.py` processes a single video to perform pedestrian anonymization and tracking at the same time (but either anonymization or tracking can be turned off). It provides various options to customize how these tasks are executed.
 - `run_all_videos.py` is a wrapper script that automatically applies `anonymize_and_track.py` to all videos in a specified folder (default: `./videos`). Each video is processed in a separate subprocess for faster speed and to ensure that tracking IDs are reset between videos and do not carry over, which could lead to ID conflicts or tracking inconsistencies across videos.
 - `define_area.py` is a convenience tool used to draw an area on the video so that trajectory tracking will only be performed within the defined area.

By default, the code will: 
1. Read all video files (e.g., `.mp4`, `.avi`, `.mov`) from a folder.
2. Output anonymized versions to a folder (default: `./videos_anonymized`).
3. Output trajectories and annotated trajectory videos to another folder (default: `./trajectories`). These videos have the same fps as the trajectories and display the tracking area if an area is defined for tracking the trajectories. The data in this folder can be directly copied to the labeling tool for processing.
4. If area-based tracking is enabled (`--restrict-area`), area definitions are read from or saved to a folder (default: `./areas`).

Options:
- `no-blur`: Disable anonymization
- `blur-all`: Blur the entire video instead of bounding boxes picked up by tracking
- `no-track`: Disable tracking
- `restrict-area`: Enable tracking area for tracking to be registered only inside the defined areas.

If `restrict-area` is turned on, it will attempt to read the area yaml file saved in the "areas" folder. For each video file, it will look for the yaml file with the same name. If it doesn't exist, it will call the `select_area` function in `define_area.py` and ask you to select the points to define the area. A white polygon representing the restricted area will also be drawn on the videos generated for the labeling tool.

You can also use `define_area.py` to define the tracking area for each video file individually first (recommended). For example:
```
python3 define_area.py --video <path to video>
```

Example ways to run the script
> To process using bbox-based anonymization and perform tracking over the entire video:
```
python3 run_videos.py --video-folder ./videos 
```
> To anonymize by painting the bbox black instead of blurring and perform tracking over the entire video:
```
python3 run_videos.py --video-folder ./videos --blur-black
```
> If no anonymization is needed and only perform tracking in restricted area:
```
python3 run_videos.py --video-folder ./videos --no-blur --restrict-area
```
> To process using full-frame blurring and perform tracking in restricted area:
```
python3 run_videos.py --video-folder ./videos --blur-all --restrict-area
```
> To process using full-frame blurring and no tracking:
```
python3 run_videos.py --video-folder ./videos --blur-all --no-track
```

## Tips
- Due to performance issues, video resolution will be capped at 1080p max.
- For more detailed pipeline customization, please modify the default values in `anonymize_and_track.py` directly.
- If the source video frame rates are not multiples of 10, `--traj-fps` needs to be modified. By default, the trajectory outputs are 10Hz.
- In `define_area.py`, scale can be helpful if the video resolution is larger than the screen display.
- For `restrict-area`, area selection should be based on the pedestrians' footing area.

## Connect to labeling-tool

labeling tool [link](https://github.com/social-navigation-group/sbpd-dataset-pipeline/tree/main/labelling-tool).

Go to the "trajectories" folder. Copy the toml files to `<path to labeling-tool>/resources/config/original_data` and the video files to `<path to labeling-tool>/resources/videos`. Then, the labeling-tool is ready to be used. Please refer to labeling-tool's readme for further instructions.
