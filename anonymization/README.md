# Anonymization and Tracking

## Overview
The codes in this folder are for anonymizing videos (Both FPV and BEV videos) and producing automatically tracked trajectories for pedestrians.

To use these codes:
1. Place all the raw videos in the "videos" folder.
2. If you would like to define an area within the videos for producing trajectories individually, run `define_area.py`
3. Run `anonymize_and_track.py`.

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

`anonymize_and_track.py` runs anonymization and tracking of pedestrians at the same time, but either anonymization or tracking can be turned off. This code has many options to customize how you want to achieve anonymization or tracking.

By default, the code will: 
1. Read all the videos saved in the "videos" folder.
2. Put the anonymized videos in the "videos_anonymized" folder
3. Put the tracked trajectories and the anonymized videos for the trajectories in the "trajectories" folder. These videos have the same fps as the trajectories and display the tracking area if an area is defined for tracking the trajectories. The data in this folder can be directly copied to the labeling tool for processing.

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
