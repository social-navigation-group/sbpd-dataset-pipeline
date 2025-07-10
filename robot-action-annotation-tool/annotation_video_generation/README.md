# Annotation video generation

Scripts to generate videos to annotate robot action labels

### Steps to run:

1. Replace the value for `BASE_PATH` [here](./annotate.sh) with the path containing all bag files. (No nested directory structure for split bags). Videos for all bag files placed under `BASE_PATH` will be generated.
2. To experiment with view parameters (described below), test with one/two bags under `BASE_PATH`, and place all bag files once view parameters are confirmed. Or uncomment the break statement at the end of [`annotate.sh`](./annotate.sh).
3. Fill parameter values in [params.yaml](./params.yaml)
4. **From this (annotation_video_generation) folder**, run [`annotate.sh`](./annotate.sh)

> **Dependencies**:
- ffmpeg
- pyvista, rosbag2_py, tqdm, yaml (python)

All output video files (with the same name as input rosbag) will be written to the current folder.

___

Working of the scripts is explained below. The script generates some temporary files, which are deleted after each bag file is processed.
### Writes frames from rosbag
> `python3 generate_cloud_video.py {BASE_PATH} {BAG_FILE_NAME}`

1. Read the bag file
2. Write all pointclouds to individual pkl files (`./temp_files/{BAG_FILE_NAME}/clouds/`)
3. Writes a list of poses from odometry data corresponding (by closest message timestamp) to each pointcloud (`./temp_files/{BAG_FILE_NAME}/odom.pkl`)
5. Construct frames with odometry trajectory and robot velocity arrow, save to `./temp_files/{BAG_FILE_NAME}/images/`

### Writes video from frames
> `ffmpeg -framerate 15 -i ./temp_files/${BAG_FILE_NAME}/images/image_%d.png -s 1920x1080 -c:v libx264 -pix_fmt yuv420p -crf 18 ./temp_files/${BAG_FILE_NAME}/cloud.mp4`

### Generate FPV video from rosbag

> `python3 video_from_rosbag.py {BASE_PATH} {BAG_FILE_NAME}`
Writes video from rosbag to .`/temp_files/{bag_file_name}/fpv_video.mp4`

### Combine videos

Writes final video file to `{BAG_FILE_NAME}.mp4` to the annotation directory

`ffmpeg -i ./temp_files/${BAG_FILE_NAME}/cloud.mp4 -i ./temp_files/${BAG_FILE_NAME}/fpv_video.mp4 -filter_complex "[1:v] scale=640:-1 [pip]; [0:v][pip] overlay=0:H-h" -r 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -preset veryfast ${BAG_FILE_NAME}.mp4`

### View Parameters

`generate_cloud_video.py`:

- in function `display_pointcloud_pyvista`

```python
    # View parameters
    elevation = 25
    azimuth = 180
    view_distance = 12.5
```

These values determine the viewpoint of the pointcloud in the video frames.
