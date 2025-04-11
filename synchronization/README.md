# Synchronization scripts

## Time and position synchronization

This file describes the functions of time and position synchronization for the social navigation dataset. The script `synchronize.py` performs time and position synchronization and writes the result to a file.

### Inputs

- 1 ROS 2 bag file
- 1 BEV video file
- Parameters from `input_params.yaml` as specified below

```yaml
input_params:
  rosbag_uri : str
  rosbag_storage_id : str
  rosbag_compression : boolean
  robot_camera_topic : str
  robot_camera_compression : boolean
  camera_info_topic : str
  sync_command_topic : str
  video_file_uri : str
  sync_command_message_type : str
  bev_skip_time_seconds : int
  bev_camera_intrinsics :
    fx : float
    fy : float
    cx : float
    cy : float
```

### Outputs

Output is written to a yaml file in the same directory as the video file.

1. **Time synchronization using dynamic QR code**:
    - ROS 2 timestamp corresponding to the start of the BEV video file
2. **Position synchronization using AprilTag**:
    - Relative poses and timestamps of the AprilTag as observed by the robot and BEV cameras.
Format

```yaml
rosbag_uri: str
video_uri: str
video_start_ros_timestamp: float
tag_pose_wrt_robot:
  pose:
    translation:
      x: float
      y: float
      z: float
    rotation:
      x: float
      y: float
      z: float
      w: float
  timestamp: float
tag_pose_wrt_bev:
  pose:
    translation:
      x: float
      y: float
      z: float
    rotation:
      x: float
      y: float
      z: float
      w: float
  timestamp: float
```

### Parameters

#### Required Parameters

- `rosbag_uri`: Path to the ROS 2 bag containing recorded data
- `rosbag_storage_id`: ROS 2 bag storage format (sqlite3 OR mcap)
- `rosbag_compression`: Boolean indicating if rosbags are compressed
- `robot_camera_topic`: ROS 2 topic containing robot FPV camera data
- `robot_camera_compression`: Boolean indicating if robot image data is compressed
- `robot_camera_info_topic`: ROS 2 topic containing camera info for the robot FPV camera
- `sync_command_topic`: ROS 2 topic containing sync command data
- `video_file_uri`: Path to the video file containing corresponding BEV data
- `bev_camera_intrinsics`: Camera matrix parameters $(f_x, f_y, c_x, c_y)$ for BEV cameras.

Please modify the parameters in the script

#### Optional Parameters

- `bev_skip_time_seconds`: Initial time in seconds to skip while searching for the time sync QR code in BEV video files. This saves processing time.

#### Requirements

1. For time synchroniation:
    -The dynamic QR code is shown to all cameras within the shortest possible interval. It is assumed that network latency is constant across this interval, and averaging across detections is used to minimize error.
2. For position synchrnonization
    - The "sync" signal data should be present on the `sync_command_topic` when both the robot and BEV cameras can see the AprilTag. The algorithm will search for AprilTag detections in frames nearest to the timestamp of the sync signal. This reduces processing time, and ensures correct frame association for the robot and BEV camera feeds.

#### Python Dependencies

- rclpy, rosbag2_py, cv_bridge
- numpy, scipy, opencv
- apriltag
- tqdm
- pyyaml

#### Assumpions/Redundancies

1. One rosbag may contain multiple "sync" signals, if there are multiple BEV video files for one rosbag. In this case the script should be run separately for each (BEV video, rosbag) pair.

#### TODOs

1. Replace print statements with logging for efficient debugging.

> Declaration of AI tools usage: Docstrings have been generated using AI tools.

___

## Time synchronization QR Code

The script `ros_time_to_qr_code.py` can be run as a python script after sourcing a ROS 2 installation. Python dependencies:

- rclpy, cv_bridge
- numpy, cv2
- qr_code
