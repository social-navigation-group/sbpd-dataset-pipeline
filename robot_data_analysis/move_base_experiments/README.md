# Move Base Experiments - Docker Setup

This Docker setup allows you to run the move_base experiments on your local rosbag files without needing to install ROS and all dependencies locally.

## Prerequisites

- Docker and Docker Compose installed on your system
- Your ros2bag files stored locally

## Quick Start

1. **Set your rosbag directory paths and other variables**:

   - in docker-compose.yml, please set the directory containing your ros2 bags in the following line:

   ```bash
      # Change the left side to the root directory containing ros2 bags directory path
      - ./ros2_bags/:/rosbags:ro
      # Change the left side to an empty directory with enough space to store the ros1 versions (with only pointcloud and odom topics) that will be generated for this experiment. 
      - /ros1_bags:/ros1_bags

   ```

   - update the footprint and velocity-related parameters in to suite your robot
     `move_base_experiments/src/spot_move_base/params/costmap_common_params.yaml` and
     `move_base_experiments/src/spot_move_base/params/base_local_planner_params.yaml`
   - in `run.sh`, please specify the lidar and odom topic names:

   ```bash
   odom_topic=${4:-"/utlidar/robot_odom"}
   lidar_topic=${5:-"/rslidar_points"}
   ```
2. **Build and run the container**:

   ```bash
   docker-compose up --build -d
   ```
3. **Build the workspace and run the experiment**:

   ```bash
   docker compose exec move_base_experiments bash
   catkin_make
   source devel/setup.bash
   ./run.sh
   ```

## Directory Structure

```
move_base_experiments/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Container orchestration
├── run.sh                 # Experiment runner script
├── output/                # Experiment results will be saved here
├── src/                   # ROS packages source code
│   ├── spot_move_base/    # Main navigation package
│   ├── lidar_rosbag_parser/ # Rosbag processing utilities
│   └── pointcloud_to_laserscan/ # Point cloud conversion
└── README.md              # This file

# Note: ROS2 rosbag files are mounted from your local system
# Converted ROS1 bags are stored in the container's /ros1_bags directory
```

### Output Data

Results will be saved in the `output/` directory. The data includes:

- Navigation paths
- Costmap data
- Performance metrics

## Troubleshooting

### Missing topics in rosbag

The script will skip rosbags that don't contain the lidar and odom topics. Check your ROS2 rosbag info:

```bash
ros2 bag info your_rosbag_directory
```

### Permission issues

If you encounter permission issues with mounted volumes:

```bash
sudo chown -R $USER:$USER output/
```

## Support

If you encounter issues:

1. Check that your ROS2 rosbag files contain the expected topics
2. Verify the topic names match your rosbag structure
3. Ensure sufficient disk space for output files and ROS1 conversions
4. The rosbags-convert tool requires Python 3.6+ and handles most ROS2 to ROS1 conversions automatically
