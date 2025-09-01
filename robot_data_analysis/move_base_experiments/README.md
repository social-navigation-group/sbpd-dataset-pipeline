# Move Base Experiments - Docker Setup

This Docker setup allows you to run the move_base experiments on your local rosbag files without needing to install ROS and all dependencies locally.

## Prerequisites

- Docker and Docker Compose installed on your system
- Your rosbag files stored locally

## Quick Start

1. **Set your rosbag directory paths and other variables**:
   - in docker-compose.yml, please set the directory containing your ros2 bags in the following line:
   ```bash
      # Change the left side to your actual rosbag directory path
      - ./ros2_bags/:/rosbags:ro
   ```
   - update the footprint and velocity-related parameters in 
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
### Script Arguments (run.sh)
The run.sh script accepts the following arguments:
```bash
./run.sh [rosbag_source] [ros1_bag_dest] [output_dir] [odom_topic] [lidar_topic] [viz_lidar]
```

- `rosbag_source`: Directory containing ROS2 rosbag files (default: `/rosbags`)
- `ros1_bag_dest`: Directory for converted ROS1 bags (default: `/ros1_bags`) 
- `output_dir`: Directory to save experiment results (default: `/output`)
- `odom_topic`: Odometry topic name (default: `/zed2/zed_node/odom`)
- `lidar_topic`: LiDAR topic name (default: `/velodyne_points`)
- `viz_lidar`: Enable visualization (default: `false`)

## Customization

### Modifying Parameters
Navigation parameters can be found in `src/spot_move_base/params/`. Key files:
- `move_base_params.yaml` - Move base configuration
- `costmap_common_params.yaml` - Costmap settings
- `base_local_planner_params.yaml` - Local planner settings

### Adding Custom Rosbags
1. Set the ROSBAG_DIR environment variable to point to your ROS2 rosbag directory
2. Ensure your ROS2 rosbags contain the required topics (odometry and lidar)
3. Run the script with your specific topic names:
   ```bash
   ./run.sh /rosbags /ros1_bags /output /your/odometry/topic /your/lidar/topic
   ```
4. The script will automatically convert ROS2 bags to ROS1 format before processing

### Output Data
Results will be saved in the `output/` directory. The data includes:
- Navigation paths
- Costmap data
- Performance metrics

## Troubleshooting

### Container won't start
- Ensure Docker and Docker Compose are installed
- Check that port 11311 (ROS master) is not already in use

### Missing topics in rosbag
The script will skip rosbags that don't contain the required topics. Check your ROS2 rosbag info:
```bash
ros2 bag info your_rosbag_directory
```

### Permission issues
If you encounter permission issues with mounted volumes:
```bash
sudo chown -R $USER:$USER output/
```

## Development

To make changes to the code:
1. Modify files in the `src/` directory
2. Rebuild the container: `docker-compose build`
3. Run with your changes: `docker-compose up`

## Support

If you encounter issues:
1. Check that your ROS2 rosbag files contain the expected topics
2. Verify the topic names match your rosbag structure  
3. Ensure sufficient disk space for output files and ROS1 conversions
4. The rosbags-convert tool requires Python 3.6+ and handles most ROS2 to ROS1 conversions automatically