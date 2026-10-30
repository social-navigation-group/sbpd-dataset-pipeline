#!/bin/bash

# Set the path to the directory containing the rosbag files
rosbag_dir=${1:-"/rosbags"}
ros1bag_dir=${2:-"/ros1_bags"}
# Set the path to the directory to save the output files
save_dir=${3:-"/output"}
# Topic names specified as command line arguments
odom_topic=${4:-"/utlidar/robot_odom"}
lidar_topic=${5:-"/rslidar_points"}
viz_lidar=${6:-"false"}


# Loop through all rosbag files in the directory
for rosbag_file in "$rosbag_dir"/*; do
  echo "Processing $rosbag_file..."
  filename=$(basename "$rosbag_file" .db3)
  ros1_bag="$ros1bag_dir/${filename}.bag"
  
  # Convert ROS2 bag to ROS1 if it doesn't already exist
  if [ ! -f "$ros1_bag" ]; then
    echo "Converting ROS2 bag $rosbag_file to ROS1 format..."
    rosbags-convert --dst "$ros1_bag" "$rosbag_file" --include-topic "$odom_topic" --include-topic "$lidar_topic"
    if [ $? -ne 0 ]; then
      echo "Failed to convert $rosbag_file, skipping..."
      continue
    fi
  else
    echo "ROS1 bag $ros1_bag already exists, using existing file"
  fi
  
  # Check if required topics exist in the converted rosbag
  rosbag_info=$(rosbag info "$ros1_bag" 2>/dev/null)
  if ! echo "$rosbag_info" | grep -q "$odom_topic" || ! echo "$rosbag_info" | grep -q "$lidar_topic"; then
    echo "Skipping $ros1_bag - missing required topics"
    continue
  fi

  # Run the roslaunch file with the converted rosbag file and save path
  roslaunch spot_move_base parse_rosbag.launch rosbag_path:="$ros1_bag" save_data_path:="$save_dir" odom_topic:="$odom_topic" lidar_topic:="$lidar_topic" viz_lidar:="$viz_lidar"
done