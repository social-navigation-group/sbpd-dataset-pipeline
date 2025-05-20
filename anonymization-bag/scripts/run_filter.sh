#!/bin/bash
# filepath: /home/allanwan/dataset/bag_file_process/scripts/run_filter.sh

# run_filter.sh
# This script executes the Python filtering script.
#
# Usage:
#   ./run_filter.sh [-p <base_path>] [-a] [-b] [-m]
#

# Set default base path and default topics.
DEFAULT_BASE_PATH="/bag_ws/data/raw_bags"
DEFAULT_OUT_PATH="/bag_ws/data/processed_bags"

# DEFAULT_TOPICS=( 
#     "/d455_rs1/color/camera_info" 
#     "/d455_rs1/color/image_raw/compressed" 
#     "/d455_rs1/color/image_raw/compressedDepth"
#     "/d455_rs1/depth/camera_info" 
#     "/tf_static" 
#     "/tf"
#     "/cabot/odometry/filtered" 
#     "/scan" 
#     "/odom" 
#     "/velodyne_points" 
#     "/cabot/event" 
#     "/cabot/imu/data" 
#     "/cabot/odometry/filtered" 
#     "/robot_description" 
#     "/local_costmap/costmap"
#     "/local_costmap/footprint"
#     "/local_costmap/published_footprint"
#     "/local_costmap/costmap_updates"
#     "/global_costmap/costmap_updates"
#     "/global_costmap/footprint"
#     "/global_costmap/published_footprint"
#     "/global_costmap/costmap"
#     "/current_floor"
#     "/map"
#     "/cabot/pressure"
#     "/zed2/zed_node/odom"
#     "/zed2/zed_node/pose"
#     "/action"
#     "/zed2/zed_node/rgb/camera_info"
#     "/zed2/zed_node/rgb/image_rect_color/compressed"
#     "/speech_command"
#     "/camera/camera/extrinsics/depth_to_color"
#     "/camera/camera/color/image_raw/compressed"
#     "/rslidar_points"
#     "/camera/camera/color/camera_info"
#     "/camera/camera/aligned_depth_to_color/image_raw/compressed"
#     "/camera/camera/aligned_depth_to_color/camera_info"
#     "/utlidar/robot_odom"
#     "/speech_command"  
#     "/camera/camera/color/image_raw/compressed" 
#     "/camera/camera/color/camera_info"
#     "/camera/camera/aligned_depth_to_color/image_raw/compressed"
#     #"/camera/camera/color/metadata"
#     "/camera/camera/aligned_depth_to_color/camera_info" 
#     )

# Initialize variables.
DEFAULT_TOPICS=(
    # "/zed2/zed_node/rgb/camera_info"
    # "/zed2/zed_node/pose"
    # "/zed2/zed_node/pose_with_covariance"
    #"/zed2/zed_node/rgb/image_rect_color/compressed"
    "/image_raw/compressed"
    # "/zed2/zed_node/depth/camera_info"
    # "/zed2/zed_node/depth/depth_registered/compressed"
)
BASE_PATH="$DEFAULT_BASE_PATH"
OUT_PATH="$DEFAULT_OUT_PATH"
all_topics=false
use_blur=false
use_mask=false

# Parse options.
while getopts "p:o:abm" opt; do
    case "$opt" in
        a)
            all_topics=true
            ;;
        p)
            BASE_PATH="$OPTARG"
            ;;
        o)
            OUT_PATH="$OPTARG"
            ;;
        b)
            use_blur=true
            ;;
        m)
            use_mask=true
            ;;
        ?)
            echo "Usage: $0 [-p <base_path>] [-o <output_path>] [-a] [-b] [-m]"
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Always use default topics.
if $all_topics; then
    # If -a is specified, use all topics y specifying an empty list.
    TOPICS=()
else
    # Otherwise, use the default topics.
    TOPICS=( "${DEFAULT_TOPICS[@]}" )
fi

echo "Using base path: $BASE_PATH"
echo "Using output path: $OUT_PATH"
echo "Use blur: $use_blur"
echo "Use mask: $use_mask"
# echo "Filtering for topics:"
# for t in "${TOPICS[@]}"; do
#     echo "  $t"
# done

# Build and display the command string, then evaluate it.
CMD1="python3 preprocess_bags.py \"$BASE_PATH\" \"$OUT_PATH\""
for t in "${TOPICS[@]}"; do
    CMD1+=" \"$t\""
done

#echo "$CMD1"
eval "$CMD1"

if $use_mask; then
    CMD2="python3 merge_videos.py --base-path \"$OUT_PATH\" --use-mask"
elif $use_blur; then
    CMD2="python3 merge_videos.py --base-path \"$OUT_PATH\" --use-blur"
else
    CMD2="python3 merge_videos.py --base-path \"$OUT_PATH\""
fi

# Append additional arguments (e.g. confidence threshold) if needed.
MERGE_ADDITIONAL_ARGS=""
CMD2="$CMD2 $MERGE_ADDITIONAL_ARGS"

#echo "$CMD2"
eval "$CMD2"
