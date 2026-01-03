#!/usr/bin/env bash
set -u  # safer; add -e or -o pipefail if you prefer strict fail-fast behavior

# Usage:
#   ./process_all_bags.sh [--dry-run] [--skip-conversion] [--robot-name <name>] <rosbag_dir>
#
# Example:
#   ./process_all_bags.sh --dry-run /data/ros2_bags
#   ./process_all_bags.sh --skip-conversion --robot-name spot /data/ros1_bags
#
# Expected environment variables:
#   ros1bag_dir, save_dir, odom_topic, lidar_topic, viz_lidar, robot_name

# --- Parse arguments ---
DRY_RUN=false
SKIP_CONVERSION=true
rosbag_dir=""  # will be set based on mode if not provided
save_dir="/output"
ros1bag_dir="/ros1_bags/ubonn"
robot_name="ubonn"
odom_topic="/go1_controller/odom" #<--- update to your odom topic
lidar_topic="/velodyne_points" #<--- update to your lidar topic
lidar_frame="velodyne"

viz_lidar=false
NEXT_ARG_IS_ROBOT_NAME=false
for arg in "$@"; do
  if $NEXT_ARG_IS_ROBOT_NAME; then
    robot_name="$arg"
    NEXT_ARG_IS_ROBOT_NAME=false
    continue
  fi
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    --skip-conversion)
      SKIP_CONVERSION=true
      ;;
    --robot-name)
      NEXT_ARG_IS_ROBOT_NAME=true
      ;;
    *)
      if [ -z "$rosbag_dir" ]; then
        rosbag_dir="$arg"
      else
        echo "Unexpected extra argument: $arg" >&2
        exit 1
      fi
      ;;
  esac
done

# Set default rosbag_dir based on mode if not provided
if [ -z "$rosbag_dir" ]; then
  if $SKIP_CONVERSION; then
    rosbag_dir="$ros1bag_dir"
  else
    rosbag_dir="/ros2_bags"
  fi
fi

if [ -z "$rosbag_dir" ]; then
  echo "Usage: $0 [--dry-run] [--skip-conversion] <rosbag_dir>"
  exit 1
fi

if [ ! -d "$rosbag_dir" ]; then
  echo "Error: provided rosbag_dir '$rosbag_dir' does not exist or is not a directory."
  exit 1
fi

if $DRY_RUN; then
  echo "🟡 Dry run mode enabled — no conversions or roslaunch commands will be executed."
fi

if $SKIP_CONVERSION; then
  echo "🔵 Skip conversion mode enabled — processing existing ROS1 bags only."
fi


# --- Find bags based on conversion mode ---
# Find all ROS1 bag files
mapfile -d '' ros1_bags < <(
  find "$rosbag_dir" -type f -name '*.bag' -print0
)
echo "Found ${#ros1_bags[@]} ROS1 bag files under $rosbag_dir"

# --- Process each ROS1 bag ---
for ros1_bag in "${ros1_bags[@]}"; do
  # Relative path w.r.t. the root input dir
  bag_relpath="${ros1_bag#"$rosbag_dir"/}"
  reldir="$(dirname "$bag_relpath")"
  if [[ "$reldir" == "." ]]; then reldir=""; fi

  filename="$(basename "$ros1_bag" .bag)"

  # Output directory mirrors structure
  out_dir="${save_dir}/${reldir}"
  savefile="${out_dir}/${filename}.pkl"

  echo
  echo "🔹 Checking: $ros1_bag"
  echo "   Relpath: $bag_relpath"
  echo "   Target output dir: $out_dir"

  if $DRY_RUN; then
    [[ -f "$savefile" ]] && echo "   ⏩ Would skip — output exists: $savefile" || echo "   📝 Would produce: $savefile"
    echo "   (dry-run) Would run: roslaunch spot_move_base parse_rosbag.launch robot_name:=\"$robot_name\" rosbag_path:=\"$ros1_bag\" save_data_path:=\"$out_dir\" ..."
    continue
  fi

  # Ensure output directory exists
  mkdir -p "$out_dir"

  # if [[ -f "$savefile" ]]; then
  #   echo "   ⏩ Skipping — output file already exists: $savefile"
  #   continue
  # fi

  # Verify required topics exist
  # rosbag_info="$(rosbag info "$ros1_bag" 2>/dev/null || true)"
  # if ! echo "$rosbag_info" | grep -Fq "$odom_topic" || ! echo "$rosbag_info" | grep -Fq "$lidar_topic"; then
  #   echo "⚠️ Skipping $ros1_bag — missing required topics ($odom_topic / $lidar_topic)"
  #   continue
  # fi

  echo "🚀 Launching processing for $ros1_bag"
  roslaunch spot_move_base parse_rosbag.launch \
    robot_name:="$robot_name" \
    rosbag_path:="$ros1_bag" \
    save_data_path:="$out_dir" \
    odom_topic:="$odom_topic" \
    lidar_topic:="$lidar_topic" \
    viz_lidar:="$viz_lidar" \
    lidar_frame:="$lidar_frame"
done