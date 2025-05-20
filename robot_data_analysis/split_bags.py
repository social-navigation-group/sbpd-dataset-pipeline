import rosbag2_py
import rclpy.serialization
from tqdm import tqdm
from std_msgs.msg import String
from builtin_interfaces.msg import Time as Stamp, Duration as Duration
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from realsense2_camera_msgs.msg import Metadata, Extrinsics
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from enum import Enum
import cv2
from cv_bridge import CvBridge
import yaml
import os
import sys
import argparse
STATIC_TF_FILE_PATH = '/home/shashank/code/my_projects/sbpd-dataset-pipeline/robot_data_analysis/go2_static_transforms.yaml' #TODO replace with local system file path


class SyncSignal(Enum):
    DISCARD = "-1"
    BEGIN = "-2"
    END = "-3"

bridge = CvBridge()
rgb_image_topic = "/camera/camera/color/image_raw/compressed"
depth_image_topic=  "/camera/camera/aligned_depth_to_color/image_raw/compressed"
odom_topic = "/utlidar/robot_odom"
camera_rgb_depth_extrinsics_topic = "/camera/camera/extrinsics/depth_to_color"
sync_command_topic = "/sync_command"

def load_static_transforms(yaml_path, start_time):
    """Loads static transforms from a YAML file and ensures proper type conversion for translation and rotation."""
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        static_transforms = []
        stamp = Stamp()
        stamp.sec = int(start_time/1e9)
        stamp.nanosec = int(1e9 * (start_time/1e9 - int(start_time/1e9)))
        for entry in data.get("static_transforms", []):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = entry["parent"]
            transform.child_frame_id = entry["child"]

            # Ensure translation and rotation are floats
            transform.transform.translation.x = float(entry["translation"][0])
            transform.transform.translation.y = float(entry["translation"][1])
            transform.transform.translation.z = float(entry["translation"][2])

            transform.transform.rotation.x = float(entry["rotation"][0])
            transform.transform.rotation.y = float(entry["rotation"][1])
            transform.transform.rotation.z = float(entry["rotation"][2])
            transform.transform.rotation.w = float(entry["rotation"][3])

            static_transforms.append(transform)
        
        return static_transforms
    except Exception as e:
        print(f"Error loading static transforms: {e}")
        return []

def extract_valid_intervals(bag_path, storage_id, compression):
    """
    Extract valid start-stop timestamp pairs from the /sync_command topic while handling discard signals.
    """
    if not compression:
        reader = rosbag2_py.SequentialReader()
    else:
        reader = rosbag2_py.SequentialCompressionReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    reader.set_filter(rosbag2_py.StorageFilter(topics=["/sync_command"]))

    intervals = []
    start_time = None
    discard_start = False

    messages = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, String).data.strip()

        if msg == SyncSignal.BEGIN.value:  # Start timestamp
            if not discard_start and start_time is None:  # Only consider if no discard flag, and the first of all begin signals before and end/discard signal
                start_time = timestamp
            discard_start = False  # Reset discard flag

        elif msg == SyncSignal.END.value and start_time is not None:  # Stop timestamp
            intervals.append((start_time, timestamp))
            start_time = None  # Reset start_time after pairing

        elif msg == SyncSignal.DISCARD.value:  # Discard signal
            start_time = None  # Ignore the previous start signal
            discard_start = True  # Set discard flag

        messages.append(msg)

    print(messages)

    del reader  # Close reader properly
    return intervals

def flip_compressed_image_msg(compressed_msg):
    """Flips a CompressedImage message (sensor_msgs/CompressedImage)."""
    image_np = bridge.compressed_imgmsg_to_cv2(compressed_msg, desired_encoding='passthrough')
    # cv_image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)  # Decode compressed image
    # _, compressed_flipped = cv2.imencode('.jpg', flipped_image)  # Re-encode the image

    flipped_image = cv2.flip(image_np, -1)  # Flip both vertically & horizontally
    flipped_msg = bridge.cv2_to_compressed_imgmsg(flipped_image, 'jpg')
    # flipped_msg = CompressedImage()
    flipped_msg.header = compressed_msg.header
    flipped_msg.format = compressed_msg.format
    # flipped_msg.data = compressed_flipped.tobytes()  # Convert back to bytes
    return flipped_msg

def get_stamp_difference(t1: Stamp, t2: Stamp) -> Duration:
    """Return the difference between two ROS 2 Time stamps as a builtin_interfaces/Duration."""
    t1_ns = t1.sec * 1_000_000_000 + t1.nanosec
    t2_ns = t2.sec * 1_000_000_000 + t2.nanosec
    delta_ns = t1_ns - t2_ns

    # Convert back to Duration
    sec = delta_ns // 1_000_000_000
    nanosec = delta_ns % 1_000_000_000

    # Handle negative nanoseconds when delta_ns is negative
    if delta_ns < 0 and nanosec != 0:
        sec -= 1
        nanosec = 1_000_000_000 + nanosec  # nanosec is already negative

    return Duration(sec=sec, nanosec=nanosec)

def add_duration_to_stamp(stamp: Stamp, duration: Duration) -> Stamp:
    total_sec = stamp.sec + duration.sec
    total_nanosec = stamp.nanosec + duration.nanosec

    # Normalize nanoseconds to stay within [0, 1_000_000_000)
    if total_nanosec >= 1_000_000_000:
        total_sec += 1
        total_nanosec -= 1_000_000_000

    return Stamp(sec=total_sec, nanosec=total_nanosec)

def odom_to_tf(odom_msg):
    """Converts Odometry message to a TFMessage for /tf topic."""
    transform = TransformStamped()
    transform.header = odom_msg.header
    # transform.header.stamp = 
    transform.child_frame_id = "base_link"
    transform.transform.translation.x = odom_msg.pose.pose.position.x
    transform.transform.translation.y = odom_msg.pose.pose.position.y
    transform.transform.translation.z = odom_msg.pose.pose.position.z
    transform.transform.rotation = odom_msg.pose.pose.orientation

    tf_msg = TFMessage()
    tf_msg.transforms.append(transform)
    return tf_msg

def split_rosbag(bag_path, storage_id, intervals, compression, output_bag_path):
    """
    Reads messages from the original bag and writes valid ones to new bag files based on intervals.
    """
    if not compression:
        reader = rosbag2_py.SequentialReader()
    else:
        reader = rosbag2_py.SequentialCompressionReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    # Data from these topics will not be written to the bag as is
    topics_to_exclude = ["/tf", "/tf_static", sync_command_topic, camera_rgb_depth_extrinsics_topic]

    metadata = reader.get_metadata()
    start_time = metadata.starting_time.nanoseconds
    topic_list = [t.topic_metadata.name for t in metadata.topics_with_message_count if t.topic_metadata.name not in topics_to_exclude]
    print(topic_list)
    #exit()
    # Topics with headers, timestamps from the headers will be written as message timestamps
    topic_to_type_map = {
        "/utlidar/robot_odom": Odometry,
        "/rslidar_points": PointCloud2,
        "/camera/camera/aligned_depth_to_color/camera_info": CameraInfo,
        "/camera/camera/color/camera_info" : CameraInfo,
        "/camera/camera/color/metadata" : Metadata,
        # "/camera/camera/extrinsics/depth_to_color" : Extrinsics,
        "/camera/camera/color/image_raw/compressed" : CompressedImage,
        "/camera/camera/aligned_depth_to_color/image_raw/compressed": CompressedImage
    }

    # Load static sensor transforms
    print(f"Loading static transforms from: {STATIC_TF_FILE_PATH}")
    static_transforms = load_static_transforms(STATIC_TF_FILE_PATH, start_time)

    # Initialize writer
    writer = None
    bag_count = 0
    first_odom_timestamp = None
    writing = False
    got_rgb_depth_extrinsics_data = False
    rgb_depth_extrinsics_data = None
    rgb_depth_extrinsics_msg = None

    print(f"Writing to output bag path: {output_bag_path}")

    with tqdm(total=len(intervals), desc="Processing intervals", unit="bag") as pbar:
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            # Check if within any valid interval
            in_valid_range = any(start <= timestamp <= stop for start, stop in intervals)

            if in_valid_range and not writing:
                # Start writing a new bag
                first_odom_timestamp = None
                bag_start_stamp = Stamp(sec=int(timestamp/1e9), nanosec=int(1e9 * (timestamp/1e9 - int(timestamp/1e9))))
                bag_count += 1
                bag_name = bag_path.split('/')[-1]
                uri = os.path.join(output_bag_path, f"{bag_name}_{bag_count}")
                    
                writer = rosbag2_py.SequentialWriter()
                writer.open(
                    rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
                    rosbag2_py.ConverterOptions()
                )

                # Create topics in new bag
                for topic_name in topic_list:
                    # if "tf" in topic_name:
                    #     continue
                    for topic_info in reader.get_metadata().topics_with_message_count:
                        if topic_info.topic_metadata.name == topic_name:
                            writer.create_topic(topic_info.topic_metadata)
                            #print(topic_info.topic_metadata.name)

                # Create /tf topic for transformed odometry data
                writer.create_topic(rosbag2_py._storage.TopicMetadata(
                    name="/tf",
                    type="tf2_msgs/msg/TFMessage",
                    serialization_format="cdr"
                ))

                writer.create_topic(rosbag2_py._storage.TopicMetadata(
                    name="/tf_static",
                    type="tf2_msgs/msg/TFMessage",
                    serialization_format="cdr"
                ))

                # writer.create_topic(rosbag2_py._storage.TopicMetadata(
                #     name=camera_rgb_depth_extrinsics_topic,
                #     type="realsense2_camera_msgs/msg/Extrinsics",
                #     serialization_format="cdr"
                # ))

                # Write static transforms to /tf_static once per bag
                static_tf_msg = TFMessage()
                static_tf_msg.transforms = static_transforms
                static_tf_data = rclpy.serialization.serialize_message(static_tf_msg)
                writer.write("/tf_static", static_tf_data, timestamp)

                writing = True
                
                # Write to start of bag (non-first bag) with the start timestamp for the split bag
                # if got_rgb_depth_extrinsics_data:
                #     rgb_depth_extrinsics_msg = rclpy.serialization.deserialize_message(got_rgb_depth_extrinsics_data, Extrinsics)
                #     if hasattr(rgb_depth_extrinsics_data, 'header'):
                #         rgb_depth_extrinsics_msg.header.stamp = Stamp(sec=int(timestamp/1e9), nanosec=int(1e9 * (timestamp/1e9 - int(timestamp/1e9))))
                #         rgb_depth_extrinsics_data = rclpy.serialization.serialize_message(rgb_depth_extrinsics_msg)
                #     writer.write(camera_rgb_depth_extrinsics_topic, rgb_depth_extrinsics_data, timestamp)

            if writing and in_valid_range and topic != sync_command_topic and topic!= camera_rgb_depth_extrinsics_topic:
                
                # # Skip writing tf or tf static data origiating from rosbag. tf_static is written separately for each bag, and tf is calculated from odom below. 
                # if "tf" in topic_name:
                #     continue

                # Reading message from bag
                # if not got_rgb_depth_extrinsics_data:
                #     if topic == camera_rgb_depth_extrinsics_topic:
                #         # Read and store message for further bags
                #         rgb_depth_extrinsics_data = data
                #         rgb_depth_extrinsics_msg = rclpy.serialization.deserialize_message(data, Extrinsics)
                #         if hasattr(rgb_depth_extrinsics_msg, 'header'):
                #             rgb_depth_extrinsics_msg.header.stamp = Stamp(sec=int(timestamp/1e9), nanosec=int(1e9 * (timestamp/1e9 - int(timestamp/1e9))))
                #             rgb_depth_extrinsics_data = rclpy.serialization.serialize_message(rgb_depth_extrinsics_msg)
                        
                #         # Write to bag directly as read, for first bag
                #         writer.write(camera_rgb_depth_extrinsics_topic, rgb_depth_extrinsics_data, timestamp)
                #         got_rgb_depth_extrinsics_data = True
                
                
                if topic in [rgb_image_topic, depth_image_topic]:
                    # Flip compressed depth image
                    image_msg = rclpy.serialization.deserialize_message(data, CompressedImage)
                    flipped_image_msg = flip_compressed_image_msg(image_msg)
                    data = rclpy.serialization.serialize_message(flipped_image_msg)

                elif topic == odom_topic:
                    data_ = rclpy.serialization.deserialize_message(data, Odometry)
                    
                    if first_odom_timestamp is None:
                        first_odom_timestamp = data_.header.stamp

                    odom_timestamp_offset = get_stamp_difference(data_.header.stamp, first_odom_timestamp)
                    # print(f"Adding difference {odom_timestamp_offset.sec}.{odom_timestamp_offset.nanosec}")
                    data_.header.stamp = add_duration_to_stamp(stamp=bag_start_stamp, duration=odom_timestamp_offset)
                    # Write extra to /tf topic
                    tf_msg = odom_to_tf(data_)
                    tf_data = rclpy.serialization.serialize_message(tf_msg)
                    writer.write("/tf", tf_data, timestamp)
                    # Modify odom data stamp since odom data stamps on GO2 are old
                    data = rclpy.serialization.serialize_message(data_)

                if topic in topic_to_type_map.keys():
                    data_ = rclpy.serialization.deserialize_message(data, topic_to_type_map.get(topic))
                    timestamp = int(data_.header.stamp.sec * 1e9 + data_.header.stamp.nanosec)
                    data = rclpy.serialization.serialize_message(data_)
                
                writer.write(topic, data, timestamp)
                    

            elif writing and not in_valid_range:
                # Stop writing when out of range
                del writer
                writer = None
                writing = False
                pbar.update(1)  # Move progress

    del reader  # Close reader properly

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split rosbag files based on sync command intervals.")
    parser.add_argument('--base_path',"-i", type=str, required=True, help='Base path to the directory containing rosbag files.')
    parser.add_argument('--output_bag_path',"-o", type=str, required=True, help='Output path for the split rosbag files.')
    parser.add_argument('--bag_files','-b', nargs='*', help='List of bag files to process. If not provided, all bag files in the base path will be processed.')
    args = parser.parse_args()
    base_path = args.base_path
    output_bag_path = args.output_bag_path
    bag_files = args.bag_files
    if not bag_files:
        # If no specific bag files are provided, process all bag files in the base path
        bag_files = [f for f in os.listdir(base_path)]

    storage_id = 'sqlite3'      # sqlite3 / mcap
    compression = False

    for bag_file_name in bag_files:
        try:
            print(f"\nProcessing bag file: {bag_file_name}\n")
            bag_file_path = os.path.join(base_path, bag_file_name)
            output_bag_path = output_bag_path
            valid_intervals = extract_valid_intervals(bag_path=bag_file_path, storage_id=storage_id, compression=compression)
            print(f"Extracted trajectory intervals: {valid_intervals}")
            split_rosbag(bag_path=bag_file_path, storage_id=storage_id, intervals=valid_intervals, compression=compression, output_bag_path=output_bag_path)
        except Exception as e:
            print(f"Exception: {e.with_traceback()}")