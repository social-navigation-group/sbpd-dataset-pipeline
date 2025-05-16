# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
import shutil
import cv2
import yaml
import sys

import pickle as pkl

from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.typesys import Stores, get_typestore, get_types_from_msg, get_types_from_idl
from ament_index_python.packages import get_package_prefix
# from builtin_interfaces.msg import Time
from rosbags.typesys.stores.ros2_humble import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__CompressedImage as CompressedImage,
    std_msgs__msg__Header as Header,
)

def merge_processed_videos(filtered_dir, merged_bag_path):
    """
    In the given _filtered directory, find all processed videos (with _processed.avi suffix),
    In the given _filtered directory, find all processed videos (with _processed.avi suffix), and tracking file,
    then for each video read its associated timestamp text file (.txt) and meta file (.yaml).
    Using this information, open the processed video with OpenCV, and write its frames as ROS
    image messages into a new bag file (using rosbags) with the appropriate timestamp and connection info.
    """
    # If a merged bag already exists at this path, remove it.
    if os.path.exists(merged_bag_path):
        shutil.rmtree(merged_bag_path)
    
    writer = Writer(merged_bag_path)
    writer.open()
    os.makedirs(merged_bag_path, exist_ok=True)
    added_connections = {}  # topic: connection handle

    OCCUPANCY_GRID_UPDATE_MSG = """
    std_msgs/Header header
    int32 x
    int32 y
    uint32 width
    uint32 height
    int8[] data
    """

    BINARY_MASK_MSG = """
    std_msgs/Header header
    int32 width
    int32 height
    bool[] data
    """

    KEYPOINT_MSG = """
    float32 x
    float32 y
    float32 visibility
    """

    KEYPOINTS_MSG = """
    custom_msgs/Keypoint[] keypoints
    """

    KEYPOINTARRAY_MSG = """
    std_msgs/Header header
    custom_msgs/Keypoints[] people
    """

    # Register the message type for occupancy grid update.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(OCCUPANCY_GRID_UPDATE_MSG, 'map_msgs/msg/OccupancyGridUpdate'))
    typestore.register(get_types_from_msg(BINARY_MASK_MSG, 'custom_msgs/msg/BinaryMask'))
    typestore.register(get_types_from_msg(KEYPOINT_MSG, 'custom_msgs/msg/Keypoint'))
    typestore.register(get_types_from_msg(KEYPOINTS_MSG, 'custom_msgs/msg/Keypoints'))
    typestore.register(get_types_from_msg(KEYPOINTARRAY_MSG, 'custom_msgs/msg/KeypointArray'))

    # First, write all the data from the original filtered bag file.
    filtered_bag_metadata = os.path.join(filtered_dir, "metadata.yaml")
    if os.path.exists(filtered_bag_metadata):
        print(f"Merging filtered bag from directory: {filtered_dir}")
        # Use AnyReader to open the filtered bag directory.
        with AnyReader([Path(filtered_dir)]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                # Register the connection if not already registered.
                if connection.topic not in added_connections:
                    connection_w = writer.add_connection(connection.topic, connection.msgtype, typestore=typestore)
                    added_connections[connection.topic] = connection_w
                writer.write(added_connections[connection.topic], timestamp, rawdata)
        print("Finished merging messages from the filtered bag file.")
    else:
        raise Exception("No filtered bag file found in this directory.")

    # Second, find all processed video files in the directory.
    # Write the processed video frames as ROS image messages into the bag.
    processed_videos = glob.glob(os.path.join(filtered_dir, "*_processed.avi"))
    for video_path in processed_videos:
        # Assume the basename is something like <save_topic_name>_processed.avi;
        # remove the _processed part.
        base_name = Path(video_path).stem  # e.g. "topic1_processed"
        save_topic_name = base_name.replace("_processed", "")
        txt_file = os.path.join(filtered_dir, save_topic_name + ".txt")
        meta_file = os.path.join(filtered_dir, save_topic_name + ".yaml")
        
        # Load meta file containing topic and message type information.
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        topic = meta['topic']
        msgtype = meta['message']
        frame_id = meta['frame_id']
        print(f"Merging processed video for topic: {topic}, message type: sensor_msgs/msg/CompressedImage")
        # Register a connection for this topic if not already done.
        if topic not in added_connections:
            connection_w = writer.add_connection(topic, CompressedImage.__msgtype__, typestore=typestore)
            added_connections[topic] = connection_w
        # Read the timestamp file into a list.
        with open(txt_file, 'r') as f:
            timestamps = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # The second element is the ROS timestamp (as string or integer).
                    timestamps.append(parts[1])
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Get the corresponding timestamp from the timestamps list.
            if frame_idx < len(timestamps):
                # Convert the timestamp string to int (or float) as needed.
                timestamp = int(timestamps[frame_idx])
            else:
                print(f"Warning: Frame index exceeds timestamp list length. Using 0 as default timestamp (Video). [{frame_idx}]")
                timestamp = 0
            # # Convert the OpenCV frame (BGR image) to a sensor_msgs/msg/CompressedImage.
            msg = CompressedImage(
                Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id=frame_id,
                ),
                format='jpeg',  # could also be 'png'
                data=cv2.imencode('.jpg', frame)[1],
            )
            data_bytes = typestore.serialize_cdr(msg, msg.__msgtype__)
            writer.write(added_connections[topic], timestamp, data_bytes)
            frame_idx += 1
        cap.release()

    # Third, add detected information (mask, tracking, keypoints) to the bag.
    # Find the tracking file if it exists.
    tracking_files = glob.glob(os.path.join(filtered_dir, "*_metadata.pkl"))
    if tracking_files:
        #register vision_msgs types
        add_types = {}
        for idl in glob.glob(os.path.join(get_package_prefix('rclpy'),"share","vision_msgs/msg/*.idl")):
            add_types.update(get_types_from_idl(Path(idl).read_text()))
        typestore.register(add_types)
        Detection2DArray = typestore.types['vision_msgs/msg/Detection2DArray']
        Detection2D = typestore.types['vision_msgs/msg/Detection2D']
        BoundingBox2D = typestore.types['vision_msgs/msg/BoundingBox2D']
        Point2D = typestore.types['vision_msgs/msg/Point2D']
        Pose2D = typestore.types['vision_msgs/msg/Pose2D']
        BinaryMask = typestore.types['custom_msgs/msg/BinaryMask']
        Keypoint = typestore.types['custom_msgs/msg/Keypoint']
        Keypoints = typestore.types['custom_msgs/msg/Keypoints']
        KeypointArray = typestore.types['custom_msgs/msg/KeypointArray']

        for tracking_file in tracking_files:
            # Assume the basename is something like <save_topic_name>_metadata.pkl;
            # remove the _metadata part.
            base_name = Path(tracking_file).stem  # e.g. "topic1_processed"
            save_topic_name = base_name.replace("_metadata", "")
            txt_file = os.path.join(filtered_dir, save_topic_name + ".txt")
            meta_file = os.path.join(filtered_dir, save_topic_name + ".yaml")

            # Load meta file containing topic and message type information.
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
            track_topic = "/tracks" + meta['topic'].replace("/","_")
            mask_topic = "/masks" + meta['topic'].replace("/","_")
            keypoint_topic = "/keypoints" + meta['topic'].replace("/","_")
            frame_id = meta['frame_id']
            print(f"Merging tracking information for topic: {track_topic}, message type: vision_msgs/msg/Detection2DArray")

            # Read the timestamp file into a list.
            with open(txt_file, 'r') as f:
                timestamps = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # The second element is the ROS timestamp (as string or integer).
                        timestamps.append(parts[1])

            # Register a connection for this topic if not already done.
            if track_topic not in added_connections:
                connection_w = writer.add_connection(track_topic, Detection2DArray.__msgtype__, typestore=typestore)
                added_connections[track_topic] = connection_w
            if mask_topic not in added_connections:
                connection_w = writer.add_connection(mask_topic, BinaryMask.__msgtype__, typestore=typestore)
                added_connections[mask_topic] = connection_w
            if keypoint_topic not in added_connections:
                connection_w = writer.add_connection(keypoint_topic, KeypointArray.__msgtype__, typestore=typestore)
                added_connections[keypoint_topic] = connection_w

            # Read the tracking file (pickle) to get the metadata.
            with open(tracking_file, 'rb') as f: 
                metadata = pkl.load(f)
                
            detections = metadata["track_info"]
            num_frames = len(detections)
            has_mask = ("mask_info" in metadata.keys())
            for frame_idx in range(num_frames):
                # Convert the timestamp string to int (or float) as needed.
                if frame_idx < len(timestamps):
                    timestamp = int(timestamps[frame_idx])
                else:
                    print(f"Warning: Frame index exceeds timestamp list length. Using 0 as default timestamp (Tracking). [{frame_idx}]")
                    timestamp = 0

                # Create a Detection2DArray message.
                header = Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id= frame_id,
                )

                # Fill in the Detection2DArray message with bounding boxes and track IDs.
                detection = detections[frame_idx]
                detection_array = Detection2DArray(
                    header = header,
                    detections = []
                )
                for bbox, track_id in zip(detection['bbox'], detection['track_ids']):
                    bounding_box = BoundingBox2D(
                        center = Pose2D(
                            position = Point2D(
                                x = int(bbox[0]+bbox[2]/2),
                                y = int(bbox[1]+bbox[3]/2)
                            ),
                            theta = 0.0
                        ),
                        size_x = int(bbox[2]),
                        size_y = int(bbox[3]),
                    )
                    detection_array.detections.append(Detection2D(
                        header = header,
                        results = [],
                        bbox = bounding_box,
                        id = str(track_id),
                    ))
                # Write the Detection2DArray message to the bag.
                #embed()
                data_bytes = typestore.serialize_cdr(detection_array, detection_array.__msgtype__)
                writer.write(added_connections[track_topic], timestamp, data_bytes)

                # Fill in the BinaryMask message if mask data is available.
                if has_mask:
                    mask = metadata["mask_info"][frame_idx]
                    mask1d = mask.flatten()
                    binary_mask = BinaryMask(
                        header = header,
                        width = mask.shape[1],
                        height = mask.shape[0],
                        data = mask1d
                    )
                    # Write the BinaryMask message to the bag.
                    data_bytes = typestore.serialize_cdr(binary_mask, binary_mask.__msgtype__)
                    writer.write(added_connections[mask_topic], timestamp, data_bytes)

                # Fill in the KeypointArray message.
                people = metadata["keypoint_info"][frame_idx]
                keypoint_array = KeypointArray(
                    header = header,
                    people = []
                )
                # Each person is a list of keypoints.
                for person in people:
                    keypoints_person = []
                    for kp in person:
                        keypoints_person.append(Keypoint(
                            x = kp[0],
                            y = kp[1],
                            visibility = kp[2]
                        ))
                    keypoint_array.people.append(Keypoints(
                        keypoints = keypoints_person
                    ))
                # Write the KeypointArray message to the bag.
                data_bytes = typestore.serialize_cdr(keypoint_array, keypoint_array.__msgtype__)
                writer.write(added_connections[keypoint_topic], timestamp, data_bytes)
    else:
        print("No label data file found.")
    writer.close()
    print(f"Merged processed videos into bag at: {merged_bag_path}")
    return

def process_filtered_directories(base_path):
    """
    Recursively search for directories ending with '_filtered'.
    In each such directory, process videos and merge the processed results into a bag.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.endswith("_filtered"):
                filtered_dir = os.path.join(root, d)
                print(f"\nProcessing _filtered directory: {filtered_dir}")
                # Create a merged bag file from the processed videos.
                merged_dir_name = d.replace("_filtered", "_merged")
                merged_bag_path = os.path.join(root, merged_dir_name)
                merge_processed_videos(filtered_dir, merged_bag_path)
    return
    
def main(base_path):
    process_filtered_directories(base_path)
    return

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_videos.py <base_path>")
        sys.exit(1)
    base_path = sys.argv[1]
    main(base_path)