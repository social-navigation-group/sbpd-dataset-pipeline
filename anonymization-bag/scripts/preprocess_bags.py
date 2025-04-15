#!/usr/bin/env python3
import os
import sys
import shutil
import cv2
import yaml
from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.image import message_to_cvimage  # utility for CompressedImage conversion

def filter_bag(input_bag_path, output_bag_path, topics_to_save):
    """
    Read an existing bag file (using rosbags), filter messages for the specified topics,
    and write them into a new bag file. For topics that contain images (e.g. CompressedImage),
    also extract the image to compile a video and log a timestamp file.
    
    The output video file and timestamp text file will be saved in the same directory
    as the filtered bag.
    """
    # Prepare video and log handlers for image topics.
    image_handlers = {}  # key: topic name -> dict with video_writer and txt_file
    added_connections = {}  # track connections that have been added to the writer

    OCCUPANCY_GRID_UPDATE_MSG = """
    std_msgs/Header header
    int32 x
    int32 y
    uint32 width
    uint32 height
    int8[] data
    """

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(OCCUPANCY_GRID_UPDATE_MSG, 'map_msgs/msg/OccupancyGridUpdate'))

    # Open the bag files with rosbags.
    with AnyReader([Path(input_bag_path)]) as reader, Writer(output_bag_path) as writer:
        # Ensure output bag directory exists; rosbags' Writer writes to a directory
        if not os.path.exists(output_bag_path):
            os.makedirs(output_bag_path, exist_ok=True)

        # Loop through all messages in the bag.
        for connection, timestamp, rawdata in reader.messages():
            # Check if the current topic is one of the topics to be saved.
            if (connection.topic in topics_to_save) or (len(topics_to_save) == 0):
                # If this connection has not been registered with the writer, add it.
                if connection.topic not in added_connections:
                    # Convert the msgtype object to a string to make it YAML representable.
                    connection_w = writer.add_connection(connection.topic, connection.msgtype, typestore=typestore)
                    added_connections[connection.topic] = connection_w
                    print(f"Registered connection for topic '{connection.topic}' with type '{connection.msgtype}'")

                # Process if the message is a CompressedImage message.
                # (Adjust the check if your topic uses uncompressed sensor_msgs/Image).
                if 'Image' in connection.msgtype:
                    try:
                        # Deserialize the message to a CompressedImage.
                        image_msg = reader.deserialize(rawdata, connection.msgtype)
                        # Convert to OpenCV image (BGR format) using rosbags-image.
                        cv_image = message_to_cvimage(image_msg, 'bgr8')
                    except Exception as e:
                        print(f"Error converting image message from topic '{connection.topic}' at time {timestamp}: {e}")
                        continue

                    # Initialize video writer and log file for this topic if not done already.
                    if connection.topic not in image_handlers:
                        safe_topic_name = connection.topic.replace('/', '_')
                        video_filename = os.path.join(output_bag_path, safe_topic_name + ".avi")
                        txt_filename = os.path.join(output_bag_path, safe_topic_name + ".txt")
                        meta_filename = os.path.join(output_bag_path, safe_topic_name + ".yaml")

                        # Initialize the video writer once we know the frame size.
                        height, width = cv_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))
                        txt_file = open(txt_filename, 'w')
                        meta_file = open(meta_filename, 'w')
                        image_handlers[connection.topic] = {
                            'video_writer': video_writer,
                            'txt_file': txt_file,
                            'frame_count': 0,
                            'meta_file': meta_file,
                        }
                        meta_dict = {
                            'topic': connection.topic,
                            'message': connection.msgtype,
                            'frame_id': image_msg.header.frame_id,
                        }
                        yaml.dump(meta_dict, meta_file)
                        print(f"Initialized video writer for topic '{connection.topic}' at resolution {width}x{height}")
                    
                    # Write the current frame and record the timestamp.
                    handler = image_handlers[connection.topic]
                    handler['video_writer'].write(cv_image)
                    handler['txt_file'].write(f"{handler['frame_count']} {timestamp}\n")
                    handler['frame_count'] += 1
                    print(f"Saved frame {handler['frame_count']} from topic '{connection.topic}' at time {timestamp}", end='\r', flush=True)
                else:
                    # Write to the new bag file.
                    writer.write(added_connections[connection.topic], timestamp, rawdata)
    
    # Close video writers and log files.
    for topic, handler in image_handlers.items():
        handler['video_writer'].release()
        handler['txt_file'].close()
        print(f"Released resources for topic '{topic}'")

    print(f"Finished filtering bag: {input_bag_path}")
    

def process_directory(base_path, topics_to_save):
    """
    Recursively search for bag directories (by looking for a 'metadata.yaml' file).
    For each bag directory found, call filter_bag to process and save the filtered bag,
    along with any generated video and timestamp files.
    If a filtered bag directory already exists, it will be overwritten.
    """
    for root, dirs, files in os.walk(base_path):
        # Skip any directories that are already filtered.
        dirs[:] = [d for d in dirs if not d.endswith('_filtered')]
        
        if 'metadata.yaml' in files:
            bag_dir = root
            # Skip processing if this bag_dir itself ends with '_filtered'
            if bag_dir.endswith('_filtered') or bag_dir.endswith('_merged'):
                print(f"Skipping already filtered or merged directory: {bag_dir}")
                continue

            parent_dir, bag_name = os.path.split(bag_dir)
            output_bag_dir = os.path.join(parent_dir, bag_name + "_filtered")

            print(f"\nFound bag directory: {bag_dir}")
            print(f"Filtered bag will be created at: {output_bag_dir}")

            # Overwrite existing filtered bag directory if it exists.
            if os.path.exists(output_bag_dir):
                print(f"Overwriting existing filtered bag: {output_bag_dir}")
                shutil.rmtree(output_bag_dir)
            
            filter_bag(bag_dir, output_bag_dir, topics_to_save)
            
            # Since this folder is a bag, don't descend further into it.
            dirs[:] = []

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rosbags_filter.py <base_path> [<topic1> <topic2> ...]")
        sys.exit(1)
    
    base_path = sys.argv[1]
    if len(sys.argv) < 2:
        print("No topics specified, all topics will be saved.")
        topics_to_save = []
    else:
        topics_to_save = sys.argv[2:]
    print(f"Starting search from base path: {base_path}")
    print(f"Filtering topics: {topics_to_save}")
    process_directory(base_path, topics_to_save)
