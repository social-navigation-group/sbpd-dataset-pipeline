# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
import copy
import shutil

import cv2
import tqdm
import yaml

from detectron2.config import get_cfg
from predictor import VisualizationDemo as Detectron2Demo

from byte_track_wrapper import ByteTrackWrapper

from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
# from builtin_interfaces.msg import Time
from rosbags.typesys.stores.ros2_humble import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__CompressedImage as CompressedImage,
    std_msgs__msg__Header as Header,
)

from arguments import get_args

def setup_cfg(args):
    # For detectron2 masking
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.segmentation_config)
    cfg.merge_from_list(["MODEL.WEIGHTS", args.segmentation_weights])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def process_video(input_video_path, output_video_path, args):
    """
    Process a single video using OpenCV.
    For demonstration, this function reads the video and writes it back unchanged.
    Insert your own processing (filters, detection, etc.) here.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {input_video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    
    if args.use_mask:
        cfg = setup_cfg(args)
        demo = Detectron2Demo(cfg)
        for vis_frame in tqdm.tqdm(demo.run_on_video(cap), total=num_frames):
            frame_idx += 1
            #print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
            out.write(vis_frame)
    else:
        if args.use_blur:
            tracker = ByteTrackWrapper(args.bytetrack_model, args.bytetrack_config)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # TODO: Here you can add any frame processing you need. 
            save_frame = copy.deepcopy(frame)
            if args.use_blur:
                bbox_tlwh, track_ids = tracker.update(frame)
                for box_index in range(len(bbox_tlwh)):
                    top_left_x, top_left_y, bb_width, bb_height = bbox_tlwh[box_index]
                    x1, y1, x2, y2 = int(top_left_x), int(top_left_y), int(top_left_x + bb_width), int(top_left_y + bb_height)

                    # Anonymize bounding box
                    new_y2 = y1 + max((y2 - y1) * args.blur_pct, args.blur_min)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width - 1, x2)
                    new_y2 = min(height - 1, new_y2)
                    x1, y1, x2, new_y2 = map(int, [x1, y1, x2, new_y2])
                    if not ((x1 < x2) and (y1 < new_y2)):
                        continue
                    #print(f"Blurring bounding box: {x1}, {y1}, {x2}, {new_y2}")
                    save_frame[y1:new_y2, x1:x2] = cv2.GaussianBlur(frame[y1:new_y2, x1:x2], (args.blur_size, args.blur_size), 0)

            frame_idx += 1
            print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
            out.write(save_frame)
    cap.release()
    out.release()
    return

def process_videos_in_directory(directory, args):
    """
    For each video in the directory (with .avi extension) that is not already processed,
    call process_video() and save a processed copy with a _processed.avi suffix.
    """
    video_files = [f for f in os.listdir(directory) 
                   if f.endswith(".avi") and not f.endswith("_processed.avi")]
    for video in video_files:
        video_path = os.path.join(directory, video)
        processed_video_path = os.path.join(directory, Path(video).stem + "_processed.avi")
        print(f"Processing video: {video_path}")
        process_video(video_path, processed_video_path, args)
        print(f"Processed video saved to: {processed_video_path}")
    return

def merge_processed_videos(filtered_dir, merged_bag_path):
    """
    In the given _filtered directory, find all processed videos (with _processed.avi suffix),
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
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(OCCUPANCY_GRID_UPDATE_MSG, 'map_msgs/msg/OccupancyGridUpdate'))

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

    # Find all processed video files in the directory.
    processed_videos = glob.glob(os.path.join(filtered_dir, "*_processed.avi"))
    for video_path in processed_videos:
        # Assume the basename is something like <safe_topic_name>_processed.avi;
        # remove the _processed part.
        base_name = Path(video_path).stem  # e.g. "topic1_processed"
        safe_topic_name = base_name.replace("_processed", "")
        txt_file = os.path.join(filtered_dir, safe_topic_name + ".txt")
        meta_file = os.path.join(filtered_dir, safe_topic_name + ".yaml")
        
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
                print(f"Warning: Frame index exceeds timestamp list length. Using 0 as default timestamp. [{frame_idx}]")
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
    writer.close()
    print(f"Merged processed videos into bag at: {merged_bag_path}")
    return

def process_filtered_directories(args):
    """
    Recursively search for directories ending with '_filtered'.
    In each such directory, process videos and merge the processed results into a bag.
    """
    base_path = args.base_path
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d.endswith("_filtered"):
                filtered_dir = os.path.join(root, d)
                print(f"\nProcessing _filtered directory: {filtered_dir}")
                # Step 1: Process the videos in the directory.
                process_videos_in_directory(filtered_dir, args)
                # Step 2: Create a merged bag file from the processed videos.
                merged_dir_name = d.replace("_filtered", "_merged")
                merged_bag_path = os.path.join(root, merged_dir_name)
                merge_processed_videos(filtered_dir, merged_bag_path)
    return
    

def main():
    args = get_args()
    process_filtered_directories(args)
    return

if __name__ == "__main__":
    main()