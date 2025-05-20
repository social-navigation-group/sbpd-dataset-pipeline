# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import os
import copy
import shutil

import cv2
import tqdm
import yaml
import pickle as pkl
# from detectron2.config import get_cfg
#from predictor import VisualizationDemo as Detectron2Demo

from byte_track_wrapper import ByteTrackWrapper

from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg, get_types_from_idl
from ament_index_python.packages import get_package_prefix
from builtin_interfaces.msg import Time
from rosbags.typesys.stores.ros2_humble import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__CompressedImage as CompressedImage,
    std_msgs__msg__Header as Header,
)
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    parser.add_argument(
        "--base-path",
        help="Base path to the bag files and video files",
    )

    parser.add_argument(
        "--use-tracking",
        action="store_true",
        help="Run Tracking only",
    )

    # Detectron2 arguments
    parser.add_argument(
        "--use-mask", 
        action = "store_true", 
        help = "Instance segmentation on the video (Higher priority than blur)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--detectron-config",
        default="/bag_ws/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        nargs=argparse.REMAINDER,
    )

    # ByteTrack arguments
    parser.add_argument(
        "--use-blur", 
        action = "store_true", 
        help = "Anonymize the video (Not performed if also instance segmentation)"
    )
    parser.add_argument(
        "--bytetrack-model", 
        #default = "/home/shashank/code/packages/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", 
        default= "../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar",
        type = str, 
        help = "Path to the YOLOX model"
    )
    parser.add_argument(
        "--bytetrack_config", 
        #default = "/home/shashank/code/packages/ByteTrack/exps/example/mot/yolox_x_mix_det.py", 
        default= "../ByteTrack/exps/example/mot/yolox_x_mix_det.py",
        type = str, 
        help = "ByteTrack experiment config file"
    )
    parser.add_argument(
        "--blur-size", 
        default = 41, 
        type = int, 
        help = "Size of the blur kernel",
    )
    parser.add_argument(
        "--blur-pct", 
        default = 0.25, 
        type = float, 
        help = "Percentage of the bounding box to blur",
    )
    parser.add_argument(
        "--blur-min", 
        default = 25, 
        type = int, 
        help = "Minimum pixels of the bounding box to blur",
    )

    args = parser.parse_args()
    if args.base_path is None:
        raise ValueError("Base path is required.")
    if args.use_blur and args.use_mask:
        print("Warning: Both --use-blur and --use-mask are set. --use-mask will take priority.")
        args.use_blur = False
    return args

def setup_cfg(args):
    # For detectron2 masking
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.detectron_config)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

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
    if args.use_tracking:
        tracker = ByteTrackWrapper(args.bytetrack_model, args.bytetrack_config)
        out_tracks_file = output_video_path.replace("_processed.avi", "_tracks.pkl")
        track_info = {}
        
    if args.use_mask: #segmentation
        return
        cfg = setup_cfg(args)
        demo = Detectron2Demo(cfg)
        for vis_frame,id_frame in tqdm.tqdm(demo.run_on_video(cap), total=num_frames):
            frame_idx += 1
            #print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
            out.write(vis_frame)
    else: #blurring
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # TODO: Here you can add any frame processing you need. 
            save_frame = copy.deepcopy(frame)
            if args.use_blur or args.use_tracking:
                bbox_tlwh, track_ids = tracker.update(frame)
                if args.use_blur:                    
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
                if args.use_tracking:
                    #save the tracking info
                    track_info[frame_idx] = {'bbox': bbox_tlwh, 'track_ids': track_ids}
                    #save_frame = plot_tracking(save_frame, bbox_tlwh, track_ids, frame_id=frame_idx)
            frame_idx += 1
            print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
            out.write(save_frame)
    cap.release()
    out.release()
    if args.use_tracking:
        with open(out_tracks_file, 'wb') as f:
            pkl.dump(track_info, f)
        print(f"Saved tracking info to {out_tracks_file}")
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
    
    # Find the tracking file if it exists.
    tracking_files = glob.glob(os.path.join(filtered_dir, "*_tracks.pkl"))
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
    
    
    #merge the tracking file if it exists
    if tracking_files:
        for tracking_file in tracking_files:
            # Assume the basename is something like <safe_topic_name>_processed.avi;
            # remove the _processed part.
            base_name = Path(tracking_file).stem  # e.g. "topic1_processed"
            safe_topic_name = base_name.replace("_tracks", "")
            txt_file = os.path.join(filtered_dir, safe_topic_name + ".txt")
            meta_file = os.path.join(filtered_dir, safe_topic_name + ".yaml")
            
            # Load meta file containing topic and message type information.
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
            topic = "/tracks"+meta['topic'].replace("/","_")
            msgtype = meta['message']
            frame_id = meta['frame_id']
            print(f"Merging processed video for topic: {topic}, message type: sensor_msgs/msg/CompressedImage")

            # Read the timestamp file into a list.
            with open(txt_file, 'r') as f:
                timestamps = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # The second element is the ROS timestamp (as string or integer).
                        timestamps.append(parts[1])

            # Register a connection for this topic if not already done.
            if topic not in added_connections:
                connection_w = writer.add_connection(topic, Detection2DArray.__msgtype__, typestore=typestore)
                added_connections[topic] = connection_w
            with open(tracking_file, 'rb') as f: 
                detections = pkl.load(f)
            
            for frame_idx,detection in detections.items():
                # Convert the timestamp string to int (or float) as needed.
                if frame_idx < len(timestamps):
                    timestamp = int(timestamps[frame_idx])
                else:
                    print(f"Warning: Frame index exceeds timestamp list length. Using 0 as default timestamp. [{frame_idx}]")
                    timestamp = 0
                    
                # Create a Detection2DArray message.
                header = Header(
                    stamp=Time(sec=int(timestamp // 10**9), nanosec=int(timestamp % 10**9)),
                    frame_id= frame_id,
                )
                detection_array = Detection2DArray(
                    header = header,
                    detections = []
                )
                
                for bbox, track_id in zip(detection['bbox'], detection['track_ids']):
                    bounding_box = BoundingBox2D(
                        center = Pose2D(
                            position = Point2D(
                                x = int((bbox[0]+bbox[2])/2),
                                y = int((bbox[1]+bbox[3])/2)
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
                writer.write(added_connections[topic], timestamp, data_bytes)
    else:
        print("No tracking file found.")
    
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
                shutil.rmtree(filtered_dir)
    return
    

def main(args):
    args = get_parser()
    process_filtered_directories(args)
    return

if __name__ == "__main__":
    args = get_parser()
    main(args)