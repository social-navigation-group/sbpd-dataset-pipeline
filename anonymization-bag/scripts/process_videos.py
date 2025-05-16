# Copyright (c) Facebook, Inc. and its affiliates.
import os
import copy
import cv2
import tqdm

import pickle as pkl
import numpy as np

from arguments import get_args
from detectron2.config import get_cfg
from predictor import VisualizationDemo as Detectron2Demo

from byte_track_wrapper import ByteTrackWrapper
from pathlib import Path

def setup_cfg(args, mask=False):
    # For detectron2 masking
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    # if mask:
    if mask:
        cfg.merge_from_file(args.segmentation_config)
        cfg.merge_from_list(["MODEL.WEIGHTS", args.segmentation_weights])
    else:
        cfg.merge_from_file(args.keypoint_config)
        cfg.merge_from_list(["MODEL.WEIGHTS", args.keypoint_weights])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def process_video_mask(cap, out, metadata, num_frames, args):
    """
    Process a video using Detectron2 for instance segmentation.
    Args:
        cap: OpenCV video capture object.
        out: OpenCV video writer object.
        metadata: Metadata dictionary to store results.
        num_frames: Number of frames in the video.
        args: Command-line arguments.
    Returns:
        metadata: Updated metadata dictionary with mask information.
    """
    cfg = setup_cfg(args, mask=True)
    demo = Detectron2Demo(cfg)
    mask_info = []
    for (_, vis_frame, mask_frame) in tqdm.tqdm(demo.run_on_video(cap), total=num_frames):
        out.write(vis_frame)
        mask_info.append(mask_frame)
    metadata["mask_info"] = np.array(mask_info)
    return metadata

def process_video_keypoints(cap, metadata, num_frames, args):
    """
    Process a video using Detectron2 for keypoint detection.
    Args:
        cap: OpenCV video capture object.
        metadata: Metadata dictionary to store results.
        num_frames: Number of frames in the video.
        args: Command-line arguments.
    Returns:
        metadata: Updated metadata dictionary with keypoint information.
    """
    cfg = setup_cfg(args, mask=False)
    demo = Detectron2Demo(cfg)
    keypoint_info = []
    for (predictions, _, _) in tqdm.tqdm(demo.run_on_video(cap), total=num_frames):
        assert predictions.has("pred_keypoints")
        keypoints = predictions.pred_keypoints
        keypoints = np.asarray(keypoints)
        keypoint_info.append(keypoints)
    metadata["keypoint_info"] = keypoint_info
    return metadata

def process_video_tracker(cap, out, metadata, num_frames, args):
    """
    Process a video using ByteTrack for object tracking.
    If args.use_blur is set, anonymize the video by blurring the tracked objects.
    Args:
        cap: OpenCV video capture object.
        out: OpenCV video writer object.
        metadata: Metadata dictionary to store results.
        num_frames: Number of frames in the video.
        args: Command-line arguments.
    Returns:
        metadata: Updated metadata dictionary with tracking information.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = ByteTrackWrapper(args.bytetrack_model, args.bytetrack_config)
    track_info = []
    frame_idx = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # TODO: Here you can add any frame processing you need. 
        save_frame = copy.deepcopy(frame)
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
                save_frame[y1:new_y2, x1:x2] = cv2.GaussianBlur(frame[y1:new_y2, x1:x2], (args.blur_size, args.blur_size), 0)

        track_info.append({'bbox': bbox_tlwh, 'track_ids': track_ids})
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
        if not args.use_mask:
            # Write the processed frame to the output video.
            out.write(save_frame)
    metadata["track_info"] = track_info
    return metadata

def process_video(input_video_path, output_video_path, args):
    """
    Process a single video using OpenCV.
    For demonstration, this function reads the video and writes it back unchanged.
    Insert your own processing (filters, detection, etc.) here.
    """
    cap = cv2.VideoCapture(input_video_path)
    cap2 = cv2.VideoCapture(input_video_path)
    cap3 = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {input_video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    metadata_path = output_video_path.replace("_processed.avi", "_metadata.pkl")
    metadata = {}
    
    if args.use_mask:
        metadata = process_video_mask(cap, out, metadata, num_frames, args)
        print("Mask processing done.")
    metadata = process_video_tracker(cap2, out, metadata, num_frames, args)
    print("Tracker processing done.")
    metadata = process_video_keypoints(cap3, metadata, num_frames, args)
    print("Keypoint processing done.")
        
    cap.release()
    cap2.release()
    cap3.release()
    out.release()
    with open(metadata_path, 'wb') as f:
        pkl.dump(metadata, f)
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
                # Process the videos in the directory.
                process_videos_in_directory(filtered_dir, args)
    return
    
def main():
    args = get_args()
    process_filtered_directories(args)
    return

if __name__ == "__main__":
    main()