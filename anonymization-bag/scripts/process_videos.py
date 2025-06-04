# Copyright (c) Facebook, Inc. and its affiliates.
import os
import copy
import cv2
import random

import pickle as pkl
import numpy as np

from arguments import get_args
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

from byte_track_wrapper import ByteTrackWrapper
from pathlib import Path


def process_video_mask(video_path, out, metadata, num_frames, args):
    """
    Process a video using Detectron2 for instance segmentation.
    Args:
        video_path: Path to the input video file.
        out: OpenCV video writer object.
        metadata: Metadata dictionary to store results.
        num_frames: Number of frames in the video.
        args: Command-line arguments.
    Returns:
        metadata: Updated metadata dictionary with mask information.
    """

    model = YOLO(args.segmentation_model)  # Load the segmentation model
    colour_bank = {}

    def get_colour(track_id):
        random.seed(track_id)              
        if track_id not in colour_bank:
            colour_bank[track_id] = tuple(random.randint(0,255) for _ in range(3))
        return colour_bank[track_id]

    frame_idx = 0
    mask_info = []
    for result in model.track(source=video_path,
                            classes=[0],            # people only
                            persist=True,           # keep IDs
                            stream=True,
                            show=False,
                            conf=args.confidence_threshold,
                            tracker=args.segmentation_tracker,
                            verbose=False):
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{num_frames}", end='\r', flush=True)
        frame = result.orig_img.copy()

        mask_bin = np.zeros(frame.shape[:2], dtype=np.bool_)  # binary mask for the current frame
        if result.masks is not None:                
            masks = result.masks.data.cpu().numpy()  
            masks_hwN = masks.transpose(1, 2, 0)      
            masks_hwN = scale_image(masks_hwN, frame.shape[:2]) 
            masks = masks_hwN.transpose(2, 0, 1) 

            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else range(len(masks))

            for mask, tid in zip(masks, ids):
                colour = get_colour(int(tid))
                frame[mask.astype(bool)] = colour     # opaque fill
                mask_bin[mask.astype(bool)] = True    # update binary mask

        mask_info.append(mask_bin)
        out.write(frame)

    metadata["mask_info"] = mask_info

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
    model = YOLO(args.keypoint_model)  # Load the keypoint detection model

    frame_idx = 0
    keypoint_info = []
    while cap.isOpened():
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{num_frames}", end='\r', flush=True)

        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=args.confidence_threshold, verbose=False)

        frame_keypoints = []
        for person in results[0].keypoints.data:
            # shape: [num_keypoints, 3], where each row is [x, y, confidence]
            keypoints = person.cpu().numpy().tolist()
            frame_keypoints.append(keypoints)
        keypoint_info.append(frame_keypoints)

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
        metadata = process_video_mask(input_video_path, out, metadata, num_frames, args)
        print("Mask processing done.")
    metadata = process_video_tracker(cap, out, metadata, num_frames, args)
    print("Tracker processing done.")
    metadata = process_video_keypoints(cap2, metadata, num_frames, args)
    print("Keypoint processing done.")
        
    cap.release()
    cap2.release()
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