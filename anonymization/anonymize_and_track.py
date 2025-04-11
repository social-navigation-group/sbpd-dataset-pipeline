import os
import cv2
import toml
import yaml
import copy
import time
import argparse
import numpy as np
from byte_track_wrapper import ByteTrackWrapper

from define_area import select_area

def get_args():
    parser = argparse.ArgumentParser(description="Anonymize a video using ByteTrack")
    # Paths
    parser.add_argument("--model", default = "../../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", type = str, help = "Path to the YOLOX model")
    parser.add_argument("--experiment_config", default = "../../ByteTrack/exps/example/mot/yolox_x_mix_det.py", type = str, help = "ByteTrack experiment config file")
    parser.add_argument("--video", required = True, type = str, help = "Path to the folder that contains video files") # default="./videos"
    parser.add_argument("--output", default="./videos_anonymized", type = str, help = "Path to the folder that contains anonymized video files")
    parser.add_argument("--trajectory-output", default = "./trajectories", type = str, help = "Path to save the automated trajectories")
    parser.add_argument("--area-path", default = "./areas", type = str, help = "Path to the area files")

    # Anonymization options
    parser.add_argument("--no-blur", action = "store_true", help = "Do not anonymize the video (highest priority argument)")
    parser.add_argument("--blur-all", action = "store_true", help = "Blur entire video frames")
    parser.add_argument("--blur-black", action = "store_true", help = "Paint everything black")
    parser.add_argument("--shallow-size", default = 21, type = int, help = "Size of the shallow blur kernel for full frame anonymization")
    parser.add_argument("--blur-size", default = 41, type = int, help = "Size of the blur kernel")
    parser.add_argument("--blur-pct", default = 0.5, type = float, help = "Percentage of the bounding box to blur")
    parser.add_argument("--blur-min", default = 25, type = int, help = "Minimum pixels of the bounding box to blur")

    # Tracking parameters
    parser.add_argument("--no-track", action = "store_true", help = "Do not track objects")
    parser.add_argument("--restrict-area", action = "store_true", help = "Restrict tracking to a specific area")
    parser.add_argument("--scale", default = 0.5, type = float, help = "Scale factor for display")
    parser.add_argument("--persist", action = "store_true", help = "Persist tracking")
    # parser.add_argument("--iou", default = 0.7, type = float, help = "IoU threshold for tracking")
    # parser.add_argument("--tracker", default = "bytetrack.yaml", type = str, help = "Tracker configuration file") # No need to download this yaml
    parser.add_argument("--smooth-len", default = 7, type = int, help = "Length of the smoothing window")
    # If the pedestrians feet cannot be seen, their trajectories will still be recorded at the bottom of the screen.
    # This parameter allows to ignore these trajectories.
    parser.add_argument("--boundary-width", default = 10, type = int, help = "Ignore trajectories within boundary width of the bottom of the frame")
    parser.add_argument("--min-length", default = 30, type = int, help = "Minimum length of a trajectory to be recorded")

    # Trajectory options
    parser.add_argument("--traj-fps", default = 10, type = int, help = "Fps of the trajectories")

    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"ERROR: The model file {args.model} does not exist.")
        exit(1)
    # if not os.path.exists(args.video):
        # print(f"ERROR: The video folder {args.video} does not exist.")
        # exit(1)
    # if not os.path.isdir(args.video):
        # print(f"ERROR: The video path {args.video} is not a folder.")
        # exit(1)
    if not os.path.isfile(args.video):
        print(f"ERROR: The video file {args.video} does not exist.")
        exit(1)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not os.path.isdir(args.output):
        print(f"ERROR: The output path {args.output} is not a folder.")
        exit(1)
    if not os.path.exists(args.trajectory_output):
        os.makedirs(args.trajectory_output)
    elif not os.path.isdir(args.trajectory_output):
        print(f"ERROR: The trajectory output path {args.trajectory_output} is not a folder.")
        exit(1)
    if args.restrict_area:
        if not os.path.exists(args.area_path):
            os.makedirs(args.area_path)
        elif not os.path.isdir(args.area_path):
            print(f"ERROR: The area path {args.area_path} is not a folder.")
            exit(1)
        if args.scale <= 0:
            print("ERROR: The scale factor must be positive.")
            exit(1)
    if args.blur_pct < 0 or args.blur_pct > 1:
        print("ERROR: The blur percentage must be between 0 and 1.")
        exit(1)
    if args.blur_min < 0:
        print("ERROR: The minimum blur pixels must be non-negative.")
        exit(1)
    if args.boundary_width < 0:
        print("ERROR: The boundary width must be non-negative.")
        exit(1)
    if args.min_length < 0:
        print("ERROR: The minimum trajectory length must be non-negative.")
        exit(1)
    return args

def main(args):
    # for video_file in os.listdir(args.video):
    video_file = os.path.basename(args.video)
    video_test_file = video_file.lower()
    if (not video_test_file.endswith(".mp4")) and \
        (not video_test_file.endswith(".avi")) and \
        (not video_test_file.endswith(".mov")):
        return
        # continue

    # Process paths
    # video_path = os.path.join(args.video, video_file)
    video_path = args.video

    if not args.no_blur:
        output_path = os.path.join(args.output, video_file)
    if not args.no_track:
        traj_fname = os.path.splitext(video_file)[0] + ".toml"
        trajectory_path = os.path.join(args.trajectory_output, traj_fname)
        trajectory_video_path = os.path.join(args.trajectory_output, video_file)
    if args.restrict_area:
        area_fname = traj_fname.replace(".toml", ".yaml")
        area_path = os.path.join(args.area_path, area_fname)
    
    # Initialize videos
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}.")
        return
        # continue

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale video
    max_width = 1920
    max_height = 1080
    scale_w = frame_width / max_width if frame_width > max_width else 1.0
    scale_h = frame_height / max_height if frame_height > max_height else 1.0
    downscale_factor = max(scale_w, scale_h)
    downscale = downscale_factor > 1.0

    if downscale:
        new_width = int(frame_width / downscale_factor)
        new_height = int(frame_height / downscale_factor)
        print(f"Downscaling video from {frame_width}x{frame_height} to {new_width}x{new_height}")
    else:
        new_width = frame_width
        new_height = frame_height

    scaled_boundary_width = int(args.boundary_width / downscale_factor) if downscale else args.boundary_width
    
    if (fps % args.traj_fps) != 0:
        print(f"ERROR: The video {video_path} has a frame rate that is not a multiple of the trajectory frame rate.")
        print(f"Current frame rate: {fps}, trajectory frame rate: {args.traj_fps}")
        return

    interval = max(1, int(fps / args.traj_fps))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not args.no_blur:
        output = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    if not args.no_track:
        trajectory_video = cv2.VideoWriter(trajectory_video_path, fourcc, args.traj_fps, (new_width, new_height))

    # Main loop
    frame_id = 0
    trajectory_dict = {}
    id_list = []
    start_time = time.time()
    print(f"Start processing video: {video_file}")

    tracker = ByteTrackWrapper(
        model_path = args.model,
        exp_file = args.experiment_config
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"\rProcessing frame {frame_id}/{num_frames}", end = "")

        if downscale:
            frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)

        bbox_tlwh, track_ids = tracker.update(frame)
        save_frame = copy.deepcopy(frame)

        # load or create restrict areas
        if (frame_id == 1) and args.restrict_area:
            if os.path.exists(area_path):
                with open(area_path, 'r') as f:
                    area = yaml.load(f, Loader = yaml.FullLoader)
            else:
                # Use matplotlib to select points until right click TODO
                area = select_area(frame, args.scale)
                with open(area_path, 'w') as f:
                    yaml.dump(area, f)

            area = np.array(area, np.float32)
            if downscale:
                area /= downscale_factor
            area = area.astype(np.int32)
        
        for box_index in range(len(bbox_tlwh)):
            top_left_x, top_left_y, width, height = bbox_tlwh[box_index]
            object_id = track_ids[box_index]

            x1, y1, x2, y2 = int(top_left_x), int(top_left_y), int(top_left_x + width), int(top_left_y + height)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(new_width - 1, x2)
            y2 = min(new_height - 1, y2)
            if not ((x1 < x2) and (y1 < y2)):
                continue

            # Anonymize bounding box
            if (not args.blur_all) and (not args.no_blur):
                new_y2 = y1 + max((y2 - y1) * args.blur_pct, args.blur_min)
                new_y2 = min(new_height - 1, new_y2)
                x1, y1, x2, new_y2 = map(int, [x1, y1, x2, new_y2])
                if (y1 < new_y2):
                    if args.blur_black:
                        save_frame[y1:new_y2, x1:x2, :] = 0
                    else:
                        save_frame[y1:new_y2, x1:x2] = cv2.GaussianBlur(frame[y1:new_y2, x1:x2], (args.blur_size, args.blur_size), 0)

            # Record tracking for trajectories
            if (not args.no_track) and ((frame_id - 1) % interval == 0):
                center_x = top_left_x + width/ 2
                bottom_y = top_left_y + height 
                coordinate = [float(center_x), float(bottom_y)]
                human_name = f"human{object_id}"

                if (object_id != -1) and \
                    ((not args.restrict_area) or (cv2.pointPolygonTest(area, (center_x, bottom_y), False) == True)) and \
                    (bottom_y < new_height - scaled_boundary_width):
                    
                    traj_frame_id = frame_id // interval
                    if (object_id not in id_list):
                        id_list.append(object_id)
                        trajectory_dict[human_name] = {}
                        trajectory_dict[human_name]["traj_start"] = traj_frame_id
                        trajectory_dict[human_name]["trajectories"] = [coordinate]
                        trajectory_dict[human_name]["frame_number"] = [traj_frame_id]
                        trajectory_dict[human_name]["human_context"] = None
                    else:
                        trajectory_dict[human_name]["trajectories"].append(coordinate)
                        trajectory_dict[human_name]["frame_number"].append(traj_frame_id)
        
        if (args.blur_all) and (not args.no_blur):
            save_frame = cv2.GaussianBlur(save_frame, (args.shallow_size, args.shallow_size), 0)

        if (not args.no_track) and ((frame_id - 1) % interval == 0):
            if args.restrict_area:
                save_frame = cv2.polylines(save_frame, [area], True, (255, 255, 255), 1)
            trajectory_video.write(save_frame)
        if (not args.no_blur):
            output.write(save_frame)

    if not args.no_track:
        # filter short trajectories
        for human_name in list(trajectory_dict.keys()):
            if len(trajectory_dict[human_name]["trajectories"]) < args.min_length:
                del trajectory_dict[human_name]

        # use linear interpolation to fill in missing frames
        for human_name in trajectory_dict:
            trajectory = trajectory_dict[human_name]["trajectories"]
            frame_number = trajectory_dict[human_name]["frame_number"]
            new_trajectory = []
            new_frame_number = []
            for i in range(len(trajectory) - 1):
                new_trajectory.append(trajectory[i])
                new_frame_number.append(frame_number[i])
                if frame_number[i + 1] - frame_number[i] > 1:
                    for j in range(frame_number[i] + 1, frame_number[i + 1]):
                        ratio = (j - frame_number[i]) / (frame_number[i + 1] - frame_number[i])
                        x = trajectory[i][0] + ratio * (trajectory[i + 1][0] - trajectory[i][0])
                        y = trajectory[i][1] + ratio * (trajectory[i + 1][1] - trajectory[i][1])
                        new_trajectory.append([x, y])
                        new_frame_number.append(j)
            new_trajectory.append(trajectory[-1])
            new_frame_number.append(frame_number[-1])
            trajectory_dict[human_name]["trajectories"] = new_trajectory
            trajectory_dict[human_name]["frame_number"] = new_frame_number

        # smooth trajectories
        smooth_len = args.smooth_len // 2
        for human_name in trajectory_dict:
            trajectory = trajectory_dict[human_name]["trajectories"]
            smooth_trajectory = []
            for i in range(len(trajectory)):
                x = 0
                y = 0
                count = 0
                for j in range(max(0, i - smooth_len), min(len(trajectory), i + smooth_len)):
                    x += trajectory[j][0]
                    y += trajectory[j][1]
                    count += 1
                smooth_trajectory.append([x / count, y / count])
            trajectory_dict[human_name]["trajectories"] = smooth_trajectory

        with open(trajectory_path, 'w') as f:
            toml.dump(trajectory_dict, f)

    if not args.no_blur:
        output.release()
    if not args.no_track:
        trajectory_video.release()
    cap.release()
    print(f"\nFinished processing video: {video_file}")
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    return

if __name__ == "__main__":
    args = get_args()
    main(args)