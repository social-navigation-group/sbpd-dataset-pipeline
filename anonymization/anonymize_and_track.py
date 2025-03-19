import cv2
import argparse
import os
import toml
import copy
import time
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(description="Anonymize a video using YOLO tracking")
    # Paths
    parser.add_argument("--model", default="./yolo11x.pt", type=str, help="Path to the YOLO model")
    parser.add_argument("--video", default="./videos", type=str, help="Path to the folder that contains video files")
    parser.add_argument("--output", default="./videos_anonymized", type=str, help="Path to the folder that contains anonymized video files")
    parser.add_argument("--trajectory-output", default="./trajectories", type=str, help="Path to save the automated trajectories")

    # Anonymization options
    parser.add_argument("--blur-all", action="store_true", help="Blur entire video frames")
    parser.add_argument("--blur-black", action="store_true", help="Paint everything black")
    parser.add_argument("--blur-size", default=41, type=int, help="Size of the blur kernel")
    parser.add_argument("--blur-pct", default=0.5, type=float, help="Percentage of the bounding box to blur")

    # Tracking parameters
    parser.add_argument("--iou", default=0.8, type=float, help="IoU threshold for tracking")
    parser.add_argument("--tracker", default="bytetrack.yaml", type=str, help="Tracker configuration file") # No need to download this yaml
    parser.add_argument("--smooth-len", default=7, type=int, help="Length of the smoothing window")

    # Trajectory options
    parser.add_argument("--traj-fps", default=10, type=int, help="Fps of the trajectories")

    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"ERROR: The model file {args.model} does not exist.")
        exit(1)
    if not os.path.exists(args.video):
        print(f"ERROR: The video folder {args.video} does not exist.")
        exit(1)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.trajectory_output):
        os.makedirs(args.trajectory_output)
    return args

def main(args):
    model = YOLO(args.model)

    for video_file in os.listdir(args.video):
        video_test_file = video_file.lower()
        if (not video_test_file.endswith(".mp4")) and (not video_test_file.endswith(".avi")):
            continue

        # Process paths
        video_path = os.path.join(args.video, video_file)
        output_path = os.path.join(args.output, video_file)
        if video_test_file.endswith(".mp4"):
            traj_fname = video_test_file.replace(".mp4", ".toml")
        else:
            traj_fname = video_test_file.replace(".avi", ".toml")
        trajectory_path = os.path.join(args.trajectory_output, traj_fname)
        trajectory_video_path = os.path.join(args.trajectory_output, video_file)

        # Initialize videos
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video {video_path}.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (fps % args.traj_fps) != 0:
            print(f"ERROR: The video {video_path} has a frame rate that is not a multiple of the trajectory frame rate.")
            print(f"Current frame rate: {fps}, trajectory frame rate: {args.traj_fps}")
            continue
        interval = max(1, int(fps / args.traj_fps))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        trajectory_video = cv2.VideoWriter(trajectory_video_path, fourcc, args.traj_fps, (frame_width, frame_height))

        # Main loop
        frame_id = 0
        trajectory_dict = {}
        id_list = []
        start_time = time.time()
        print(f"Start processing video: {video_file}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            print(f"\rProcessing frame {frame_id}/{num_frames}", end="")

            results = model.track(frame, persist = True, iou = args.iou, show = False, tracker = args.tracker, verbose = False)
            save_frame = copy.deepcopy(frame)

            for box in results[0].boxes:
                if box.cls[0] == 0:  # Only track humans (label 0)
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox

                    # Anonymize bounding box
                    if not args.blur_all:
                        new_y2 = y1 + (y2 - y1) * args.blur_pct
                        x1, y1, x2, new_y2 = map(int, [x1, y1, x2, new_y2])
                        if args.blur_black:
                            save_frame[y1:new_y2, x1:x2, :] = 0
                        else:
                            save_frame[y1:new_y2, x1:x2] = cv2.GaussianBlur(frame[y1:new_y2, x1:x2], (args.blur_size, args.blur_size), 0)

                    # Record tracking for trajectories
                    if (frame_id - 1) % interval == 0:
                        coordinate = [(float(x1) + float(x2)) / 2, float(y2)]
                        object_id = int(box.id[0]) if box.id is not None else -1
                        human_name = f"human{object_id}"

                        if (object_id != -1):
                            if (object_id not in id_list):
                                id_list.append(object_id)
                                trajectory_dict[human_name] = {}
                                trajectory_dict[human_name]["traj_start"] = frame_id // interval
                                trajectory_dict[human_name]["trajectories"] = [coordinate]
                                trajectory_dict[human_name]["human_context"] = None
                            else:
                                trajectory_dict[human_name]["trajectories"].append(coordinate)
            
            if args.blur_all:
                save_frame = cv2.GaussianBlur(save_frame, (args.blur_size, args.blur_size), 0)

            if (frame_id - 1) % interval == 0:
                trajectory_video.write(save_frame)
            output.write(save_frame)

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

        output.release()
        trajectory_video.release()
        cap.release()
        print(f"\nFinished processing video: {video_file}")
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")

    return

if __name__ == "__main__":
    args = get_args()
    main(args)