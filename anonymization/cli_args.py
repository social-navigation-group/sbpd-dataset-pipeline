import argparse


def add_anonymize_arguments(parser: argparse.ArgumentParser, include_video: bool = True) -> argparse.ArgumentParser:
    # Paths
    parser.add_argument("--model", default = "../../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", type = str, help = "Path to the YOLOX model")
    parser.add_argument("--experiment-config", default = "../../ByteTrack/exps/example/mot/yolox_x_mix_det.py", type = str, help = "ByteTrack experiment config file")
    if include_video:
        parser.add_argument("--video", required = True, type = str, help = "Path to the folder that contains video files")
    parser.add_argument("--output", default = "./videos_anonymized", type = str, help = "Path to the folder that contains anonymized video files")
    parser.add_argument("--trajectory-output", default = "./trajectories", type = str, help = "Path to save the automated trajectories")
    parser.add_argument("--area-path", default = "./areas", type = str, help = "Path to the area files")

    # Anonymization options
    parser.add_argument("--no-blur", action = "store_true", help = "Do not anonymize the video (highest priority argument)")
    parser.add_argument("--blur-all", action = "store_true", help = "Blur entire video frames")
    parser.add_argument("--blur-black", action = "store_true", help = "Paint everything black")
    parser.add_argument("--shallow-size", default = 81, type = int, help = "Size of the shallow blur kernel for full frame anonymization")
    parser.add_argument("--blur-size", default = 41, type = int, help = "Size of the blur kernel")
    parser.add_argument("--blur-pct", default = 0.5, type = float, help = "Percentage of the bounding box to blur")
    parser.add_argument("--blur-min", default = 25, type = int, help = "Minimum pixels of the bounding box to blur")

    # Tracking parameters
    parser.add_argument("--no-track", action = "store_true", help = "Do not track objects")
    parser.add_argument("--restrict-area", action = "store_true", help = "Restrict tracking to a specific area")
    parser.add_argument("--scale", default = 0.5, type = float, help = "Scale factor for display")
    parser.add_argument("--persist", action = "store_true", help = "Persist tracking")
    parser.add_argument("--smooth-len", default = 7, type = int, help = "Length of the smoothing window")
    parser.add_argument("--boundary-width", default = 10, type = int, help = "Ignore trajectories within boundary width of the bottom of the frame")
    parser.add_argument("--min-length", default = 30, type = int, help = "Minimum length of a trajectory to be recorded")

    # Trajectory options
    parser.add_argument("--traj-fps", default = 10, type = int, help = "Fps of the trajectories")

    # Debug options
    parser.add_argument("--debug", action = "store_true", help = "Process only a limited number of frames.")
    parser.add_argument("--debug-frames", default = 1000, type = int, help = "Number of frames to process in debug mode.")
    return parser