import os
import argparse
import subprocess
import sys

from cli_args import add_anonymize_arguments

def main():
    parser = argparse.ArgumentParser(description = "Run anonymize script on multiple videos.")
    parser.add_argument("--video-folder", default = "./videos", type=str, help = "Folder containing video files.")
    parser.add_argument("--anonymize-script", default = "./anonymize_and_track.py", type = str, help = "Path to anonymize script.")
    add_anonymize_arguments(parser, include_video = False)

    args = parser.parse_args()

    videos = [f for f in os.listdir(args.video_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]

    for video in videos:
        video_path = os.path.join(args.video_folder, video)
        print(f"\nProcessing video: {video}")

        cmd = [
            sys.executable, args.anonymize_script,
            "--video", video_path,
            "--model", args.model,
            "--experiment-config", args.experiment_config,
            "--output", args.output,
            "--trajectory-output", args.trajectory_output,
            "--area-path", args.area_path,
            "--shallow-size", str(args.shallow_size),
            "--blur-size", str(args.blur_size),
            "--blur-pct", str(args.blur_pct),
            "--blur-min", str(args.blur_min),
            "--scale", str(args.scale),
            "--smooth-len", str(args.smooth_len),
            "--boundary-width", str(args.boundary_width),
            "--min-length", str(args.min_length),
            "--traj-fps", str(args.traj_fps),
            "--debug-frames", str(args.debug_frames),
        ]

        # Pass along arguments explicitly
        if args.blur_all:
            cmd.append("--blur-all")
        if args.no_blur:
            cmd.append("--no-blur")
        if args.blur_black:
            cmd.append("--blur-black")
        if args.restrict_area:
            cmd.append("--restrict-area")
        if args.no_track:
            cmd.append("--no-track")
        if args.persist:
            cmd.append("--persist")
        if args.debug:
            cmd.append("--debug")

        subprocess.run(cmd, check=True)
        print(f"Finished video: {video}")

if __name__ == "__main__":
    main()
