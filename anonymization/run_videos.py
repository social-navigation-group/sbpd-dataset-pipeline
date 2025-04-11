import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description = "Run anonymize script on multiple videos.")
    parser.add_argument("--video-folder", default = "./videos", type=str, help = "Folder containing video files.")
    parser.add_argument("--anonymize-script", default = "./anonymize_and_track.py", type = str, help = "Path to anonymize script.")

    # Forwarded arguments
    parser.add_argument("--no-blur", action = "store_true")
    parser.add_argument("--no-track", action = "store_true")
    parser.add_argument("--blur-all", action = "store_true")
    parser.add_argument("--restrict-area", action = "store_true")

    args = parser.parse_args()

    videos = [f for f in os.listdir(args.video_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]

    for video in videos:
        video_path = os.path.join(args.video_folder, video)
        print(f"\nProcessing video: {video}")

        cmd = [
            "python", args.anonymize_script,
            "--video", video_path,
        ]

        # Pass along arguments explicitly
        if args.blur_all:
            cmd.append("--blur-all")
        if args.no_blur:
            cmd.append("--no-blur")
        if args.restrict_area:
            cmd.append("--restrict-area")
        if args.no_track:
            cmd.append("--no-track")

        subprocess.run(cmd, check=True)
        print(f"Finished video: {video}")

if __name__ == "__main__":
    main()
