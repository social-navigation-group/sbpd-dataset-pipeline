import os
import argparse
import subprocess
import csv

'''
Folder structure:

video-folder
    ├── raw
    │   ├── video1.mp4
    │   ├── video2.avi
    ├── trimmed
    │   ├── video1_trimmed.mp4
    │   ├── video2_trimmed.avi
    
trajectory-output
    ├── anonymized_10fps
    │   ├── video1_trimmed.mp4
    │   ├── video2_trimmed.avi
    ├── trajectories
    │   ├── video1_trimmed.toml
    │   ├── video2_trimmed.toml
'''



def main():
    parser = argparse.ArgumentParser(description = "Run anonymize script on multiple videos.")
    parser.add_argument("--video-folder", default = "./videos", type=str, help = "Folder containing video files.")
    parser.add_argument("--output", default = "./anonymized_videos", type=str, help = "Folder to save anonymized videos.")
    parser.add_argument("--trajectory-output", default = "./trajectories", type=str, help = "Folder to save trajectory videos.")
    parser.add_argument("--anonymize-script", default = "./anonymize_and_track.py", type = str, help = "Path to anonymize script.")
    parser.add_argument("--trim", action = "store_true", help = "Whether to trim the videos before anonymization. Ensure there is a file called video_limits.csv")
    # Forwarded arguments
    parser.add_argument("--no-blur", action = "store_true")
    parser.add_argument("--no-track", action = "store_true")
    parser.add_argument("--blur-all", action = "store_true")
    parser.add_argument("--blur-black", action = "store_true")
    parser.add_argument("--restrict-area", action = "store_true")

    args = parser.parse_args()

    videos = [f for f in os.listdir(os.path.join(args.video_folder,'raw')) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    if args.trim:
        print("Trimming videos before anonymization...")
        # Assuming a script or function to trim videos is available
        video_limits = {}
        with open("video_limits.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_limits[row["video"]] = {
                    "start_time": row["start"],
                    "end_time": row["end"]
                }
    for video in videos:
        video_path = os.path.join(args.video_folder,'raw', video)
        print(f"\nProcessing video: {video}")
        if args.trim:
            #remove the file extension for the trimmed video
            video_name = os.path.splitext(video)[0]
            extension = os.path.splitext(video)[1]
            print(video)
            if video_name in video_limits:
                start_time = video_limits[video_name]["start_time"]
                end_time = video_limits[video_name]["end_time"]
                trimmed_video_path = os.path.join(args.video_folder, 'trimmed',f"{video_name}_trimmed{extension}")
                if os.path.exists(trimmed_video_path):
                    print(f"Trimmed video {trimmed_video_path} already exists, skipping trimming.")
                else:
                    print(f"Trimming video from {start_time} to {end_time} and saving to {trimmed_video_path}")
                    cmd = [
                        "ffmpeg", "-i", video_path, "-ss", start_time, "-to", end_time,
                        "-c:v", "copy", "-c:a", "copy", trimmed_video_path
                    ]
                    subprocess.run(cmd, check=True)
                video_path = trimmed_video_path
            else:
                print(f"No limits found for {video}")
                exit(0)
                
        if os.path.exists(os.path.join(args.trajectory_output,'anonymized_10fps',f"{video_name}_trimmed{extension}")) and os.path.exists(os.path.join(args.trajectory_output,'trajectories',f"{video_name}_trimmed.toml")):
            print(f"Anonymized video {video} already exists in trajectory folder, skipping.")
            continue
        cmd = [
            "python3", args.anonymize_script,
            "--video", video_path, "--output", args.output,
            "--trajectory-output", args.trajectory_output
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

        subprocess.run(cmd, check=True)
        print(f"Finished video: {video}")

if __name__ == "__main__":
    main()
