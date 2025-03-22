import cv2
import yaml
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Define area of interest for tracking for a single video")
    parser.add_argument("--video", type=str, required=True, help="Path to the video folder")
    parser.add_argument("--area_path", default="./areas", type=str, help="Path to the area of interest folder")
    parser.add_argument("--scale", default=1.0, type=float, help="Scale factor for display")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: The video folder {args.video} does not exist.")
        exit(1)
    if not os.path.isfile(args.video):
        print(f"ERROR: The video path {args.video} is not a file.")
        exit(1)
    if not os.path.exists(args.area_path):
        os.makedirs(args.area_path)
    elif not os.path.isdir(args.area_path):
        print(f"ERROR: The area path {args.area_path} is not a folder.")
        exit(1)
    if args.scale <= 0:
        print("ERROR: The scale factor must be positive.")
        exit(1)
    return args

def select_area(frame, scale=1.0):
    window_name = "Select Area of Interest (At least 3 points)"
    cv2.namedWindow(window_name)
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    area = []
    cv2.imshow(window_name, frame)

    def on_mouse(event, x, y, flags, param):
        frame_copy = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            area.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(area) > 0:
                area.pop()
        for point in area:
            cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)
        cv2.imshow(window_name, frame_copy)

    cv2.setMouseCallback(window_name, on_mouse, frame)

    print("Select the area of interest by clicking on the image. Press 'q' to finish.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if key == ord("q"):
            break

    cv2.destroyWindow(window_name)

    if len(area) < 3:
        print("ERROR: At least 3 points are needed to define the area of interest.")
        exit(1)
    
    for i in range(len(area)):
        area[i] = (int(area[i][0] / scale), int(area[i][1] / scale))
    return area

def main(args):
    video_fname = os.path.basename(args.video)
    video_name = os.path.splitext(video_fname)[0]
    area_path = os.path.join(args.area_path, video_name + ".yaml")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {args.video}")
        exit(1)
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read video {args.video}")
        exit(1)

    area = select_area(frame, args.scale)
    cap.release()

    with open(area_path, "w") as f:
        yaml.dump(area, f)
    print(f"Area of interest saved to {area_path}")

if __name__ == "__main__":
    args = get_args()
    main(args)