import cv2
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def main(model_path, video_path, bbox_path):
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}.")
        return

    with open(bbox_path, 'w') as bbox_file:
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            results = model.track(frame, persist = True, iou = 0.8, show = False, tracker = "bytetrack.yaml")

            for box in results[0].boxes:
                print(box)
                if box.cls[0] == 0:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    class_id = int(box.cls[0].item()) 
                    confidence = float(box.conf[0].item()) 
                    object_id = int(box.id[0]) if box.id is not None else -1
                    bbox_file.write(f"{frame_id}, {x1}, {y1}, {x2}, {y2}, {class_id}, {confidence:.4f}, {object_id}\n")
            exit()

    # RELEASE RESOURCES
    cap.release()

if __name__ == "__main__":
    bbox_file = "./btrack_bboxes.txt"
    video_path = "./example_small_data.avi"
    model_path = "./yolo11x.pt" # download the model: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

    main(model_path, video_path, bbox_file)
