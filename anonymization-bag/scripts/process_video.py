# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import cv2
from byte_track_wrapper import ByteTrackWrapper
import numpy as np 

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    # parser.add_argument(
    #     "--base-path",
    #     help="Base path to the bag files and video files",
    # )
    parser.add_argument(
        "--video_path",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--out_path",
        help="Path to the output video file",
    )
    # Detectron2 arguments
    parser.add_argument(
        "--use-mask", 
        action = "store_true", 
        help = "Instance segmentation on the video (Higher priority than blur)",
        default=False
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--detectron-config",
        default="/bag_ws/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"],
        nargs=argparse.REMAINDER,
    )

    # ByteTrack arguments
    parser.add_argument(
        "--use-blur", 
        action = "store_true", 
        help = "Anonymize the video (Not performed if also instance segmentation)",
        default=True
    )
    parser.add_argument(
        "--bytetrack-model", 
        default = "/home/shashank/code/packages/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", 
        type = str, 
        help = "Path to the YOLOX model"
    )
    parser.add_argument(
        "--bytetrack_config", 
        default = "/home/shashank/code/packages/ByteTrack/exps/example/mot/yolox_x_mix_det.py", 
        type = str, 
        help = "ByteTrack experiment config file"
    )
    parser.add_argument(
        "--blur-size", 
        default = 41, 
        type = int, 
        help = "Size of the blur kernel",
    )
    parser.add_argument(
        "--blur-pct", 
        default = 0.25, 
        type = float, 
        help = "Percentage of the bounding box to blur",
    )
    parser.add_argument(
        "--blur-min", 
        default = 25, 
        type = int, 
        help = "Minimum pixels of the bounding box to blur",
    )

    args = parser.parse_args()
    # if args.base_path is None:
    #     raise ValueError("Base path is required.")
    if args.use_blur and args.use_mask:
        print("Warning: Both --use-blur and --use-mask are set. --use-mask will take priority.")
        args.use_blur = False
    return args

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def process_video(input_video_path, output_video_path, args):
    """
    Process a single video using OpenCV.
    For demonstration, this function reads the video and writes it back unchanged.
    Insert your own processing (filters, detection, etc.) here.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {input_video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0
    if args.use_blur:
        tracker = ByteTrackWrapper(args.bytetrack_model, args.bytetrack_config)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # TODO: Here you can add any frame processing you need. 
        save_frame = copy.deepcopy(frame)
        if args.use_blur:
            bbox_tlwh, track_ids = tracker.update(frame)
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
                #print(f"Blurring bounding box: {x1}, {y1}, {x2}, {new_y2}")
                save_frame[y1:new_y2, x1:x2] = cv2.GaussianBlur(frame[y1:new_y2, x1:x2], (args.blur_size, args.blur_size), 0)

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{num_frames}", end='\r', flush=True)
        out.write(plot_tracking(save_frame, bbox_tlwh, track_ids))
        
    cap.release()
    out.release()
    return

def main(args):
    args = get_parser()
    process_video(args.video_path,args.out_path,args)
    return

if __name__ == "__main__":
    args = get_parser()
    main(args)