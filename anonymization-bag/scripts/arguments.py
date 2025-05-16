import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Anonymization script for FPV data")

    parser.add_argument(
        "--base-path",
        help="Base path to the bag files and video files",
    )

    # Detectron2 arguments
    parser.add_argument(
        "--use-mask", 
        action = "store_true", 
        help = "Instance segmentation on the video (Higher priority than blur)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--segmentation-config",
        default="/bag_ws/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to segmentation config file",
    )
    parser.add_argument(
        "--segmentation-weights",
        help="segmentation model weights",
        default="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--keypoint-config",
        default="/bag_ws/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to keypoint config file",
    )
    parser.add_argument(
        "--keypoint-weights",
        help="keypoint model weights",
        default="detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl",
        nargs=argparse.REMAINDER,
    )

    # ByteTrack arguments
    parser.add_argument(
        "--use-blur", 
        action = "store_true", 
        help = "Anonymize the video (Not performed if also instance segmentation)"
    )
    parser.add_argument(
        "--bytetrack-model", 
        default = "../ByteTrack/pretrained/bytetrack_x_mot17.pth.tar", 
        type = str, 
        help = "Path to the YOLOX model"
    )
    parser.add_argument(
        "--bytetrack_config", 
        default = "../ByteTrack/exps/example/mot/yolox_x_mix_det.py", 
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
    if args.base_path is None:
        raise ValueError("Base path is required.")
    if args.use_blur and args.use_mask:
        print("Warning: Both --use-blur and --use-mask are set. --use-mask will take priority.")
        args.use_blur = False
    return args