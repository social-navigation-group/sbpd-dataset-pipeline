#https://github.com/robodhruv/visualnav-transformer

import os
import pickle
import argparse
import cv2
from tqdm import tqdm
import yaml
import json
import multiprocessing
from functools import partial
from process_data_utils import get_images_pcl_and_odom
import numpy as np
from pprint import pprint
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rosbags.typesys import get_types_from_msg,get_typestore,Stores
from ros2_utils import get_synced_raw_messages_from_bag
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, CompressedImage, LaserScan
from velodyne_msgs.msg import VelodyneScan
from std_msgs.msg import String
import pickle as pkl
from tf2_msgs.msg import TFMessage
from IPython import embed    
from PIL import Image
from scipy.spatial.transform import Rotation

BINARY_MASK_MSG = """
std_msgs/Header header
int32 width
int32 height
bool[] data
"""
typestore = get_typestore(Stores.ROS2_HUMBLE)
typestore.register(get_types_from_msg(BINARY_MASK_MSG, "custom_msgs/msg/BinaryMask"))
BinaryMask = typestore.types['custom_msgs/msg/BinaryMask']

def process_bag_file(bag_path,config):
    traj_name = bag_path.split("/")[-1].replace('.bag','')
    #get the relative path of the bag file with respect to input_dir
    root = os.path.relpath(os.path.dirname(bag_path), args.input_dir)
    traj_name_i = traj_name + f"_{0}"
    traj_folder_i = os.path.join(args.output_dir,root,traj_name_i)
    if os.path.exists(traj_folder_i) and args.contin:
        print(f"{bag_path} already processed. Skipping...")
        return True 
    
    type_map = {}
    for topic in config[args.dataset_name]['imtopics']:
        type_map[topic] = CompressedImage
    for topic in config[args.dataset_name]['odomtopics']:
        type_map[topic] = Odometry
    for topic in config[args.dataset_name]['pcltopics']:
        if 'packet' in topic:
            type_map[topic] = VelodyneScan
        else:
            type_map[topic] = PointCloud2
    for topic in config[args.dataset_name]['trackingtopics']:
        type_map[topic] = Detection2DArray
    for topic in config[args.dataset_name]['scantopics']:
        type_map[topic] = LaserScan
    # Add mask topics to type map if available
    if 'masktopics' in config[args.dataset_name]:
        # try:
        for topic in config[args.dataset_name]['masktopics']:
            type_map[topic] = BinaryMask
        # except ImportError:
        #     print("Warning: custom_msgs.msg.BinaryMask not available. Mask topics will be skipped.")
    type_map['/tf_static'] = TFMessage
    if 'speech_topic' in config[args.dataset_name]:
        type_map[config[args.dataset_name]['speech_topic']] = String
    
    print(f"Processing {bag_path}...")
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('cdr','cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    storage_filter = StorageFilter(topics = list(type_map.keys()))
    reader.set_filter(storage_filter)
    b = (reader,type_map)
    timestamps,synced_imdata, synced_odomdata, synced_pcldata, synced_scandata, synced_trackingdata, synced_maskdata, speech_data = get_synced_raw_messages_from_bag(
        b = b,
        imtopics = config[args.dataset_name]["imtopics"],
        pcltopics = config[args.dataset_name]["pcltopics"] if config[args.dataset_name]["process_pcl"] else [],
        odomtopics = config[args.dataset_name]["odomtopics"],
        scantopics = config[args.dataset_name]["scantopics"] if config[args.dataset_name]["process_scan"] else [], 
        trackingtopics = config[args.dataset_name]["trackingtopics"] if config[args.dataset_name]["process_tracking"] else [],
        masktopics = config[args.dataset_name].get("masktopics", []) if config[args.dataset_name].get("process_masks", False) else [],
        speechtopics = config[args.dataset_name].get('speech_topic',None),
        typestore = typestore
    )
    
    # Check if we got any data back
    if synced_imdata is None or synced_odomdata is None:
        print(f"{bag_path} did not have synced_imdata and synced_odomdata. Skipping...")
        return False
    
    # Print statistics - handle multi-camera case
    if isinstance(synced_imdata, dict):
        total_images = sum(len(img_list) for img_list in synced_imdata.values())
        print(f"Found {total_images} total images from {len(synced_imdata)} cameras")
        for topic, img_list in synced_imdata.items():
            print(f"  {topic}: {len(img_list)} images")
    else:
        print(f"Found {len(synced_imdata)} images")
    
    # Print mask statistics if available
    if synced_maskdata and isinstance(synced_maskdata, dict):
        total_masks = sum(len(mask_list) for mask_list in synced_maskdata.values())
        print(f"Found {total_masks} total masks from {len(synced_maskdata)} mask topics")
        for topic, mask_list in synced_maskdata.items():
            print(f"  {topic}: {len(mask_list)} masks")
    
    print(f" {len(synced_odomdata)} odom,\n {len(synced_pcldata)} pcl,\n {len(synced_scandata)} scans,\n {len(synced_trackingdata)}  tracks \n {len(timestamps)} timestamps in {bag_path}")


    bag_img_data, bag_traj_data, bag_pcl_data, bag_scan_data, bag_track_data, bag_mask_data, bag_speech_data= get_images_pcl_and_odom(
        synced_imdata,
        synced_odomdata,
        synced_pcldata if config[args.dataset_name]["process_pcl"] else [],
        synced_scandata if config[args.dataset_name]["process_scan"] else [],
        synced_trackingdata if config[args.dataset_name]["process_tracking"] else [],
        synced_maskdata if config[args.dataset_name].get("process_masks", False) else {},
        speech_data,
        img_process_func=config[args.dataset_name]["img_process_func"],
        pcl_process_func=config[args.dataset_name]["pcl_process_func"],
        odom_process_func=config[args.dataset_name]["odom_process_func"],
        scan_process_func=config[args.dataset_name]["scan_process_func"], 
        tracking_process_func=config[args.dataset_name]["tracking_process_func"],
        mask_process_func=config[args.dataset_name].get("mask_process_func"),
        speech_process_func=config[args.dataset_name].get("speech_process_func"),
        ang_offset=config[args.dataset_name]["ang_offset"],
        rosversion = 2,
    )
    
    print(f"""{(len(bag_img_data) == 0 and len(config[args.dataset_name]["imtopics"])!=0) },      
          {bag_traj_data is None},
          {(bag_pcl_data is None and config[args.dataset_name]["process_pcl"])},
          {(bag_scan_data is None and config[args.dataset_name]["process_scan"])},
          {(bag_track_data is None and config[args.dataset_name]["process_tracking"])},
          {(bag_mask_data is None and config[args.dataset_name].get("process_masks", False))}""")

    if (len(bag_img_data) == 0 and len(config[args.dataset_name]["imtopics"])!=0) or \
        bag_traj_data is None or (bag_pcl_data is None and config[args.dataset_name]["process_pcl"]) or\
            (bag_scan_data is None and config[args.dataset_name]["process_scan"]) or\
                (bag_track_data is None and config[args.dataset_name]["process_tracking"]) or\
                    (bag_mask_data is None and config[args.dataset_name].get("process_masks", False)):
        print(
        f"{bag_path} did not have all processed data. Skipping..."
        )
        print("len of bag_img_data:", len(bag_img_data) if bag_img_data is not None else 0)
        print("len of bag_traj_data:", len(bag_traj_data) if bag_traj_data is not None else 0)
        print("len of bag_pcl_data:", len(bag_pcl_data) if bag_pcl_data is not None else 0)
        print("len of bag_scan_data:", len(bag_scan_data) if bag_scan_data is not None else 0)
        print("len of bag_track_data:", len(bag_track_data) if bag_track_data is not None else 0)
        print("len of bag_mask_data:", len(bag_mask_data) if bag_mask_data is not None else 0)
        #embed()
        return False
    else:
        print("Saving Processed Data...")
        
    traj_pos = bag_traj_data["position"]
    traj_yaws = bag_traj_data["yaw"]
    traj_linear_vels = bag_traj_data["linear_vel"]
    traj_angular_vels = bag_traj_data["angular_vel"]
    trajs = {"position": [], "yaw": [], "linear_vel": [], "angular_vel": []}
    if not os.path.exists(traj_folder_i):
        os.makedirs(traj_folder_i)
        
        # Create camera-specific directories if multiple cameras
        if isinstance(bag_img_data, dict) and len(bag_img_data) > 1:
            for topic in bag_img_data.keys():
                camera_name = topic.replace('/', '_').replace('compressed', '').strip('_')
                os.makedirs(os.path.join(traj_folder_i, f"imgs_{camera_name}"))
        else:
            os.makedirs(os.path.join(traj_folder_i, "imgs"))
        
        # Create mask-specific directories if masks are available
        if bag_mask_data and config[args.dataset_name].get("process_masks", False):
            if isinstance(bag_mask_data, dict) and len(bag_mask_data) > 1:
                for topic in bag_mask_data.keys():
                    camera_name = topic.replace('/', '_').replace('masks_', '').replace('compressed', '').strip('_')
                    os.makedirs(os.path.join(traj_folder_i, f"masks_{camera_name}"))
            else:
                os.makedirs(os.path.join(traj_folder_i, "masks"))
        
        if bag_pcl_data:
            os.makedirs(os.path.join(traj_folder_i, "pcd"))
        if bag_scan_data:
            os.makedirs(os.path.join(traj_folder_i, "scans"))  # New directory for scans

    tracks=[]
    timestamps_filtered = []
    # Determine number of frames
    if len(bag_img_data) == 0:
        num_frames = len(bag_mask_data) if bag_mask_data else len(traj_pos)
    elif isinstance(bag_img_data, dict):
        # Use the camera with the most frames as reference
        num_frames = max(len(img_list) for img_list in bag_img_data.values())
        reference_camera = max(bag_img_data.keys(), key=lambda k: len(bag_img_data[k]))
        print(f"Using {reference_camera} as reference camera with {num_frames} frames")
    elif isinstance(bag_img_data, list):
        num_frames = len(bag_img_data)
    else:
        print("Unexpected format for bag_img_data. Skipping...")
        return False
    
    n = 0
    for i in tqdm(range(max(config[args.dataset_name]['start_slack'], 0), min(num_frames, len(traj_pos)) - config[args.dataset_name]['end_slack'])):
        # Save images - handle multi-camera case
        if len(bag_img_data) > 1:
            if isinstance(bag_img_data, dict):
                # Multi-camera case
                for topic, img_list in bag_img_data.items():
                    if i < len(img_list):  # Check if this camera has data at this index
                        camera_name = topic.replace('/', '_').replace('compressed', '').strip('_')
                        img_list[i].save(os.path.join(traj_folder_i, f'imgs_{camera_name}', f"{n}.jpg"))
            else:
                # Single camera case (backward compatibility)
                if isinstance(bag_img_data, dict):
                    # Single camera in dict format
                    img_list = next(iter(bag_img_data.values()))
                else:
                    # Legacy list format
                    img_list = bag_img_data
                
                if i < len(img_list):
                    img_list[i].save(os.path.join(traj_folder_i, 'imgs', f"{n}.jpg"))
        
        # Save masks - handle multi-mask case
        if bag_mask_data and config[args.dataset_name].get("process_masks", False):
            if isinstance(bag_mask_data, dict) and len(bag_mask_data) > 1:
                # Multi-mask case
                for topic, mask_list in bag_mask_data.items():
                    if i < len(mask_list):  # Check if this mask topic has data at this index
                        camera_name = topic.replace('/', '_').replace('masks_', '').replace('compressed', '').strip('_')
                        # Save mask as a binary image
                        #mask_img = Image.fromarray((mask_list[i] * 255).astype(np.uint8), mode='L')
                        #mask_img.save(os.path.join(traj_folder_i, f'masks_{camera_name}', f"{n}.png"))
                        cv2.imwrite(os.path.join(traj_folder_i, f'masks_{camera_name}', f"{n}.jpg"),mask_list[i]*255)
            else:
                # Single mask case (backward compatibility)
                if isinstance(bag_mask_data, dict):
                    # Single mask in dict format
                    mask_list = next(iter(bag_mask_data.values()))
                else:
                    # Legacy list format
                    mask_list = bag_mask_data
                
                if i < len(mask_list):
                    #np.save(os.path.join(traj_folder_i, 'masks', f"{n}.npy"), mask_list[i])
                    #mask_img = Image.fromarray((mask_list[i] * 255).astype(np.uint8), mode='L')
                    cv2.imwrite(os.path.join(traj_folder_i, f'masks', f"{n}.jpg"),mask_list[i]*255)

        trajs["position"].append(traj_pos[i])
        trajs["yaw"].append(np.array(traj_yaws[i]))
        trajs["linear_vel"].append(np.array(traj_linear_vels[i]))
        trajs["angular_vel"].append(np.array(traj_angular_vels[i]))
        timestamps_filtered.append(timestamps[i])

        if config[args.dataset_name]["process_tracking"] and bag_track_data:
            try:
                if bag_track_data[i] is None:
                    tracks.append((None,None))
                else:
                    tracks.append((bag_track_data[i][0], bag_track_data[i][1]))
            except Exception as e:
                print(e)
                print(f"Error processing tracks for {bag_path}. Skipping...")
                embed()
                continue

        
        if config[args.dataset_name]["process_scan"]:
            with open(os.path.join(traj_folder_i,'scans',f"{i}.json"), 'w') as f:
                json.dump(bag_scan_data[i], f)
         
        if config[args.dataset_name]["process_pcl"]:
            np.savez(os.path.join(traj_folder_i,'pcd',f"{i}.npz"), bag_pcl_data[i])
        
        n += 1
        
    with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
        trajs["yaw"] = np.stack(trajs["yaw"])
        trajs["position"] = np.stack(trajs["position"])
        trajs["linear_vel"] = np.stack(trajs["linear_vel"])
        trajs["angular_vel"] = np.stack(trajs["angular_vel"])
        pickle.dump(trajs, f)
    
    if config[args.dataset_name]["process_tracking"]:
        with open(os.path.join(traj_folder_i,'tracks.pkl'), 'wb') as f:
            pkl.dump(tracks, f)
    
    with open(os.path.join(traj_folder_i, "timestamps.txt"), "w") as f:
        for ts in timestamps_filtered:
            f.write(f"{ts}\n")
    if bag_speech_data is not None:
        with open(os.path.join(traj_folder_i, "speech.txt"), "w") as f:
            for t,msg in bag_speech_data.items():
                f.write(f"{t}: {msg}\n")
    return True

def main(args: argparse.Namespace):
    # load the config file
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.input_bag:
        bag_files = [args.input_bag]
    elif args.file_list:
        with open(args.file_list, 'r') as f:
            lines = f.readlines()
            bag_files = [os.path.join(args.input_dir, line.strip()) for line in lines if line.strip()]
    else:
        # collect bag files
        bag_files = []
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".db3"):
                    bag_files.append(root)
                    break
            
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]
    # multiprocessingodel
    
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        partial_func = partial(process_bag_file, config=config)
        results = list(
            tqdm(
                pool.imap_unordered(partial_func, bag_files),
                total=len(bag_files),
                desc="Bags processed",
            )
        )
    # results = []
    # for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
    #     p = process_bag_file(bag_path, config)
    #     results.append(p)
        
    for bag_path, success in zip(bag_files, results):
        if success:
            print(f"Processed {bag_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="ubonn",
        required=False,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="/ros2_bags",
        type=str,
        help="path of the datasets with rosbags",
        required=False,
    )
    parser.add_argument(
        "--input-bag",
        "-b",
        default=None,
        type=str,
        help="path of the bag file to process",
        required=False, 
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/processed",
        type=str,
        help="path for processed dataset (default: ../datasets/)",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default="process_bags_config.yaml",
        type=str,
        help="path to the config file (default: process_bags_config.yaml)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        default=4,
        type=int,
        help="number of processes to use (default: 4)",
    )
    parser.add_argument(
        "--contin",
        "-cont",
        action="store_true",
        help="do not overwrite processed bags",
    )
    parser.add_argument(
        "--file-list",
    )
    args = parser.parse_args()
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
