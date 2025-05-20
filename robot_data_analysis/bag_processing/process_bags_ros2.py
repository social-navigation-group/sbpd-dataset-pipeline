#https://github.com/robodhruv/visualnav-transformer

import os
import pickle
import argparse
import tqdm
import yaml
import json
import multiprocessing
from functools import partial
from process_data_utils import get_images_pcl_and_odom
import numpy as np
from pprint import pprint
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from ros2_utils import get_synced_raw_messages_from_bag
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, CompressedImage
import pickle as pkl
from IPython import embed

def process_bag_file(bag_path,config):
    traj_name = bag_path.split("/")[-1].replace('.bag','')
    traj_name_i = traj_name + f"_{0}"
    traj_folder_i = os.path.join(args.output_dir, traj_name_i)
    if os.path.exists(traj_folder_i) and args.contin:
        print(f"{bag_path} already processed. Skipping...")
        return True 
    type_map = {
        '/utlidar/robot_odom': Odometry,
        '/camera/camera/color/image_raw/compressed': CompressedImage,
        '/rslidar_points': PointCloud2,
        '/tracks_camera_camera_color_image_raw_compressed': Detection2DArray
    }
    #try:
    print(f"Processing {bag_path}...")
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('cdr','cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    storage_filter = StorageFilter(topics = list(type_map.keys()))
    reader.set_filter(storage_filter)
    b = (reader,type_map)
    #except Exception as e:
    #    print(e)
    #    print(f"Error loading {bag_path}. Skipping...")
    #   return False

    # load the hdf5 file
    synced_imdata, synced_odomdata, synced_pcldata, synced_scandata, synced_trackingdata = get_synced_raw_messages_from_bag(
        b = b,
        imtopics = config[args.dataset_name]["imtopics"],
        pcltopics = config[args.dataset_name]["pcltopics"] if config[args.dataset_name]["process_pcl"] else [],
        odomtopics = config[args.dataset_name]["odomtopics"],
        scantopics = config[args.dataset_name]["scantopics"] if config[args.dataset_name]["process_scan"] else [], 
        trackingtopics = config[args.dataset_name]["trackingtopics"] if config[args.dataset_name]["process_tracking"] else [],
    )
    
    print(f"Found {len(synced_imdata)} images,\n {len(synced_odomdata)} odom,\n {len(synced_pcldata)} pcl,\n {len(synced_scandata)} scans,\n {len(synced_trackingdata)} tracks in {bag_path}")
    
    bag_img_data, bag_traj_data, bag_pcl_data, bag_scan_data, bag_track_data = get_images_pcl_and_odom(
        synced_imdata,
        synced_odomdata,
        synced_pcldata if config[args.dataset_name]["process_pcl"] else [],
        synced_scandata if config[args.dataset_name]["process_scan"] else [],
        synced_trackingdata if config[args.dataset_name]["process_tracking"] else [],
        config[args.dataset_name]["img_process_func"],
        config[args.dataset_name]["pcl_process_func"],
        config[args.dataset_name]["odom_process_func"],
        config[args.dataset_name]["scan_process_func"], 
        config[args.dataset_name]["tracking_process_func"],
        ang_offset=config[args.dataset_name]["ang_offset"],
        rosversion = 2,
    )
    
    
    if bag_img_data is None or bag_traj_data is None or (bag_pcl_data is None and config[args.dataset_name]["process_pcl"]) or (bag_scan_data is None and config[args.dataset_name]["process_scan"]) or (bag_track_data is None and config[args.dataset_name]["process_tracking"]):
        print(
        f"{bag_path} did not have the topics we were looking for. Skipping..."
        )
        return False
    
    traj_pos = bag_traj_data["position"]
    traj_yaws = bag_traj_data["yaw"]
    traj_linear_vels = bag_traj_data["linear_vel"]
    traj_angular_vels = bag_traj_data["angular_vel"]
    trajs = {"position": [], "yaw": [], "linear_vel": [], "angular_vel": []}
    if not os.path.exists(traj_folder_i):
        os.makedirs(traj_folder_i)
        os.makedirs(os.path.join(traj_folder_i, "imgs"))
        if bag_pcl_data:
            os.makedirs(os.path.join(traj_folder_i, "pcd"))
        if bag_scan_data:
            os.makedirs(os.path.join(traj_folder_i, "scans"))  # New directory for scans

    tracks=[]
    for i in range(max(config[args.dataset_name]['start_slack'], 0), len(traj_pos) - config[args.dataset_name]['end_slack']):
        bag_img_data[i-1].save(os.path.join(traj_folder_i,'imgs', f"{i-1}.jpg"))
        trajs["position"].append(traj_pos[i - 1])
        trajs["yaw"].append(np.array(traj_yaws[i - 1]))
        trajs["linear_vel"].append(np.array(traj_linear_vels[i - 1]))
        trajs["angular_vel"].append(np.array(traj_angular_vels[i - 1]))
        
        if config[args.dataset_name]["process_tracking"] and bag_track_data:
            try:
                if bag_track_data[i - 1] is None:
                    tracks.append((None,None))
                else:
                    tracks.append((bag_track_data[i-1][0], bag_track_data[i-1][1]))
            except Exception as e:
                print(e)
                print(f"Error processing tracks for {bag_path}. Skipping...")
                embed()
                continue

        
        if config[args.dataset_name]["process_scan"]:
            with open(os.path.join(traj_folder_i,'scans',f"{i-1}.json"), 'w') as f:
                json.dump(bag_scan_data[i - 1], f)
         
        if config[args.dataset_name]["process_pcl"]:
            np.savez(os.path.join(traj_folder_i,'pcd',f"{i-1}.npz"), bag_pcl_data[i - 1])
            
    with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
        trajs["yaw"] = np.stack(trajs["yaw"])
        trajs["position"] = np.stack(trajs["position"])
        trajs["linear_vel"] = np.stack(trajs["linear_vel"])
        trajs["angular_vel"] = np.stack(trajs["angular_vel"])
        pickle.dump(trajs, f)
    
    if config[args.dataset_name]["process_tracking"]:
        with open(os.path.join(traj_folder_i,'tracks.pkl'), 'wb') as f:
            pkl.dump(tracks, f)
        
    return True

def main(args: argparse.Namespace):
    # load the config file
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
            tqdm.tqdm(
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
        default="scand",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default=None,
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
        default="../datasets/",
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
    args = parser.parse_args()
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
