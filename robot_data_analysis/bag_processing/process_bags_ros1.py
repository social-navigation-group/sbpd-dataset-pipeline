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
import rosbag
from ros1_utils import get_synced_raw_messages_from_bag
def process_bag_file(bag_path,config):
    traj_name = bag_path.split("/")[-1].replace('.bag','')
    traj_name_i = traj_name + f"_{0}"
    traj_folder_i = os.path.join(args.output_dir, traj_name_i)
    if os.path.exists(traj_folder_i) and args.contin:
        print(f"{bag_path} already processed. Skipping...")
        
    try:
        b = rosbag.Bag(bag_path)
    except rosbag.ROSBagException as e:
        print(e)
        print(f"Error loading {bag_path}. Skipping...")
        return False

    # load the hdf5 file
    synced_imdata, synced_odomdata, synced_pcldata, synced_scandata = get_synced_raw_messages_from_bag(
        b,
        config[args.dataset_name]["imtopics"],
        config[args.dataset_name]["pcltopics"] if config[args.dataset_name]["process_pcl"] else 
        [],
        config[args.dataset_name]["odomtopics"],
        config[args.dataset_name]["scantopics"] if config[args.dataset_name]["process_scan"] else [], 
    )
    
    bag_img_data, bag_traj_data, bag_pcl_data, bag_scan_data = get_images_pcl_and_odom(
        synced_imdata,
        synced_odomdata,
        synced_pcldata if config[args.dataset_name]["process_pcl"] else [],
        synced_scandata if config[args.dataset_name]["process_scan"] else [], 
        config[args.dataset_name]["img_process_func"],
        config[args.dataset_name]["pcl_process_func"],
        config[args.dataset_name]["odom_process_func"],
        config[args.dataset_name]["scan_process_func"], 
        ang_offset=config[args.dataset_name]["ang_offset"],
        rosversion = 1,
    )
    
    
    if bag_img_data is None or bag_traj_data is None or (bag_pcl_data is None and config[args.dataset_name]["process_pcl"]) or (bag_scan_data is None and config[args.dataset_name]["process_scan"]):
        print(
        f"{bag_path} did not have the topics we were looking for. Skipping..."
        )
        return False
    
    traj_pos = bag_traj_data["position"]
    traj_yaws = bag_traj_data["yaw"]
    trajs = {"position": [], "yaw": []}
    if not os.path.exists(traj_folder_i):
        os.makedirs(traj_folder_i)
        os.makedirs(os.path.join(traj_folder_i, "imgs"))
        if bag_pcl_data:
            os.makedirs(os.path.join(traj_folder_i, "pcd"))
        if bag_scan_data:
            os.makedirs(os.path.join(traj_folder_i, "scans"))  # New directory for scans

        
    for i in range(max(config[args.dataset_name]['start_slack'], 0), len(traj_pos) - config[args.dataset_name]['end_slack']):
        bag_img_data[i-1].save(os.path.join(traj_folder_i,'imgs', f"{i}.jpg"))
        trajs["position"].append(traj_pos[i - 1])
        trajs["yaw"].append(np.array(traj_yaws[i - 1]))
        
        if config[args.dataset_name]["process_scan"]:
            with open(os.path.join(traj_folder_i,'scans',f"{i}.json"), 'w') as f:
                json.dump(bag_scan_data[i - 1], f)    
        
        if config[args.dataset_name]["process_pcl"]:
            np.savez(os.path.join(traj_folder_i,'pcd',f"{i}.npz"), bag_pcl_data[i - 1])
            
    with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
        trajs["yaw"] = np.stack(trajs["yaw"])
        trajs["position"] = np.stack(trajs["position"])
        pickle.dump(trajs, f)
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
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
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
        default="",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
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
        default=1,
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
