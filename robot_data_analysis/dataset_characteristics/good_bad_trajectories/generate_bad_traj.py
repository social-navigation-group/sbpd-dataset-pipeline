"""
Generates ground_truth ("good") and geometrical planner ("bad") trajectories for a given dataset by processing trajectory data.
Args:
    --dataset (-d): Name of the dataset to process (default: 'scand_jackal')
    --num_files (-n): Number of files to process (default: all files)
    --num_workers (-w): Number of parallel workers (default: 1)
    --config (-c): Path to the configuration file (default: 'config.yaml')
    --multiproc (-mp): Whether to use multiprocessing (default: False) -> Only used for 'save' mode
"""
from tqdm import tqdm
from trajectory_utils import *
import numpy as np
import os
import pickle as pkl
import argparse
import multiprocessing
import json,yaml
from IPython import embed  
def trajectory_processor(traj_name,data_config,skip_bbox_area,yolo_results):
    #prepare inputs and process trajectory
    with open(os.path.join(data_config['data_folder'],traj_name,"traj_data.pkl"),"rb") as f:
        traj_data = pkl.load(f) 
    
    traj_len = len(traj_data["position"])
    scan_folder = os.path.join(data_config['data_folder'],traj_name,"scans")
    pcl_folder = os.path.join(data_config['data_folder'],traj_name,"pcd")      
    
    use_scan, use_pcl = False,False
    if os.path.exists(scan_folder):
        use_scan = True
    elif os.path.exists(pcl_folder):
        use_pcl = True
    else:
        print(f"Neither scan nor pcl folder exists for {traj_name}")
        return None
    
    occ_grids = []  
    occ_grid_params = data_config['local_planner_params']['occ_grid_params']
    for i in range(1,traj_len):
        if use_scan: #get occ_grid from laser_scan
            try:
                with open(os.path.join(scan_folder,f"{i}.json"),"rb") as f:
                    scan = json.load(f)            
            except FileNotFoundError:
                print(f"Scan file {i}.json not found in {scan_folder}")
                return None
            occ_grid = laser_to_grid(scan,occ_grid_params)
        
        elif use_pcl:#get occ_grid by slicing pointcloud
            try:
                with open(os.path.join(pcl_folder,f"{i}.npz"),"rb") as f:
                    pcl = np.load(f)['arr_0']
            except FileNotFoundError:
                print(f"Pointcloud file {i}.npz not found in {pcl_folder}")
                return None
            
            z_height = data_config.get('z_height',-np.inf)
            pcl = pcl[pcl[:,2]>z_height]
            #filter out points that are too close to the robot
            #remove points that are too close to the robot
            pcl = pcl[np.linalg.norm(pcl[:,:2],axis=1)>0.6]
            occ_grid = pcl_to_grid(pcl,occ_grid_params)
            #plt.imshow(occ_grid)
            #plt.pause(0.1)
        
        occ_grids.append(occ_grid)
    
    start_slack = data_config['start_slack']
    waypoint_spacing = data_config['waypoint_spacing']
    end_slack = data_config['end_slack']
    begin_time = start_slack + data_config['context_size']*waypoint_spacing
    end_time = traj_len - end_slack-data_config['len_traj_eval']*waypoint_spacing
    return process_trajectory(
        traj_data=traj_data,
        occ_grids=occ_grids,
        waypoint_spacing=data_config['waypoint_spacing'],
        len_traj_eval=data_config['len_traj_eval'],
        planner_config=data_config['local_planner_params'],
        skip_bbox_area=skip_bbox_area,
        yolo_results= yolo_results,
        begin_time=begin_time,
        end_time=end_time
    )
    
def main(args):
    dataset = args.dataset
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    data_config = cfg['datasets'][dataset]
    data_config['len_traj_eval'] = cfg['len_traj_eval']
    data_config['context_size'] = cfg['context_size']
    data_config['local_planner_params'] = cfg['local_planner_params']
    action_file = os.path.join(data_config['data_folder'],'actions.pkl')
    
    if args.trajectory_list is not None:
        with open(args.trajectory_list,'r') as f:
            file_lines = f.read()
            traj_names = file_lines.split("\n")
            if "" in traj_names:
                traj_names.remove("")
    elif args.trajectory_name is not None:
        traj_names = [args.trajectory_name]
    else:
        return None
    
    try:
        with open(os.path.join(data_config['data_folder'],'yolo_human_annotation.pkl'), 'rb') as f:
            yolo_results = pkl.load(f)
        print("Loaded yolo results")
        skip_bbox_area = 2000
    except FileNotFoundError:
        yolo_results = {}
        skip_bbox_area = -1
    
    try:
        with open(action_file, 'rb') as f:
            actions_dict = pkl.load(f)    
    except FileNotFoundError:
        actions_dict = {}
    
    max_files = args.num_files
    if max_files<0:
        max_files = max(1,len(traj_names)-1)
        
    for i,name in enumerate(tqdm(traj_names[:max_files])):
        result = trajectory_processor(name,data_config,skip_bbox_area,yolo_results.get(name,{}))
        if result is None:
            print(f"Error processing {name}")
            continue
        with open(os.path.join(data_config['data_folder'],name,"actions.pkl"),"wb") as f:
             pkl.dump(result,f)
        
        # actions_dict.update({name: result})
        # with open(action_file, 'wb') as f:
        #     pkl.dump(actions_dict, f)
                    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Bad traj generator")
    parser.add_argument("--dataset","-d", type=str, default='go2nus')
    parser.add_argument("--trajectory_list","-t", type=str,default=None)
    parser.add_argument("--trajectory_name","-tn", type=str,default=None)
    parser.add_argument("--num_files","-n", type=int, default=-1)
    parser.add_argument("--num_workers","-w", type=int, default=1)
    parser.add_argument("--config","-c", type=str, default='config.yaml')
    parser.add_argument("--multiproc","-mp", action='store_true')
    args = parser.parse_args()
    main(args)