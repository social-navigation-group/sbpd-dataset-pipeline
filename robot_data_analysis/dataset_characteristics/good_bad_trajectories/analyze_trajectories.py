"""
This script analyzes robot trajectories from a dataset and evaluates their quality based on a threshold distance 
between "good" and "bad" waypoints. It also provides visualization of the trajectories and occupancy grids for 
further inspection.
Run the script with the required dataset and configuration file as arguments. For example:
    python analyze_trajectories.py --dataset scand_jackal --config config.yaml
"""
from tqdm import tqdm
from robot_data_analysis.good_bad_trajectories.trajectory_utils import *
import numpy as np
import os
import pickle as pkl
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from PIL import Image
import json,yaml
from scipy.interpolate import CubicSpline

def analyze_trajectories(trajectories,data_config,threshold=0.2,view=False):
    '''
    Analyze trajectories from actions pickle file
    '''
    above_thresh = []
    unsuccessful = []
    below_thresh = []
    outside_cmap = []
    occ_grid_params = data_config['local_planner_params']['occ_grid_params']
    len_traj_eval = data_config['len_traj_eval']
    waypoint_spacing = data_config['waypoint_spacing']
    
    for f_curr,timesteps in tqdm(trajectories.items()):        
        for curr_time,actions in timesteps.items():
            actions_good = actions['good']
            #check if the first item in actions_good is all zeros
            if not np.all(actions_good[0] == 0):
                actions_good = np.vstack((np.array([0.0, 0.0, 0.0]), actions_good))
            actions_bad = actions['bad']
            # If actions_good has fewer waypoints than actions_bad, fit a cubic spline and resample
            if np.where((actions_good[:,0]>occ_grid_params['width']/2.0) | (actions_good[:,1]>occ_grid_params['height']/2.0))[0].shape[0] > 0:
                outside_cmap.append((f_curr,curr_time))
                continue
            if actions_bad is None or len(actions_bad)<2:
                unsuccessful.append((f_curr,curr_time))
                continue
            if len(actions_good) < len(actions_bad):
                cs_x = CubicSpline(np.arange(len(actions_good)), actions_good[:, 0])
                cs_y = CubicSpline(np.arange(len(actions_good)), actions_good[:, 1])
                new_indices = np.linspace(0, len(actions_good) - 1, len(actions_bad))
                x_new = cs_x(new_indices)
                y_new = cs_y(new_indices)
                actions_good = np.vstack((x_new, y_new)).T
            elif len(actions_good) > len(actions_bad):
                cs_x = CubicSpline(np.arange(len(actions_bad)), actions_bad[:, 0])
                cs_y = CubicSpline(np.arange(len(actions_bad)), actions_bad[:, 1])
                new_indices = np.linspace(0, len(actions_bad) - 1, len(actions_good))
                x_new = cs_x(new_indices)
                y_new = cs_y(new_indices)
                actions_bad = np.vstack((x_new, y_new)).T
                
            #calculate the mean euclidean distance between the good and bad waypoints
            dist = ((actions_good[:,:2] - actions_bad[:,:2])**2).mean() 
            if dist > threshold:
                above_thresh.append(dist) 
                if view:    
                    plot_timesteps = len_traj_eval               
                    #plot and save
                    grids = []
                    scan_folder = os.path.join(data_config['data_folder'],f_curr,"scans")
                    pcl_folder = os.path.join(data_config['data_folder'],f_curr,"pcl")
                    img_folder = os.path.join(data_config['data_folder'],f_curr,"imgs")                        
                    end_time = curr_time + len_traj_eval*waypoint_spacing
                    for i in range(plot_timesteps):
                        if os.path.exists(scan_folder):
                            with open(os.path.join(scan_folder,f'{curr_time}.json'),"rb") as f:
                                scan = json.load(f)                        
                            occ_grid = laser_to_grid(scan,occ_grid_params)
                        elif os.path.exists(pcl_folder):
                            z_height = data_config.get('z_height',-np.inf)
                            with open(os.path.join(pcl_folder,f'{curr_time}.npz'),"rb") as f:
                                pcl = np.load(f)['arr_0']
                            pcl = pcl[pcl[:,2]>z_height]
                            occ_grid = pcl_to_grid(pcl,occ_grid_params)
                        else:
                            raise FileNotFoundError(f"No scan or pcl folder found for {f_curr}")
                        grids.append(occ_grid)
                    rgbs = []
                    for i in range(plot_timesteps):
                        rgbs.append(np.asarray(Image.open(os.path.join(img_folder,f"""{curr_time+i*waypoint_spacing}.jpg"""))))

                    fig, ax = plt.subplots(2, plot_timesteps, figsize=(2*plot_timesteps, 5))
                    if plot_timesteps == 1:
                        ax = np.array([[ax[0]], [ax[1]]])  # Ensure ax is 2D for consistency
                    for i in range(plot_timesteps):
                        extents = np.array([-grids[i].shape[1]/2, grids[i].shape[1]/2, -grids[i].shape[0]/2, grids[i].shape[0]/2]) * occ_grid_params['resolution']
                        ax[0, i].imshow(grids[i],origin='lower',extent=extents,cmap='binary')
                        ax[0, i].set_title(f'Occupancy Grid {i+1}')
                        ax[0, i].grid(False)
                        ax[1, i].imshow(rgbs[i])
                        ax[1, i].set_title(f'RGB {i+1}')
                        ax[1, i].grid(False)

                    # center = np.array((grids[0].shape[1] // 2, grids[0].shape[0] // 2))
                    #occ_a0 = actions_good[:, :2] / occ_grid_params['resolution'] + center
                    #occ_a1 = actions_bad[:, :2] / occ_grid_params['resolution'] + center
                    indices = np.linspace(0, len(actions_good) - 1, plot_timesteps, dtype=int)
                    for i in range(plot_timesteps):    
                        temp_actions_good = actions_good[indices[i]:] - actions_good[indices[i]]
                        temp_actions_bad = actions_bad[indices[i]:] - actions_bad[indices[i]]
                        ax[0, i].plot(temp_actions_good[:, 0], temp_actions_good[:, 1], color='green', linewidth=1, label="good")
                        ax[0, i].plot(temp_actions_bad[:, 0], temp_actions_bad[:, 1], color='red',linewidth=1, label="bad")
                        #ax[0, i].scatter(occ_a0[0, 0], occ_a0[0, 1], c='r', s=5)
                        #ax[0, i].scatter(occ_a1[0, 0], occ_a1[0, 1], c='r', s=5)

                    fig.suptitle(f"Distance: {dist:.2f}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(data_config['data_folder'],'train','trajectory_comparisons_'+data_config['local_planner_params']['name'], f'{f_curr}_{curr_time}_{end_time}.jpg'),dpi=300)
                    plt.close()
            else:
                below_thresh.append((f_curr,curr_time))
        
    print(f'Filtered out {len(below_thresh)} samples out of {len(below_thresh)+len(above_thresh)+len(unsuccessful)}({100.0*len(below_thresh)/(len(below_thresh)+len(above_thresh)+len(unsuccessful))}%)')


def main(args):
    dataset = args.dataset
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    data_config = cfg['datasets'][dataset]
    data_config['len_traj_eval'] = cfg['len_traj_eval']
    data_config['context_size'] = cfg['context_size']
    data_config['local_planner_params'] = cfg['local_planner_params']
    action_file = os.path.join(data_config['train'],'actions.pkl')
    with open(action_file, 'rb') as f:
        trajectories = pkl.load(f)
    analyze_trajectories(trajectories,data_config,threshold=0.2,view=True)
                
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Bad traj generator")
    parser.add_argument("--dataset","-d",default='scand_jackal')
    parser.add_argument("--config","-c", type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)