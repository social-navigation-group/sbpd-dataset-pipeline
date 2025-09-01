# iterate through each trajectory in the dataset, convert point cloud to occupancy grid and calculate how much of the grid is occupied. average that across the dataset.
import os
import pickle
import numpy as np
import argparse
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from IPython import embed
from shapely.geometry import box
from shapely.ops import unary_union
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_dilation
SAMPLE_RATE = 4 #rate at which the bag was processed
OCC_GRID_PARAMS= {
            "width": 20, #40m x 40m grid around the robot 
            "height": 20,
            "resolution": 0.05, # 1 unit = 5 cm
            "occupied_cell_value": 1.0,
            "unoccupied_cell_value": 0.0,
            "inflation_radius": 0.0,
            "inflation_scale_factor": 0.0,
        }

DATASET_CONFIG={
    "go2nus": {
        "robot_width": 0.3,
        "robot_height": 0.7,
        "max_height": 1.0,
        "min_height": -0.1,
        "lidar_range_min":0.3,
        "lidar_range_max":30.0,    
        },
    "scand_jackal":{
        "robot_width": 0.4,
        "robot_height": 0.43,
        "max_height": 1.0,
        "min_height": -0.3,
        "lidar_range_min":0.4,
        "lidar_range_max":30.0,    
    },
    "scand_spot":{
        "robot_width": 0.5,
        "robot_height": 1.1,
        "max_height": 1.0,
        "min_height": -0.6,
        "lidar_range_min":0.5,
        "lidar_range_max":30.0,    
    },
    "musohu":{
        "robot_width": 0.1,
        "robot_height": 0.1,
        "max_height": 0.5,
        "min_height": -0.1,
        "lidar_range_min":0.1,
        "lidar_range_max":30.0,    
    }
}

def pcl_to_grid(pcl,params):
    # Convert to grid indices
    width = params['width']
    height = params['height']
    resolution = params['resolution']
    occupied_cell_value = params['occupied_cell_value']
    unoccupied_cell_value = params['unoccupied_cell_value']
    inflation_radius = params['inflation_radius']
    inflation_scale_factor = params['inflation_scale_factor']
    grid_cell_width = int(width / resolution)
    grid_cell_height = int(height / resolution)
    # Create empty grid initialized to unoccupied
    grid = np.full((grid_cell_width,grid_cell_height), unoccupied_cell_value, dtype=np.float32)
    
    x_grid = (((pcl[:,0]  + (width / 2)) / resolution)).astype(int)
    y_grid = (((pcl[:,1]  + (height / 2)) / resolution) + (height/ 2)).astype(int) 

    # Filter valid indices within bounds
    valid_idx = (0 <= x_grid) & (x_grid < grid_cell_width) & (0 <= y_grid) & (y_grid < grid_cell_height)
    x_grid, y_grid = x_grid[valid_idx], y_grid[valid_idx]

    # Set occupied cells
    grid[x_grid,y_grid] = occupied_cell_value 
    # Inflation step using binary dilation
    grid_radius = int(inflation_radius / resolution)
    inflated_mask = None
    if grid_radius > 0:
        structuring_element = np.ones((2 * grid_radius + 1, 2 * grid_radius + 1))  # Circular kernel
        inflated_mask = binary_dilation(grid == occupied_cell_value, structure=structuring_element)
        grid[inflated_mask & (grid!=occupied_cell_value)] = inflation_scale_factor*occupied_cell_value
    return grid

def free_space_analysis(dataset_path,dataset_name):
    """
    Iterates through folders in the dataset path, processes 'tracks.pkl' and 'pedestrian_3d.pkl' files,
    and calculates statistics of the number of people across the dataset.
    """
    cfg = DATASET_CONFIG[dataset_name]
    occupancy_level = {}
    for trajectory in tqdm(os.listdir(dataset_path)): 
        #get the image size 
        occupancy_level[trajectory] = []
        occ_grids = []
        try:
            pcds = os.listdir(os.path.join(dataset_path,trajectory,'pcd'))
        except FileNotFoundError:
            print(f"Error: No point clouds found in {os.path.join(dataset_path,trajectory,'pcd')}")
            continue
        filnames_int = sorted([int(os.path.splitext(pcd)[0]) for pcd in pcds])
        assert len(pcds)>0, f"Error: No point clouds found in {os.path.join(dataset_path,trajectory,'pcd')}"
        for index in filnames_int:
            points = np.load(os.path.join(dataset_path,trajectory,'pcd',f'{index}.npz'))['arr_0'].reshape(-1,3)
            points = points[(np.linalg.norm(points,axis=1) < cfg['lidar_range_max']) & (np.linalg.norm(points,axis=1) >cfg['lidar_range_min'])]
            filtered_points = points[(points[:,-1] > cfg['min_height']) & (points[:,-1] < cfg['max_height'])]            
            occ_grids.append(pcl_to_grid(filtered_points,OCC_GRID_PARAMS))
        occ_grids = np.stack(occ_grids, axis=0)
        np.savez(os.path.join(dataset_path, trajectory, 'occ_grids.npz'), occ_grids)
        #occupancy_level[trajectory] = (occ_grids.sum(axis=(1, 2)) / np.product(occ_grids.shape)).mean()
        #break
    return {
        "occupancy_level": occupancy_level
    }
    
if __name__ == "__main__":
    datasets = {
        #"scand_spot": "/media/shashank/T/processed/scand_spot",
        "scand_jackal": "/media/shashank/T/processed/scand_jackal",
        #"go2nus": "/media/shashank/T/processed/go2nus",
        #"musohu": "/media/shashank/T/processed/musohu",
    }
    proportion_free_space = {}
    for dataset_name, dataset_path in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        stats = free_space_analysis(dataset_path, dataset_name)
        # with open(f'free_space_stats_{dataset_name}.pkl', 'wb') as f:
        #     pickle.dump(stats, f)
        #embed()