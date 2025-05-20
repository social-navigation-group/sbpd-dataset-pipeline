import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from bc_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class BCDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        end_slack: int = 0,
        normalize: bool = True,
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            normalize (bool): Whether to normalize the distances or actions
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        self.end_slack = end_slack
        self.normalize = normalize
        
        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self.pedestrian_cache = {}
        self._load_index()
        #self._build_caches()
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time,"imgs")
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_context_n{self.context_size}_wp{self.waypoint_spacing}_{self.len_traj_pred}_{self.max_dist_cat}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time,"imgs")

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data
    
    def _get_pedestrians(self,trajectory_name):
        if trajectory_name in self.pedestrian_cache:
            return self.pedestrian_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "pedestrians_3d.pkl"), "rb") as f:
                pedestrian_data = pickle.load(f)
            self.pedestrian_cache[trajectory_name] = pedestrian_data
            return pedestrian_data
    
    def __len__(self) -> int:
        return len(self.index_to_data)

    '''
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
    '''
    
    def load_pedestrian_map(self,f_curr, curr_time):
        pedestrians_3d = self._get_pedestrians(f_curr) #dict with time:[(id,position),]
        pedestrians_3d = {t: {id:p for (p,id) in pedestrians_3d[t]} for t in pedestrians_3d}
        traj_data = self._get_trajectory(f_curr)
        start_index = curr_time-self.waypoint_spacing
        end_index = curr_time
        pedmap_size = self.data_config['pedmap']['size']
        pedmap_resolution = self.data_config['pedmap']['resolution']
        L = int(pedmap_size/pedmap_resolution)
        robot_positions = traj_data["position"][start_index:end_index:self.waypoint_spacing] 
        pedestrian_position_map = torch.zeros((L,L,1), dtype=torch.int32)
        pedestrian_velocity_map = torch.zeros((L,L,2),dtype=torch.float32)
        # Assuming x-axis is pointing north, y-axis is pointing left, and the robot is at the bottom center of the map
        # Convert the pedestrian positions into a pedestrian map in the robot's local frame
        # where a cell in the frame has a value of 1 if a pedestrian is in that cell

        robot_in_map_x = L
        robot_in_map_y = int(L/2)  # Robot is at the bottom center of the map
        prev_time = curr_time - self.waypoint_spacing    
        #assuming top left is the origin and robot is at the bottom center of the map
        if curr_time not in pedestrians_3d:
            return pedestrian_position_map, pedestrian_velocity_map
        
        for ped_id, position in pedestrians_3d[curr_time].items():
            # Convert pedestrian position to local frame            
            # Convert local coordinates to map indices
            map_x = robot_in_map_x - int((position[0][0]) / pedmap_resolution) #map x is flipped wrt robot x
            map_y = robot_in_map_y - int(( position[0][1]) / pedmap_resolution)
            # Ensure indices are within bounds
            if 0 <= map_x < pedestrian_position_map.shape[1] and 0 <= map_y < pedestrian_position_map.shape[0]:
                pedestrian_position_map[map_x, map_y] = 1
            # Calculate velocity
            if prev_time in pedestrians_3d:
                if ped_id in pedestrians_3d[prev_time]:
                    prev_position = dict(pedestrians_3d[prev_time])[ped_id]
                    robot_displacement = robot_positions[-1] - robot_positions[0]
                    pedestrian_displacement = (np.array(position[0][:2]) - np.array(prev_position[0][:2])) + robot_displacement
                    # Assign velocity to the map
                    if map_x>=0 and map_x<pedestrian_velocity_map.shape[1] and map_y>=0 and map_y<pedestrian_velocity_map.shape[0]:
                        pedestrian_velocity_map[map_x, map_y] = torch.tensor([pedestrian_displacement[0], pedestrian_displacement[1]])*self.data_config['sample_rate']

        return pedestrian_position_map, pedestrian_velocity_map

    def process_scan(self,scan_data:Tuple):
        """
        Process the scan data by min and avg pooling and stacking
        Args:
            scan_data (Tuple): Tuple of scan data
        Returns:
            processed_scan (torch.Tensor): tensor of shape [1, L, L] containing the processed scan data
        """
        num_bins = int(self.data_config['pedmap']['size']/self.data_config['pedmap']['resolution'])
        pooled_scans = []
        for scan in scan_data:
            # Perform average pooling
            scan[np.isinf(scan)] = 100.0 #default value for no scan return
            avg_pooled = np.mean(scan.reshape(-1, len(scan) // num_bins), axis=1)
            # Perform min pooling
            min_pooled = np.min(scan.reshape(-1, len(scan) // num_bins), axis=1)
            # Stack the results to get a 2xK tensor
            pooled_scan = np.stack([avg_pooled, min_pooled], axis=0)
            pooled_scans.append(pooled_scan)
        
        # Stack all scans to get a tensor of shape 2NxK
        processed_scan = np.concatenate(pooled_scans, axis=0)
        return np.repeat(processed_scan,int(num_bins/(processed_scan.shape[0])), axis=0)[:,:,None]
        
    def __getitem__(self,i:int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing:
                pedestrian position map (torch.Tensor): tensor of shape [2, L, L] containing the x and y position map of the pedestrians in the scene.
                laser_scan (torch.Tensor): tensor of shape [1,L,L] containing min and avg pooled laser scan history of robot
                goal_coordinates (torch.Tensor): tensor of shape (2, ) containing the goal coordinates in the local frame of the robot
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (traj_pred_len, 2) or (traj_pred_len, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        curr_traj_data = self._get_trajectory(f_curr)
        
        #print(f"Trajectory: {f_curr}, Current Time: {curr_time}, Max Goal Distance: {max_goal_dist}")
        
        goal_time = None
        #find the goal time where the robot is Km away from the current position
        for t in range(curr_time+1,len(self._get_trajectory(f_curr)["position"])):
            pos = curr_traj_data["position"][goal_time]
            yaw = curr_traj_data["yaw"][goal_time]
            dist = np.linalg.norm(pos - curr_traj_data["position"][curr_time])
            if dist>=self.data_config['goal_distance']:
                goal_time = t
                break
        if goal_time is None:
            goal_time = curr_time + int(max_goal_dist * self.waypoint_spacing)
    
        context = []
        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        
        context = [(f_curr, t) for t in context_times]
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_curr)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} and {goal_traj_len}"
        
        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        actions = actions[:self.len_traj_pred*self.waypoint_spacing]
        distance = (goal_time - curr_time) // self.waypoint_spacing
        
        # assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
    
        pedestrian_position_map, pedestrian_velocity_map = self.load_pedestrian_map(f_curr,curr_time)
        with open(os.path.join(self.data_folder, f_curr,'scan.pkl'), 'rb') as f:
            all_scans = pickle.load(f)
        laser_scans = [all_scans[t] for f,t in context]
        processed_scan = self.process_scan(laser_scans)
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance)
        )
        return  (
            #torch.as_tensor(np.concatenate([pedestrian_position_map,pedestrian_velocity_map],axis = 2), dtype=torch.float32),
            torch.as_tensor(pedestrian_velocity_map, dtype=torch.float32),
            torch.as_tensor(processed_scan, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
    
    def getitem(self,f_curr,curr_time,max_goal_dist) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing:
                pedestrian position map (torch.Tensor): tensor of shape [2, L, L] containing the x and y position map of the pedestrians in the scene.
                laser_scan (torch.Tensor): tensor of shape [1,L,L] containing min and avg pooled laser scan history of robot
                goal_coordinates (torch.Tensor): tensor of shape (2, ) containing the goal coordinates in the local frame of the robot
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (traj_pred_len, 2) or (traj_pred_len, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        curr_traj_data = self._get_trajectory(f_curr)
        
        print(f"Trajectory: {f_curr}, Current Time: {curr_time}, Max Goal Distance: {max_goal_dist}")
        
        goal_time = None
        #find the goal time where the robot is Km away from the current position
        for t in range(curr_time+1,len(self._get_trajectory(f_curr)["position"])):
            pos = curr_traj_data["position"][goal_time]
            dist = np.linalg.norm(pos - curr_traj_data["position"][curr_time])
            if dist>=self.data_config['goal_distance']:
                goal_time = t
                break
        if goal_time is None:
            goal_time = curr_time + int(max_goal_dist * self.waypoint_spacing)
    
        context = []
        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        
        context = [(f_curr, t) for t in context_times]
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_curr)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} and {goal_traj_len}"
        
        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        actions = actions[:min(len(actions),self.len_traj_pred*self.waypoint_spacing)]
        distance = (goal_time - curr_time) // self.waypoint_spacing
        
        # assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
    
        pedestrian_position_map, pedestrian_velocity_map = self.load_pedestrian_map(f_curr,curr_time)
        with open(os.path.join(self.data_folder, f_curr,'scan.pkl'), 'rb') as f:
            all_scans = pickle.load(f)
        laser_scans = [all_scans[t] for f,t in context]
        processed_scan = self.process_scan(laser_scans)
        
        return  (
            #torch.as_tensor(np.concatenate([pedestrian_position_map,pedestrian_velocity_map],axis = 2), dtype=torch.float32),
            torch.as_tensor(pedestrian_velocity_map, dtype=torch.float32),
            torch.as_tensor(processed_scan, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
        )
    
    