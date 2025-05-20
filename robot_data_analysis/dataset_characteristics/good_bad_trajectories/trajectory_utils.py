import numpy as np
import torch
from roboticstoolbox import DstarPlanner
from mppi.src.controller.mppi import MPPI
from mppi.src.envs.navigation_2d import Navigation2DEnv
from scipy.ndimage import binary_dilation
from tqdm import tqdm

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

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

def laser_to_grid(scan,params):
    angle_min = scan['angle_min']
    angle_max = scan['angle_max']
    width = params['width']
    height = params['height']
    resolution = params['resolution']
    occupied_cell_value = params['occupied_cell_value']
    unoccupied_cell_value = params['unoccupied_cell_value']
    inflation_radius = params['inflation_radius']
    inflation_scale_factor = params['inflation_scale_factor']
    
    ranges = np.array(scan['ranges'])
    # Compute grid size
    grid_cell_width = int(width / resolution)
    grid_cell_height = int(height / resolution)
    
    # Create empty grid initialized to unoccupied
    grid = np.full((grid_cell_width,grid_cell_height), unoccupied_cell_value, dtype=np.float32)
    
    # Generate angle array
    angles = np.linspace(angle_min, angle_max, len(ranges), endpoint=False)

    # Convert ranges to Cartesian coordinates
    valid_mask = np.isfinite(ranges)  # Mask invalid range readings
    x_dist = ranges[valid_mask] * np.cos(angles[valid_mask]) + (width / 2)
    y_dist = ranges[valid_mask] * np.sin(angles[valid_mask]) + (height / 2)

    # Convert to grid indices
    x_grid = (x_dist / resolution).astype(int)
    y_grid = (y_dist / resolution).astype(int)

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

def compute_actions(traj_data, curr_time, goal_time,len_traj_eval, waypoint_spacing):
    """
    Compute actions based on trajectory data, current time, and goal time.

    Args:
        traj_data (dict): A dictionary containing trajectory data with keys "position" and "yaw".
        curr_time (int): The current time index in the trajectory data.
        goal_time (int): The goal time index in the trajectory data.
        len_traj_eval (int): The length of the trajectory evaluation.
        waypoint_spacing (int): The spacing between waypoints.

    Returns:
        tuple: A tuple containing:
            - actions (np.ndarray): An array of actions with shape (len_traj_eval, 3).
            - goal_pos (np.ndarray): The goal position in local coordinates.
    """
    start_index = curr_time
    #end_index = curr_time + len_traj_eval * waypoint_spacing + 1
    end_index = min(goal_time, len(traj_data["position"]) - 1) + 1
    yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:waypoint_spacing]
    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_eval + 1,):
        const_len = len_traj_eval + 1 - yaw.shape[0]
        yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
        positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

    assert yaw.shape == (len_traj_eval + 1,), f"{yaw.shape} and {(len_traj_eval + 1,)} should be equal"
    assert positions.shape == (len_traj_eval + 1, 2), f"{positions.shape} and {(len_traj_eval + 1, 2)} should be equal"

    waypoints = to_local_coords(positions, positions[0], yaw[0])
    
    assert waypoints.shape == (len_traj_eval + 1, 2), f"{waypoints.shape} and {(len_traj_eval + 1, 2)} should be equal"

    yaw = yaw[1:] - yaw[0]
    actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1) #first waypoint is the current position = (0,0) in local coords

    assert actions.shape == (len_traj_eval, 3), f"{actions.shape} and {(len_traj_eval, 3)} should be equal"

    return actions

def dstar_planner(occ_grid,goal,start):
    """
    Plan a path using the D* algorithm.

    Args:
        occ_grid (np.ndarray): The occupancy grid representing the environment.
        goal (Tuple[int, int]): The goal position in grid coordinates.
        start (Tuple[int, int], optional): The starting position in grid coordinates. Defaults to [0,0].

    Returns:
        np.ndarray or None: The planned trajectory as an array of waypoints in grid coordinates, or None if planning fails.
    """
    ds = DstarPlanner(occ_grid,goal=goal)
    start = (int(start[0]),int(start[1]),start[2])
    try:
        ds.plan()
        traj,status = ds.query(start=tuple(start))
    except IndexError:
        traj = None
    return traj

def mppi_planner(env,mppi_params,render=False):
    """
    Plan a path using the Model Predictive Path Integral (MPPI) algorithm.

    Args:
        env (Navigation2DEnv): The environment instance that simulates the navigation task, providing dynamics and cost evaluation.
        mppi_params (dict): A dictionary containing parameters for the MPPI algorithm, such as horizon, number of samples, control limits and waypoints
    Returns:
        np.ndarray or None: The planned trajectory as an array of waypoints in metric coordinates, or None if planning fails.
    """
    solver = MPPI(
        horizon=mppi_params.get('horizon', 30),
        num_samples=mppi_params.get('num_samples', 3000),
        dim_state=mppi_params.get('dim_state', 3),
        dim_control=mppi_params.get('dim_control', 2),
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.tensor(mppi_params.get('sigmas', [0.5, 0.5])),
        lambda_=mppi_params.get('lambda_', 1.0),
        auto_lambda=mppi_params.get('auto_lambda', False),
    )
    if render:
        state = env.reset(vis=True)
    else:
        state = env.reset(vis=False)
    #state = env.reset()
    max_steps = mppi_params.get('max_steps',500)
    states = []
    wp = 1
    waypoints = mppi_params.get('waypoints', None)
    trajectory = None
    for i in range(max_steps):
        action_seq, state_seq = solver.forward(state=state)
        state, is_goal_reached = env.step(action_seq[0, :])
        states.append(state)
        if render:
            is_collisions = env.collision_check(state=state_seq)
            top_samples, top_weights = solver.get_top_samples(num_samples=300)
            env.render(
                    predicted_trajectory=state_seq,
                    is_collisions=is_collisions,
                    top_samples=(top_samples, top_weights),
                    mode="human",
                    waypoints=waypoints
                )
        
        if is_goal_reached:
            if waypoints is not None and wp<waypoints.shape[0]:
                next_goal = waypoints[wp]
                wp+=1
            else:
                next_goal = None    

            if next_goal is not None:
                env.update_goal(next_goal)
                solver.reset()
                #print(next_goal)
                #print("Waypoint Reached!")
            else:
                #print("Goal Reached!")
                trajectory = torch.vstack(states).cpu().numpy()
                break
    env.close()
    return trajectory   

def occupancy_grid_planner(name,goal,occ_grid,grid_params,planner_params,start=[0,0],render=False):
    if name == 'dstar':       
        center = (np.array(occ_grid.shape)/2).astype(np.int64)
        grid_goal = np.int64(np.array(goal[:2])/grid_params['resolution']) + center
        grid_start = np.array(start) + center
   
        theta = np.arctan2(
                goal[1] - start[1],
                goal[0] - start[0],
            )
        theta =  ((theta + np.pi) % (2 * np.pi)) - np.pi #align initial orientation with goal
        grid_start = np.hstack([grid_start,theta])
        traj = dstar_planner(occ_grid,grid_goal,grid_start)
        traj = (traj - np.array(occ_grid.shape)/2) * grid_params['resolution']
        return traj
    
    elif name == 'mppi':
        env = Navigation2DEnv(
                _map = occ_grid,
                map_size = (grid_params['width'], grid_params['height']),
                cell_size = grid_params['resolution'],
                start = start,
                goal = goal,
                goal_threshold = 0.25
        )
        #if the goal is unreachable, reduce the goal threshold
        _goal = torch.zeros((1,1,2))
        _goal[0,0,:] += goal
        if env.collision_check(state=_goal)>grid_params['unoccupied_cell_value']:
            env.goal_threshold = 0.5
            print("Goal unreachable, increasing goal threshold")
        trajectory = mppi_planner(env,planner_params,render) 
        return trajectory
    return NotImplementedError

def process_trajectory(traj_data, 
                       occ_grids, 
                       waypoint_spacing,
                       len_traj_eval,
                       planner_config,
                       skip_bbox_area = -1,
                       yolo_results={},
                       begin_time = 0,
                       end_time = -1):
    #Processes a single trajectory by computing good and bad actions
    actions_dict = {}   
    for i,curr_time in tqdm(enumerate(range(begin_time,end_time))):  
        
        #DEBUG
        # if curr_time!=44:
        #    continue
          
   
        #constant time analysis
        goal_time_traj_eval = curr_time + len_traj_eval*waypoint_spacing #int(data_config['len_traj_eval']*waypoint_spacing)
        #compute GT actions
        actions_good = compute_actions(traj_data, curr_time, goal_time_traj_eval,len_traj_eval,waypoint_spacing)
        actions_good = np.vstack([np.zeros(3),actions_good])
        
        '''
        #constant distance analysis
        actions_full = compute_actions(traj_data, curr_time, len(traj_data['position']),len(traj_data['position'])-curr_time + 1,waypoint_spacing)
        # Find the first waypoint in actions_full that is 10m away from (0, 0)
        distances = np.linalg.norm(actions_full[:, :2], axis=1)
        goal_index = np.argmax(distances >= 10.0)
        if distances[goal_index] < 10.0:
            continue  # Skip if no waypoint is 10m away
        goal_time_traj_eval = curr_time + goal_index * waypoint_spacing
        actions_good = actions_full[:goal_index + 1]
        '''
        if skip_bbox_area>=0: #skip timestep if too few humans in the scene
            yr = [yolo_results.get(f'{t}.jpg',None) for t in range(curr_time, goal_time_traj_eval)]
            skip_flags = []
            for result in yr:
                if result is None:
                    skip_flags.append(True)
                else:
                    total_area = sum((box.xywh[0][2] * box.xywh[0][3]).cpu().numpy() for box in result if box.conf>0.5)
                    skip_flags.append(total_area < skip_bbox_area)
            if all(skip_flags):
                continue
        
        #compute geometrical planner actions
        actions_bad = None
        waypoints = None
        occ_grid_params = planner_config['occ_grid_params']
        if 'dstar' in planner_config['name']:
            actions_bad = occupancy_grid_planner(name='dstar',
                                                goal = actions_good[-1][:2],
                                                occ_grid=occ_grids[curr_time],
                                                grid_params=occ_grid_params,
                                                planner_params = planner_config)
        if 'mppi' in planner_config['name']:
            pcfg = planner_config.copy()
            if actions_bad is not None and len(actions_bad) > planner_config['num_waypoints']:
                indices = np.linspace(0, len(actions_bad) - 1, planner_config['num_waypoints'], dtype=int)
                waypoints = actions_bad[indices]
            else:
                waypoints = actions_bad
            pcfg['waypoints'] = waypoints
            actions_bad = occupancy_grid_planner(name='mppi',
                                                goal = actions_good[-1][:2] if actions_bad is None else actions_bad[0][:2],
                                                occ_grid=occ_grids[curr_time],
                                                grid_params=occ_grid_params,
                                                planner_params = pcfg,render = False)
        if actions_bad is not None:
            actions_bad = np.array(actions_bad)
        actions_dict[curr_time] = {"good":actions_good,"bad":actions_bad}          
        
    return actions_dict 