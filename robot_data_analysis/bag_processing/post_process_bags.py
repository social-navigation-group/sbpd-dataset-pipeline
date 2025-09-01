import os
from pathlib import Path
import pickle as pkl
import argparse
import yaml
import numpy as np
import open3d as o3d
import cv2
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import re
from collections import defaultdict
from IPython import embed
from scipy.spatial.transform import Rotation 
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from natsort import natsorted
DATASET_CONFIG = {
    "go2nus":{
        'rgb_T_lidar': np.array([
        [0, -1, 0, 0.045],
        [0, 0, -1, -0.03],
        [1, 0, 0, -0.186],
        [0, 0, 0, 1]
        ]),
        'camera_intrinsics': np.array([607.6533813476562, 0.0, 420.5920104980469, 0.0,
                                       0.0, 607.3688354492188, 247.97695922851562, 0.0, 
                                       0.0, 0.0, 1.0, 0.0, 
                                       0.0, 0.0, 0.0, 1.0]).reshape(4,4),
        'occ_grid_params': {
            "width": 40,
            "height": 40,
            "resolution": 0.02 ,
            "occupied_cell_value": 1.0,
            "unoccupied_cell_value": 0.0,
            "inflation_radius": 0.0,
            "inflation_scale_factor": 0.0,
            "robot_width": 0.0,
            "robot_height": 0.0,
            "fpv": False, #only consider area ahead of the robot
        },
        'scan_params':{
            'angle_min': -np.pi/2,
            'angle_max': np.pi/2,
            'angle_increment': 0.25*np.pi/180.0,
            'range_min': 0.1,
            'range_max': 30.0,
            'min_height': -0.1,
            'max_height': 0.5,
            'use_inf': True,
        },
        'lidar_range_max': 10.0,
        'lidar_range_min': 0.3,
        'bbox_ratio_threshold':0.0, 
    },
    "scand_spot":{
        
    }
}
IMG_WIDTH, IMG_HEIGHT = 848, 480
# --- YOLO Model Setup ---
# Load a pre-trained YOLOv8 segmentation model
# 'yolov8n-seg.pt' is small and fast. For higher accuracy, use 'yolov8s-seg.pt' etc.
model = YOLO('yolo11s-seg.pt')
# The 'person' class index in the COCO dataset is 0
PERSON_CLASS_INDEX = 0

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    _, im_w = im.shape[:2]
    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    

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
    cv2.putText(im, f'num:{len(tlwhs)}',
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    return im

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

def project_lidar_to_image(lidar_points, camera_T_lidar, camera_intrinsics, img_height, img_width):
    """
    # By ChatGPT
    Projects 3D LiDAR points onto an RGB image.

    Args:
        lidar_points (np.ndarray): Nx3 array of 3D LiDAR points.
        lidar_to_rgb_transform (np.ndarray): 4x4 matrix transforming LiDAR to camera frame.
        camera_intrinsics (np.ndarray): 4x4 intrinsic camera matrix.

    Returns:
        np.ndarray: Depth image with projected depth values at corresponding pixel locations.
        np.ndarray: 2D pixel coordinates of the valid LiDAR points.
    """
    # Step 1: Convert LiDAR points to homogeneous coordinates
    ones = np.ones((lidar_points.shape[0], 1))
    lidar_points_hom = np.hstack((lidar_points, ones))  # Nx4
    
    # Step 2: Transform to the camera frame
    camera_points_hom = (camera_T_lidar @ lidar_points_hom.T).T  # Nx4
    
    # Step 3: Normalize to 3D (remove homogeneous component)
    camera_points = camera_points_hom[:, :3] / camera_points_hom[:, 3, np.newaxis]  # Nx3
    
    # Step 4: Project to the 2D image plane using intrinsic matrix
    pixels_hom = (camera_intrinsics @ camera_points_hom.T).T  # Nx4
    pixels = pixels_hom[:, :2] / pixels_hom[:, 2, np.newaxis]  # Nx2
    
    # Step 5: Filter points with positive depth and within image bounds
    valid_mask = (
        (camera_points[:, 2] > 0) &  # Positive depth
        (pixels[:, 0] >= 0) & (pixels[:, 0] < img_width) &  # x within img_width
        (pixels[:, 1] >= 0) & (pixels[:, 1] < img_height)  # y within img_height
    )
    
    valid_pixels = pixels[valid_mask].astype(int)
    valid_depths = camera_points[valid_mask, 2]
    
    # Step 6: Create a depth image and populate it
    depth_image = np.zeros((img_height, img_width), dtype=np.float32)
    depth_image[valid_pixels[:, 1], valid_pixels[:, 0]] = valid_depths
    return depth_image, valid_pixels

def assign_depth_points_with_occlusion(valid_pixels, depth_image, tlwh_id,mask = None):
    """
    Given a set of pixels within bounding boxes, a depth image and the bounding boxes,
    assign each pixel to the nearest bounding box based on depth, considering occlusion.
    """
    point_to_bbox = defaultdict(list)
    
    # First pass: record all bboxes each point falls into
    for (tlwh, track_id) in tlwh_id:
        x, y, w, h = map(int, tlwh)
        x2, y2 = x + w, y + h
        for i, (px, py) in enumerate(valid_pixels):  
            if x <= px < x2 and y <= py < y2 and mask[py,px]:
                depth = depth_image[py, px]
                point_to_bbox[(px, py)].append((depth, track_id)) # if a particular pixel falls into multiple bounding boxes, store all candidates with their depth and track_id 

    # Second pass: assign each point to the nearest (depth) bounding box
    bbox_to_points = defaultdict(list)
    for (px, py), candidates in point_to_bbox.items():
        if not candidates:
            continue
        depth, best_track_id = min(candidates)  # select nearest
        bbox_to_points[best_track_id].append((px, py, depth))

    # Format output
    valid_pixels_in_bbox = [
        (np.array(points,dtype=np.int32)[:,:2],int(track_id)) for track_id, points in bbox_to_points.items()
    ]
    return valid_pixels_in_bbox

def track_humans_3d_naive(img,tlwh_id,lidar,camera_intrsincs,lidar_T_rgb,n = "",threshold=0.0):
    global model
    results = model(img, verbose=False)
    pedestrian_masks = []
    if results[0].masks:
        for i, mask in enumerate(results[0].masks):
            if int(results[0].boxes.cls[i]) == PERSON_CLASS_INDEX:
                pedestrian_masks.append(mask.xy[0].astype(int))
    mask_bin = np.zeros(img.shape[:2], dtype=np.bool_)  # binary mask for the current sample_image
    if results[0].masks is not None:                
        masks = results[0].masks.data.cpu().numpy()  
        masks_hwN = masks.transpose(1, 2, 0)      
        masks_hwN = scale_image(masks_hwN, img.shape[:2]) 
        masks = masks_hwN.transpose(2, 0, 1) 
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(masks))
        for mask, tid in zip(masks, ids):
            mask_bin[mask.astype(bool)] = True    # update binary mask
                    
    fx,fy,cx,cy = camera_intrsincs[0,0],camera_intrsincs[1,1],camera_intrsincs[0,2],camera_intrsincs[1,2]
    means = []
    id_to_tlwh = {}
    
    #filter the tlwh_id according the size of the bbox
    filtered_tlwh= []
    filtered_id = []  
    for tlwh,id in zip(tlwh_id[0],tlwh_id[1]):
        if ((tlwh[2]*tlwh[3]*1.0)/(img.shape[0]*img.shape[1])) > threshold:
            filtered_tlwh.append(tlwh)
            filtered_id.append(id)
    tlwh_id = (filtered_tlwh,filtered_id)
    
    for tlwh, id in zip(tlwh_id[0],tlwh_id[1]):
        id_to_tlwh[int(id)] = tlwh
    
    depth_image, valid_pixels = project_lidar_to_image(lidar, np.linalg.inv(lidar_T_rgb), camera_intrsincs, img.shape[0], img.shape[1])
    valid_pixels_in_bbox = []
    
    # for twlh,id in zip(tlwh_id[0],tlwh_id[1]):
    #     x1,y1,w,h = twlh
    #     x2,y2 = x1+w, y1+h
    #     valid_pixels_in_bbox.append((valid_pixels[(valid_pixels[:,0] > x1) & (valid_pixels[:,0] < x2) & (valid_pixels[:,1] > y1) & (valid_pixels[:,1] < y2)],int(id)))
    valid_pixels_in_bbox = assign_depth_points_with_occlusion(valid_pixels, depth_image, zip(tlwh_id[0],tlwh_id[1]),mask=mask_bin)
    
    overlaid_img = plot_tracking(img,tlwh_id[0],tlwh_id[1])
    # for pixel in valid_pixels:
    #     cv2.circle(overlaid_img, (int(pixel[0]), int(pixel[1])), radius=3, color=(0, 255, 0), thickness=-1)
    for pixels in valid_pixels_in_bbox:
        #print(pixels[0].shape)
        #print(pixels[1])
        [cv2.circle(overlaid_img, (int(pixel[0]), int(pixel[1])), radius=3, color=get_color(pixels[1]), thickness=-1) for pixel in pixels[0]]
        
        
    cv2.imwrite(n,overlaid_img)
    
    for pixels,id in valid_pixels_in_bbox:
        if pixels.shape[0] == 0:
            continue
        depth = depth_image[pixels[:,1],pixels[:,0]]
        pixels = np.hstack((pixels,depth[:,None]))
        tlwh = id_to_tlwh[int(id)]
        pixel_center = np.array([tlwh[0] + tlwh[2]/2,tlwh[1] + tlwh[3]/2],dtype=np.int32)
        #re-project valid pixels to 3D coordinates
        reprojected_pixels = np.zeros((pixels.shape[0],4))
        reprojected_pixels[:,0] = np.multiply((pixels[:,0]-cx)/fx,depth)
        reprojected_pixels[:,1] = np.multiply((pixels[:,1]-cy)/fy,depth)
        reprojected_pixels[:,2] = depth
        reprojected_pixels[:,3] = np.ones((pixels.shape[0]))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(reprojected_pixels[:,:3])      
        cl, ind = pc.remove_statistical_outlier(nb_neighbors=int(reprojected_pixels.shape[0]), std_ratio=0.5)
        if np.array(cl.points).shape[0] == 0:
           continue
        average_z = np.array(cl.points)[:,2].mean()
        average_position = np.zeros((1,4))
        average_position[:,0] = ((pixel_center[0]-cx)/fx)*average_z
        average_position[:,1] = ((pixel_center[1]-cy)/fy)*average_z
        average_position[:,2] = average_z
        #convert the mean back to lidar frame
        reprojected_pixels = (lidar_T_rgb@average_position.T).T[:,:3]
        temp_pixels = np.array(cl.points)
        temp_pixels = (lidar_T_rgb @ np.hstack((temp_pixels,np.ones((temp_pixels.shape[0],1)))).T).T[:,:3]
        #means.append((temp_pixels,id))
        means.append((reprojected_pixels,id))
    return means

def pointcloud_to_laserscan(pointcloud: np.ndarray, params: dict) -> dict:
    """
    Converts a point cloud (Nx3 numpy array) to a 2D LaserScan-like dictionary.
    
    Args:
        pointcloud (np.ndarray): Nx3 array of points (x, y, z).
        params (dict): Dictionary with keys:
            - angle_min, angle_max, angle_increment (in radians)
            - range_min, range_max (floats)
            - min_height, max_height (floats)
            - use_inf (bool), inf_epsilon (float)
            - scan_time (optional float)

    Returns:
        dict: A dictionary representing a LaserScan message.
    """
    angle_min = params['angle_min']
    angle_max = params['angle_max']
    angle_increment = params['angle_increment']
    range_min = params['range_min']
    range_max = params['range_max']
    min_height = params.get('min_height', -np.inf)
    max_height = params.get('max_height', np.inf)
    use_inf = params.get('use_inf', True)
    inf_epsilon = params.get('inf_epsilon', 1.0)

    ranges_size = int(np.ceil((angle_max - angle_min) / angle_increment))
    if use_inf:
        ranges = np.full(ranges_size, np.inf)
    else:
        ranges = np.full(ranges_size, range_max + inf_epsilon)

    pointcloud = pointcloud[~np.isnan(pointcloud).any(axis=1)]  # Remove NaN points
    range_val = np.hypot(pointcloud[:, 0], pointcloud[:, 1]) #distance
    angles = np.arctan2(pointcloud[:, 1], pointcloud[:, 0]) #angle
    index = ((angles - angle_min) / angle_increment).astype(int)
    filtered_pointcloud = pointcloud[np.where(
                                            (range_val>=range_min) &
                                            (range_val <= range_max) &
                                            (pointcloud[:,2] >= min_height) &
                                            (pointcloud[:,2]<=max_height) & 
                                            (angles>=angle_min) &
                                            (angles <= angle_max)
                                        )] #valid points for the scan
    angles = np.arctan2(filtered_pointcloud[:, 1], filtered_pointcloud[:, 0]) #angle
    range_val = np.hypot(filtered_pointcloud[:, 0], filtered_pointcloud[:, 1]) #distance
    index = ((angles - angle_min) / angle_increment).astype(int)
    for i in np.unique(index):
        ranges[i] = np.min(range_val[np.where(index == i)])
    return ranges

def process_folders(args) -> None:
    """
    Iterate through folders and process image-pointcloud pairs.
    
    Args:
        root_dir: Root directory containing subfolders with imgs and pcds
        process_fn: Function to process image and point cloud pairs
    """
    # Convert to Path object for easier handling
    if args.input_dir:
        root_dir = Path(args.input_dir)
        root_path = Path(root_dir)
        folders = [f for f in root_path.iterdir() if f.is_dir()]
    elif args.bag:
        folders = [Path(args.bag)]
    else:
        print("No input")
        return
    
    dataset = args.dataset
    
    cfg = DATASET_CONFIG[dataset]
    
    if dataset == "go2nus":
        #apply pitch correction for go2nus dataset
        with open('calibration_fixes_go2nus/calibration_pitch_fix_by_session.txt','r') as f:
            lines = f.read().splitlines()
        calibration_fix_by_session = {}
        for line in lines:
            calibration_fix_by_session[line.split(':')[0]] = float(line.split(':')[1])

    # Iterate through all immediate subdirectories
    for folder in tqdm(folders):
        if not folder.is_dir():
            continue
        '''
        if os.path.exists(folder / 'scan.pkl') and os.path.exists(folder / 'pedestrians_3d.pkl'):
            print(f"Skipping {folder}: already processed")
            continue
        '''
        #uncomment for bev
        if not os.path.exists(folder / 'bev'):
            os.makedirs(folder / 'bev')
        
        rgb_T_lidar = cfg['rgb_T_lidar']
        
        if dataset == "go2nus":
            date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
            match = date_pattern.search(folder.name)
            session = match.group()
            pitch_correction = calibration_fix_by_session[session]
            lidar_T_rgb = np.linalg.inv(rgb_T_lidar) 
            RX = Rotation.from_euler('x', pitch_correction, degrees=True).as_matrix()
            lidar_T_pitched_rgb = lidar_T_rgb.copy()
            lidar_T_pitched_rgb[:3,:3] = lidar_T_pitched_rgb[:3,:3] @ RX   # Apply pitch rotation    
            rgb_T_lidar = np.linalg.inv(lidar_T_pitched_rgb)  # Update rgb_T_lidar with pitch correction
            print(f"Applying pitch correction {pitch_correction} for session: {session}")
            
        # Check for imgs and pcds subdirectories
        imgs_dir = folder / 'imgs'
        pcds_dir = folder / 'pcd'
        with open(folder / 'tracks.pkl','rb') as f:
            tlwh_id = pkl.load(f)
        
        with open(folder / 'traj_data.pkl','rb') as f:
            trajs = pkl.load(f)
           
        if not (imgs_dir.exists() and pcds_dir.exists()):
            print(f"Skipping {folder}: missing imgs or pcds directory")
            continue
            
        # Get sorted lists of files
        img_files = natsorted(imgs_dir.glob('*'),key=lambda x: int(x.name.replace(x.suffix,'')))
        pcd_files = natsorted(pcds_dir.glob('*'),key=lambda x: int(x.name.replace(x.suffix,'')))
        
        # Ensure we have matching numbers of files
        if len(img_files) != len(pcd_files):
            print(f"Warning: Unequal number of files in {folder}")
            continue
        
        pedestrians_3d = {}
        #bev = {}
        scan = {}
        # Process each pair
        for i in tqdm(range(len(img_files))):
            img_file = img_files[i]
            t = int(img_file.name.replace(img_file.suffix,''))
            pcd_file = os.path.join(pcds_dir, f'{t}.npz')
            img = cv2.imread(img_file)
            lidar = np.load(pcd_file)['arr_0']  
            lidar = lidar[(np.linalg.norm(lidar,axis=1) < cfg['lidar_range_max']) & (np.linalg.norm(lidar,axis=1) >cfg['lidar_range_min'])]
            #scan[t] = pointcloud_to_laserscan(lidar,cfg['scan_params'])
            #try:
            if tlwh_id[t][0] is None:
                continue
            
            annotated_img = plot_tracking(img,tlwh_id[t][0],tlwh_id[t][1])
            '''
            #uncomment for naive 3d pedestrian tracking
            try:
                #get pedestrian 3d position
                pedestrians_3d[t] = track_humans_3d_naive(
                    img,tlwh_id[t],lidar,cfg['camera_intrinsics'],
                    lidar_T_rgb=np.linalg.inv(rgb_T_lidar),
                    n=folder / f"annotated_{img_file.name}",
                    threshold=cfg['bbox_ratio_threshold']
                )
                #cv2.imwrite(folder / f"annotated_new_{img_file.name}",annotated_img)
            
            except Exception as e:
                print(f"Error processing pedestrians for {folder}\{img_file}: {e}")
                pedestrians_3d[t] = []
                continue       
            '''
            #uncomment for pcd to bev conversion
            try:
                bev = pcl_to_grid(lidar,cfg['occ_grid_params'])
                bev = np.clip(bev,0,1).T
                bev = (bev * 255).astype(np.uint8)
                cv2.imwrite(folder / 'bev' / f'{t}.png', bev)
            except Exception as e:
                print(f"Error processing BEV for {folder}\{img_file}: {e}")
                bev = []
                continue

        # with open(folder / 'scan.pkl','wb') as f:
        #     pkl.dump(scan,f)
        
        # with open(folder / 'pedestrians_3d.pkl','wb') as f:
        #     pkl.dump(pedestrians_3d,f)        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=False,
        help="Root directory containing subfolders with imgs and pcds",
    )  
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="go2nus",
        help="Dataset",
    )
    parser.add_argument(
        "--bag",
        "-b",
        type=str,
        default=False,
        help="process specific bag",
    )
    args = parser.parse_args()
    process_folders(args)