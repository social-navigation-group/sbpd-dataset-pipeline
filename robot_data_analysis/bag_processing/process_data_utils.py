#https://github.com/robodhruv/visualnav-transformer
from collections import defaultdict
import numpy as np
import io
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF
import scipy.spatial.transform as transform
# from img_utils import *
#from ros_msg_python_classes import *
#import velodyne_decoder as vd
from sensor_msgs.msg import PointCloud2
from velodyne_msgs.msg import VelodyneScan
import open3d as o3d
import ros2_numpy
from pypcd import pypcd

def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps

def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw

def process_nus_dog_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    IMAGE_SIZE = (848, 480)
    IMAGE_ASPECT_RATIO = 16 / 9
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to aspect ratio
    img = img.resize(IMAGE_SIZE)
    return img

def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    IMAGE_SIZE = (1280, 720)
    IMAGE_ASPECT_RATIO = 16 / 9
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img

def process_odom_vel(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    rolls = []
    pitches = []
    yaws = []
    linear_v = []
    angular_v = []
    for odom_msg in odom_list:
        xy,yaw, linear_vel,angular_vel = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
        linear_v.append(linear_vel)
        angular_v.append(angular_vel)
    return {"position": np.array(xys), "yaw": np.array(yaws),"linear_vel": np.array(linear_v), "angular_vel": np.array(angular_v)}

def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}
  
def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw

def nav_to_xyz_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y, position.z], yaw

def nav_to_xy_yaw_vel(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    #roll, pitch, yaw = R.from_quat(()
    v = odom_msg.twist.twist.linear
    w = odom_msg.twist.twist.angular
    return [position.x, position.y],yaw, [v.x, v.y], w.z

def process_velodyne(msg) -> np.ndarray:
    """
    Process pcl data from a topic that publishes velodyne_msgs/VelodyneScan and convert it to a numpy array
    """
    decoder = vd.ScanDecoder(vd.Config())
    decoded_msg = decoder.decode_message(msg)
    return decoded_msg[1][:,:3]

def process_data(msg_list, process_func) -> List:
    """
    Process data from a topic that publishes sensor_msgs/PointCloud2 and convert it to a numpy array
    """
    data = [process_func(msg) for msg in msg_list if msg is not None]
    return data

def process_pointcloud2_with_intensity(msg)-> np.ndarray:
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    '''
    cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(msg)
    # remove crap points
    mask = np.isfinite(cloud_array['x']) & \
            np.isfinite(cloud_array['y']) & \
            np.isfinite(cloud_array['z']) & \
            np.isfinite(cloud_array['intensity'])
    cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (4,), dtype=np.float32)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    points[...,3] = cloud_array['intensity']

    return points

def process_pointcloud2_pypcd(msg) -> np.ndarray:
    cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(msg)
    # remove crap points
    mask = np.isfinite(cloud_array['x']) & \
            np.isfinite(cloud_array['y']) & \
            np.isfinite(cloud_array['z']) & \
            np.isfinite(cloud_array['intensity'])
    cloud_array = cloud_array[mask]
    pc = pypcd.PointCloud.from_array(cloud_array)
    return pc

def process_pointcloud2(msg) -> np.ndarray:
    """
    Process pcl data from a topic that publishes sensor_msgs/PointCloud2 and convert it to a numpy array
    """
    return ros2_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)

def process_velodyne(msg) -> np.ndarray:
    """
    Process pcl data from a topic that publishes velodyne_msgs/VelodyneScan and convert it to a numpy array
    """
    decoder = vd.ScanDecoder(vd.Config())
    decoded_msg = decoder.decode(msg)
    return decoded_msg[1][:,:3]

#######################################################################
def process_detection2d(detections):
    tlwhs = []
    obj_ids = []
    if detections is None:
        return (None,None)
    for box in detections.detections:
        x = 2*box.bbox.center.position.x - box.bbox.size_x
        y = 2*box.bbox.center.position.y - box.bbox.size_y 
        w = box.bbox.size_x
        h = box.bbox.size_y
        tlwhs.append([x, y, w, h])
        obj_ids.append(box.id)
    return tlwhs,obj_ids

# cut out non-positive velocity segments of the trajectory
def filter_trajectories(
    img_list: List[Image.Image],
    traj_data: Dict[str, np.ndarray],
    pcl_list: List[Any],
    feat_list: List[Any]=None,
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Args:
        traj_type: type of trajectory to cut
        img_list: list of images
        traj_data: dictionary of position and yaw data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        trajs: list of trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    trajs = []

    traj_pairs = []
    imgs = []
    trajs = {"position": [], "yaw": []}
    pcl = []
    features = defaultdict(list)
    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        imgs.append(img_list[i - 1])
        trajs["position"].append(traj_pos[i - 1])
        trajs["yaw"].append(traj_yaws[i - 1])
        pcl.append(pcl_list[i - 1])
        if feat_list is not None:
            for feature, values in feat_list.items():
                features[feature].append(values[i - 1])  

    #trajs.append(traj_pairs)
    return [traj_pairs]

def get_images_pcl_and_odom(
    synced_imdata, 
    synced_odomdata, 
    synced_pcldata, 
    synced_scandata,
    synced_trackingdata,
    img_process_func: Any,
    pcl_process_func: Any,
    odom_process_func: Any,
    scan_process_func: Any,
    tracking_process_func: Any,
    ang_offset: float = 0.0,
    rosversion: int = 1
):          
    img_data = process_data(synced_imdata, eval(img_process_func))
    traj_data = process_odom_vel(
        synced_odomdata,
        eval(odom_process_func),
        ang_offset=ang_offset,
    )
    pcl_data = None
    if len(synced_pcldata)!=0: 
        if type(synced_pcldata[0]) == PointCloud2:
            # if the pcl data is in bytes, we need to decode it
            pcl_data = [process_pointcloud2_with_intensity(msg) for msg in synced_pcldata]
            #pcl_data = [process_pointcloud2_pypcd(msg) for msg in synced_pcldata]
        elif type(synced_pcldata[0]) == VelodyneScan:
            pcl_data = [process_velodyne(msg) for msg in synced_pcldata]
        else:
            raise ValueError(f"Unsupported pcl data type: {type(synced_pcldata[0])}")
        
    scan_data = None
    if len(synced_scandata)!=0:
        scan_data = process_data(
            synced_scandata,
            eval(scan_process_func)
        )
    track_data = None
    if len(synced_trackingdata)!=0:
        process_func = eval(tracking_process_func)
        track_data = [process_func(msg) for msg in synced_trackingdata]
        
    return img_data, traj_data, pcl_data, scan_data, track_data