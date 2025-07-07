import rclpy.serialization
import rosbag2_py
from tqdm import tqdm
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
import cv2

import numpy as np
import os
import pickle
import sys
import pyvista as pv
import yaml

def get_relative_odom(odom_list, start_pose_index):
    start_pose = odom_list[start_pose_index]
    T_a_inv = get_inverse(
        pose_to_matrix(position=start_pose.pose.pose.position,
                       quaternion=start_pose.pose.pose.orientation)
    )
    return np.array([
        (T_a_inv @ pose_to_matrix(odom.pose.pose.position, odom.pose.pose.orientation))[:3, 3]
        for odom in odom_list
    ])

def get_relative_pose(pose_a, pose_b):
    """
    Returns relative pose of pose_b wrt pose_a, when both pose_a and pose_b are in the same global frame.
    """
    T_a = pose_to_matrix(position=pose_a.pose.pose.position, quaternion=pose_a.pose.pose.orientation)
    T_b = pose_to_matrix(position=pose_b.pose.pose.position, quaternion=pose_b.pose.pose.orientation)
    T_a_inv = get_inverse(T_a)
    T_rel = T_a_inv @ T_b

    rel_position, rel_quat = matrix_to_pose(T_rel)
    return rel_position

def pose_to_matrix(position, quaternion):
    """Convert position and quaternion to a 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w]).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z])
    return T

def matrix_to_pose(T):
    """Convert a 4x4 transformation matrix to position and quaternion."""
    position = T[:3, 3]
    quaternion = R.from_matrix(T[:3, :3]).as_quat()
    return position, quaternion

def get_inverse(T_in):
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))

def display_pointcloud_pyvista(pointcloud_msg: PointCloud2, index: int, odom_list: list, pcl_odom_association: list, bag_file_name: str):
    distance_threshold_upper = 10.0
    distance_threshold_lower = 0.3
    z_ground_filter = -0.25
    
    # View parameters
    elevation = 25
    azimuth = 180
    view_distance = 12.5

    points_raw = list(pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points_raw:
        print("No points received.")
        return
    points_array = np.array([[p[0], p[1], p[2]] for p in points_raw], dtype=np.float32)

    x, y, z = points_array[:, 0], points_array[:, 1], points_array[:, 2]
    dists_squared = x**2 + y**2 + z**2
    
    mask = (
        (dists_squared >= distance_threshold_lower**2) &
        (dists_squared <= distance_threshold_upper**2) &
        (points_array[:, 2] > z_ground_filter)
    )

    points = points_array[mask]

    odom_trajectory = get_relative_odom(odom_list=odom_list, start_pose_index=pcl_odom_association[index])
    velocity_estimate = abs(odom_list[index].twist.twist.linear.x)

    dists = np.linalg.norm(points, axis=1)
    cmap = plt.get_cmap("cool")
    normed_dists = (dists - dists.min()) / (dists.ptp() + 1e-6)
    colors = (cmap(normed_dists)[:, :3] * 255).astype(np.uint8)

    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = colors

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("black")

    # Add the point cloud
    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=3, scalars="colors", cmap="cool", show_scalar_bar=False)
    # plotter.remove_scalar_bar()

    # Add the odometry trajectory (as white lines)
    odom_trajectory_array = np.array(odom_trajectory)
    plotter.add_lines(odom_trajectory_array, color="lightgreen", width=3, connected=True)

    # Add velocity arrow (simplified for visualization)
    # arrow_tip = [velocity_estimate * 1.5, 0, 0]
    # arrow_points = np.array([[0, 0, 0], arrow_tip])
    # plotter.add_lines(arrow_points, color="red", width=15)


    start = (0, 0, 0)
    direction = (1.0, 0, 0)
    vel_arrow = pv.Arrow(start=start, direction=direction, scale=1.5 * velocity_estimate, tip_length=0.25, tip_radius=0.1, shaft_radius=0.05)
    plotter.add_mesh(vel_arrow, color="red")

    # Set up camera view (azimuth, elevation)
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    camera_eye = [
        np.cos(azimuth_rad) * np.cos(elevation_rad) * view_distance,
        np.sin(azimuth_rad) * np.cos(elevation_rad) * view_distance,
        np.sin(elevation_rad) * view_distance,
    ]
    # plotter.view_vector(camera_eye, viewup=(0, 0, 1))
    # plotter.view_vector(vector=(-1, 0, 0.5), viewup=(0, 0, 1))
    plotter.camera_position = [
        camera_eye,   # Camera location (eye)
        (0, 0, 0),     # Focal point (look-at)
        (0, 0, 1)      # View-up direction
    ]

    plotter.show(screenshot=f"./temp_files/{bag_file_name}/cloud_images/image_{index}.png")

def process_bag_file(uri, storage_id, odom_topic, pointcloud_topic, image_topic, image_encoding_str):
   
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=uri,
        storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    metadata = reader.get_metadata()
    bag_duration_s = metadata.duration.nanoseconds / 1e9

    bag_file_name = uri.split('/')[-1]

    topic_msg_count = {}

    for topic, message_count in [(t.topic_metadata.name, t.message_count) for t in metadata.topics_with_message_count if t.topic_metadata.name in [odom_topic, pointcloud_topic, image_topic]]:
        topic_msg_count[topic] = message_count

    # Filter such that odom data is downsampled to pcl frequency
    odom_frequency = topic_msg_count[odom_topic] / bag_duration_s
    pointcloud_frequency = topic_msg_count[pointcloud_topic] / bag_duration_s
    image_frequency = topic_msg_count[image_topic] / bag_duration_s

    sampling_frequency = int(np.floor(odom_frequency / pointcloud_frequency))

    print(f"Bag file duration: {bag_duration_s} [s]\npointcloud frequency: {pointcloud_frequency}\nodom frequency {odom_frequency}\nsampling frequency {sampling_frequency}\nimage frequency {image_frequency}")

    topics_list = [odom_topic, pointcloud_topic, image_topic]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics_list))
    odom_data = []
    cloud_count = 0
    odom_count = 0
    image_count = 0

    pointcloud_timestamps = []
    odom_timestamps = []

    encoding_map = {
        "rgb": cv2.IMREAD_COLOR_RGB,  
        "bgr": cv2.IMREAD_COLOR_BGR,  
    }

    encoding = None
    for encoding_ in encoding_map.keys():
        if image_encoding_str in encoding_:
            encoding = encoding_map.get(encoding_)

    assert encoding is not None, "Invalid image encoding"

    for _ in tqdm(generator(reader=reader)):
        
        topic, raw_data, timestamp  = reader.read_next()
        if topic == pointcloud_topic:
            data_ = rclpy.serialization.deserialize_message(raw_data, message_type=PointCloud2)
            
            with open(f"./temp_files/{bag_file_name}/clouds/pcl_{cloud_count}.pkl", 'wb') as f:
                pickle.dump(data_, f)
                pointcloud_timestamps.append(timestamp)
            cloud_count += 1

        elif topic == odom_topic:
            if odom_count % sampling_frequency == 0:
                # Downsample odom data frequency to match pointcloud frequency
                data_ = rclpy.serialization.deserialize_message(raw_data, message_type=Odometry)
                odom_timestamps.append(timestamp)
                odom_data.append(data_)
            odom_count += 1

        elif topic == image_topic:
            # Write fpv images to temp folder
            compressed_image_msg = rclpy.serialization.deserialize_message(raw_data, message_type=CompressedImage)
            image = cv2.imdecode(np.frombuffer(compressed_image_msg.data, dtype=np.uint8), encoding)
            cv2.imwrite(filename=f"./temp_files/{bag_file_name}/fpv_images/image_{image_count}.png", img=image)
            image_count += 1
        
    assert len(odom_timestamps) == len(odom_data), "Total odom timestamps and data lengths do not match"
    
    with open(f"./temp_files/{bag_file_name}/odom.pkl", 'wb') as f:
        pickle.dump(odom_data, f)

    print(f"Wrote:\n{cloud_count} PCL messages\n{len(odom_data)} odom messages\n{image_count} fpv images")

    del reader

    pointcloud_timestamps = np.array(pointcloud_timestamps)
    odom_timestamps = np.array(odom_timestamps)
    # Save association
    
    odom_indices = [np.searchsorted(odom_timestamps, pcl_stamp) for pcl_stamp in pointcloud_timestamps]
    with open(f'./temp_files/{bag_file_name}/pcl_odom_association.pkl', 'wb') as f:
        pickle.dump(odom_indices, f)

    return {
        "cloud_fps": cloud_count / bag_duration_s,
        "fpv_fps" : image_count / bag_duration_s
    } 

def generator(reader):
    while reader.has_next():
        yield

def load_pcl(index, bag_file_name):
    with open(f"./temp_files/{bag_file_name}/clouds/pcl_{index}.pkl", 'rb') as f:
        pc_ = pickle.load(f)
    return pc_

def load_odom(bag_file_name):
    with open(f"./temp_files/{bag_file_name}/odom.pkl", 'rb') as f:
        odom_arr_ = pickle.load(f)
    return odom_arr_

def load_pcl_odom_association(bag_file_name):
    with open(f"./temp_files/{bag_file_name}/pcl_odom_association.pkl", 'rb') as f:
        pcl_odom_association = pickle.load(f)
    return pcl_odom_association

def load_input_params(file_path):
    with open(file=file_path) as file:
        return yaml.safe_load(file).get("annotation_video_generation_params")

def main():
    try:
        if len(sys.argv) > 2:
            base_path = str(sys.argv[1])
            bag_file_name = str(sys.argv[2])
    
        else:
            print("Invalid CLI arguments. Please provide base_path and bag_file_name.")
            sys.exit(1)

        assert os.path.basename(os.getcwd()) == "annotation_video_generation", "Error, please run the script from the annotation_video_generation folder"

        params = load_input_params("params.yaml")
        odom_topic = params.get("odom_topic")
        pointcloud_topic = params.get("pointcloud_topic")
        compressed_image_topic = params.get("compressed_image_topic")
        image_encoding = params.get("image_encoding") 

        uri = os.path.join(base_path, bag_file_name)
        fps_dict = process_bag_file(uri=uri, storage_id='sqlite3', odom_topic=odom_topic, pointcloud_topic=pointcloud_topic, image_topic=compressed_image_topic, image_encoding_str=image_encoding)

        
        with open(f"./temp_files/{bag_file_name}/fps.txt", "w") as f:
            f.write(f"FPV_FRAME_RATE={fps_dict.get('fpv_fps')}\n")
            f.write(f"CLOUD_FRAME_RATE={fps_dict.get('cloud_fps')}\n")

        print(f"FPV frame rate: {fps_dict.get('fpv_fps')} | Cloud frame rate: {fps_dict.get('cloud_fps')}")
                
        n_clouds = len([f for f in os.listdir(f"./temp_files/{bag_file_name}/clouds") if ".pkl" in f])
        print(f"Processing {n_clouds} clouds")
        odom_list = load_odom(bag_file_name=bag_file_name)
        pcl_odom_association = load_pcl_odom_association(bag_file_name=bag_file_name)
    
        for index in tqdm(range(n_clouds)):
            cloud = load_pcl(index, bag_file_name=bag_file_name)
            # start = timeit.default_timer()
            display_pointcloud_pyvista(pointcloud_msg=cloud, index=index, odom_list=odom_list, pcl_odom_association=pcl_odom_association, bag_file_name=bag_file_name)
            # print(f"Processed cloud index {index} / {n_clouds}")
            # print(timeit.default_timer() - start)

        print("Wrote cloud and fpv images.")
        
    
    except Exception as e:
        print(e.with_traceback())

if __name__ == '__main__':
    main()



