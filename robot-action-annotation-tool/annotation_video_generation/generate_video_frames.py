import rclpy.serialization
import rclpy.time
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

from collections import deque

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

def display_pointcloud_pyvista(pointcloud_msg: PointCloud2, index: int, odom_list: list, bag_file_name: str):
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

    odom_trajectory = get_relative_odom(odom_list=odom_list, start_pose_index=index)
    velocity_estimate = abs(np.sqrt(odom_list[index].twist.twist.linear.x**2 +  odom_list[index].twist.twist.linear.y**2))
    
    # If velocity estimate is not provided in odom message, estimating it from consecutive poses
    if velocity_estimate == 0 and index > 0:
        # print(f"x1: {odom_list[index].pose.pose.position.x} x2: {odom_list[index - 1].pose.pose.position.x} y1: {odom_list[index].pose.pose.position.y} y2: {odom_list[index - 1].pose.pose.position.y}")
        distance_covered = np.sqrt(
            (odom_list[index].pose.pose.position.x - odom_list[index - 1].pose.pose.position.x)**2 + (odom_list[index].pose.pose.position.y - odom_list[index - 1].pose.pose.position.y)**2
        )
        time_interval = (rclpy.time.Time.from_msg(odom_list[index].header.stamp).nanoseconds - rclpy.time.Time.from_msg(odom_list[index - 1].header.stamp).nanoseconds) / 1e9
        
        # Calculate with earlier consecutive poses if not enough timestamp difference
        if time_interval < 1e-2 and index > 1:
            distance_covered = np.sqrt(
            (odom_list[index].pose.pose.position.x - odom_list[index - 2].pose.pose.position.x)**2 + (odom_list[index].pose.pose.position.y - odom_list[index - 2].pose.pose.position.y)**2
            )
            time_interval = (rclpy.time.Time.from_msg(odom_list[index].header.stamp).nanoseconds - rclpy.time.Time.from_msg(odom_list[index - 2].header.stamp).nanoseconds) / 1e9

        velocity_estimate = distance_covered / (time_interval + 1e-6)
    
    # Clip at max value for visualization
    velocity_estimate = np.clip(velocity_estimate, -1.5, 1.5)

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

    # Add the odometry trajectory (as lines)
    odom_trajectory_array = np.array(odom_trajectory)
    plotter.add_lines(odom_trajectory_array, color="lightgreen", width=3, connected=True)

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

    plotter.camera_position = [
        camera_eye,   # Camera location (eye)
        (0, 0, 0),     # Focal point (look-at)
        (0, 0, 1)      # View-up direction
    ]

    plotter.show(screenshot=f"./temp_files/{bag_file_name}/cloud_images/image_{index}.png")

def get_timestamp_from_stamped_message(stamped_msg):
        assert hasattr(stamped_msg, 'header'), "Message does not contain header."
        return rclpy.time.Time.from_msg(stamped_msg.header.stamp).nanoseconds

def read_odometry_from_bag_file(uri, storage_id, odom_topic):
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

    for topic, message_count in [(t.topic_metadata.name, t.message_count) for t in metadata.topics_with_message_count if t.topic_metadata.name in [odom_topic]]:
        topic_msg_count[topic] = message_count

    odom_frequency = topic_msg_count[odom_topic] / bag_duration_s

    print(f"Bag file {bag_file_name} duration: {bag_duration_s} [s]\nodom frequency {odom_frequency}")

    topics_list = [odom_topic]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics_list))
    odom_data = []
    odom_timestamps_dict = {}
    odom_index = 0

    for _ in tqdm(generator(reader=reader)):
        
        topic, raw_data, timestamp  = reader.read_next()
        data_ = rclpy.serialization.deserialize_message(raw_data, message_type=Odometry)
        odom_data.append((data_))
        odom_timestamps_dict[get_timestamp_from_stamped_message(data_)] = odom_index
        odom_index += 1

    return odom_data, odom_timestamps_dict

def process_bag_file(uri, storage_id, odom_topic, pointcloud_topic, image_topic, image_encoding_str):
    
    odom_list, odom_timestamps_dict = read_odometry_from_bag_file(uri=uri, storage_id=storage_id, odom_topic=odom_topic)

    odom_timestamps_sorted = np.array([keys:=sorted(odom_timestamps_dict.keys())]).flatten()

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

    for topic, message_count in [(t.topic_metadata.name, t.message_count) for t in metadata.topics_with_message_count if t.topic_metadata.name in [pointcloud_topic, image_topic]]:
        topic_msg_count[topic] = message_count

    pointcloud_frequency = topic_msg_count[pointcloud_topic] / bag_duration_s
    image_frequency = topic_msg_count[image_topic] / bag_duration_s

    encoding_map = {
        "rgb": cv2.IMREAD_COLOR_RGB,  
        "bgr": cv2.IMREAD_COLOR_BGR,  
    }

    encoding = None
    for encoding_ in encoding_map.keys():
        if image_encoding_str in encoding_:
            encoding = encoding_map.get(encoding_)

    assert encoding is not None, "Invalid image encoding"

    print(f"Bag file duration: {bag_duration_s} [s]\npointcloud frequency: {pointcloud_frequency}\nimage frequency {image_frequency}")

    topics_list = [pointcloud_topic, image_topic]
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics_list))
    cloud_count = 0
    image_count = 0

    # Queue to process batches of pointclouds
    pointcloud_queue = deque()
    associated_odom_list = []
    processed_pointcloud_index = 0
    
    PROCESSING_INTERVAL_MSG_COUNT = int(pointcloud_frequency)

    association_diff = []

    # Internal function to process pointcloud queues at regular intervals
    def process_pointcloud_queue():
        if not pointcloud_queue:
            return
        
        nonlocal processed_pointcloud_index, odom_list, odom_timestamps_sorted, odom_timestamps_dict, associated_odom_list, association_diff
        
        n_pc_to_process_ = len(pointcloud_queue)
        best_pairs = []

        for pc, pc_stamp in pointcloud_queue:
            closest_odom = None

            # search for index at which pc_stamp must be inserted in odom_timestamps_sorted to maintain sorted order
            closest_odom_timestamp_ = odom_timestamps_sorted[np.abs(odom_timestamps_sorted - pc_stamp).argmin()]
            closest_odom_index_ = odom_timestamps_dict[closest_odom_timestamp_]
            closest_odom = odom_list[closest_odom_index_]

            best_pairs.append((pc, closest_odom))
            association_diff.append(abs(closest_odom_timestamp_ - pc_stamp))
            

        assert len(best_pairs) == n_pc_to_process_, f"ERROR: Could not find odometry associations for all pointclouds at pc index {processed_pointcloud_index}"

        for pc, odom in best_pairs:
            write_cloud_pkl(data=pc, bag_file_name=bag_file_name, index=processed_pointcloud_index)
            associated_odom_list.append(odom)
            processed_pointcloud_index += 1
            assert len(associated_odom_list) == processed_pointcloud_index, "ERROR: Unequal lengths of pointcloud and associated odom lists"

        pointcloud_queue.clear()

    print("Processing point clouds...")
    for _ in tqdm(generator(reader=reader)):
        
        topic, raw_data, timestamp  = reader.read_next()

        if topic == pointcloud_topic:
            data_ = rclpy.serialization.deserialize_message(raw_data, message_type=PointCloud2)
            pointcloud_queue.append((data_, get_timestamp_from_stamped_message(data_)))           
            cloud_count += 1

        elif topic == image_topic:
            # Write fpv images to temp folder
            compressed_image_msg = rclpy.serialization.deserialize_message(raw_data, message_type=CompressedImage)
            image = cv2.imdecode(np.frombuffer(compressed_image_msg.data, dtype=np.uint8), encoding)
            image_count += 1
            cv2.imwrite(filename=f"./temp_files/{bag_file_name}/fpv_images/image_{image_count}.png", img=image)

        if cloud_count % PROCESSING_INTERVAL_MSG_COUNT == 0:
            process_pointcloud_queue()

    # Process point clouds left in queue left not captured after the last modulo
    process_pointcloud_queue()

    print(f"Association difference mean: {np.mean(association_diff) / 1e9} [s] max: {np.max(association_diff) / 1e9} [s]")

    print(f"Length of odometry list: {len(associated_odom_list)} | Number of pointclouds processed {processed_pointcloud_index}")

    with open(f"./temp_files/{bag_file_name}/odom.pkl", 'wb') as f:
        pickle.dump(associated_odom_list, f)

    print(f"Wrote:\n{cloud_count} PCL messages\n{len(associated_odom_list)} odom messages\n{image_count} fpv images")

    del reader

    return {
        "cloud_fps": processed_pointcloud_index / bag_duration_s,
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

def load_input_params(file_path):
    with open(file=file_path) as file:
        return yaml.safe_load(file).get("annotation_video_generation_params")

def write_cloud_pkl(data, bag_file_name, index):
    with open(f"./temp_files/{bag_file_name}/clouds/pcl_{index}.pkl", 'wb') as f:
        pickle.dump(data, f)

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
        print(f"Generating images from {n_clouds} clouds")
        odom_list = load_odom(bag_file_name=bag_file_name)
    
        for index in tqdm(range(n_clouds)):
            cloud = load_pcl(index, bag_file_name=bag_file_name)
            # start = timeit.default_timer()
            display_pointcloud_pyvista(pointcloud_msg=cloud, index=index, odom_list=odom_list, bag_file_name=bag_file_name)
            # print(f"Processed cloud index {index} / {n_clouds}")
            # print(timeit.default_timer() - start)

        print("Wrote cloud and fpv images.")
        
    
    except Exception as e:
        print(e.with_traceback())

if __name__ == '__main__':
    main()



