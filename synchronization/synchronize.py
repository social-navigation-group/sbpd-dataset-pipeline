import rclpy
import rclpy.serialization
import rosbag2_py
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import String, Float64, Int64
from cv_bridge import CvBridge
import numpy as np
import cv2
import apriltag
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, asdict, is_dataclass
from tqdm import tqdm
from enum import Enum
import yaml
import os
import pickle

# Dataclasses
@dataclass
class CameraIntrinsics():
    """
    Stores the intrinsic parameters of a camera.

    Attributes:
        fx (float): Focal length in the x-direction (in pixels).
        fy (float): Focal length in the y-direction (in pixels).
        cx (float): Principal point x-coordinate (optical center).
        cy (float): Principal point y-coordinate (optical center).
    """
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class Translation():
    """
    Represents the translational component of a 3D transformation.

    Attributes:
        x (float): Translation along the x-axis.
        y (float): Translation along the y-axis.
        z (float): Translation along the z-axis.
    """
    x: float
    y: float
    z: float

@dataclass
class Rotation():
    """
    Represents the rotational component of a 3D transformation as a quaternion.

    Attributes:
        x (float): x-component of the quaternion.
        y (float): y-component of the quaternion.
        z (float): z-component of the quaternion.
        w (float): w-component (scalar part) of the quaternion.
    """
    x: float
    y: float
    z: float
    w: float

@dataclass
class FrameTransform():
    """
    Represents a 3D transformation, including both translation and rotation.

    Attributes:
        translation (Translation): The translational part of the transformation.
        rotation (Rotation): The rotational part of the transformation as a quaternion.
    """
    translation: Translation
    rotation: Rotation

# Map the data values for the sync commands. TODO: Should be replaced with appropriate values as per the recorded data.
class SyncCommandMap(Enum):
    """
    Enum representing different synchronization commands used for timestamp syncing. The `data` field of the sync command message will be compared with the value of the corresponding enum.

    Attributes:
        SYNC: Command to synchronize timestamps.
        DISCARD: Command to discard a timestamp.
        BEGIN: Command indicating the start of a recorded trajectory.
        END: Command indicating the end of a recorded trajectory.
    """
    SYNC = "0"
    DISCARD = "-1"
    BEGIN = "-2"
    END = "-3"

# TODO should be replaced with the appropriate data type if not included
sync_command_message_map = {
    "String": String,
    "Float64": Float64,
    "Int64": Int64
}

def dataclass_to_dict(obj) -> dict:
    """
    Recursively converts a dataclass object (or any nested dataclass structure) into a dictionary.
    :param obj (Any): The dataclass instance or nested structure to convert.
    :return: A dictionary representation of the dataclass (and its nested dataclasses).
    """
    if is_dataclass(obj):
        return {field: dataclass_to_dict(value) if is_dataclass(value) else value 
                for field, value in asdict(obj).items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) if is_dataclass(item) else item for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) if is_dataclass(value) else value for key, value in obj.items()}
    else:
        return obj  # Return primitive values as is

def none_check(obj):
    if isinstance(obj, dict):
        return all(none_check(v) for v in obj.values())
    elif isinstance(obj, list):
        return all(none_check(v) for v in obj)
    else:
        return obj is not None

def load_input_params(file_path):
    with open(file=file_path) as file:
        return yaml.safe_load(file).get("input_params")
    
def write_to_yaml(file_path, dict_to_write):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data=dict_to_write, stream=file, sort_keys=False, default_flow_style=False)

# Runs a qr code detection and returns the decoded value as a string.
def compute_qr_code_timestamp(image, verbose=False):
    """
    Detects and decodes a QR code from an image to extract a timestamp.

    :param image: Input image containing a QR code.
    :param verbose: If True, prints the decoded timestamp information. Defaults to False.
    :return: A tuple (status, timestamp) where status is True if decoding is successful, and timestamp is the decoded string value or -1 if not found.
    """
    qr_detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(image)
    if retval and decoded_info[0] != '':
        if verbose:
            print(f'Decoded timestamp: {decoded_info}')
        return (True, decoded_info[0])
    else:
        return False, -1

# Reads the rosbag, gets time sync information and sync command signals
def get_time_sync_offset_and_sync_commands(rosbag_uri: str, rosbag_storage_id: str, rosbag_compression: bool, robot_camera_topic: str, robot_camera_compression: str, n_detections_threshold: int, sync_command_topic: str, sync_command_message_type):
   
    # Create ROSbag reader
    if rosbag_compression:
        reader = rosbag2_py.SequentialCompressionReader()
    else:
        reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=rosbag_uri,
        storage_id=rosbag_storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    metadata = reader.get_metadata()
    start_time = metadata.starting_time
    end_time = metadata.starting_time + metadata.duration
    print(f"Opened bag with start time: {start_time.nanoseconds} and end time: {end_time.nanoseconds}")

    reader.set_filter(rosbag2_py.StorageFilter(topics=[robot_camera_topic, sync_command_topic]))
    n_detections = 0
    detections = []
    detection_successful = False
    
    sync_commands = []
    image_message_type = CompressedImage if robot_camera_compression else Image
    bridge = CvBridge()

    for _ in tqdm(_helper_rosbag_generator(reader=reader)):
        topic, raw_data, timestamp  = reader.read_next()
        
        # Process images till (n_detections_threshold) QR codes are detected
        if topic == robot_camera_topic and not detection_successful:
            image_message = rclpy.serialization.deserialize_message(raw_data, message_type=image_message_type)
            # image = cv2.imdecode(np.frombuffer(image_message.data, dtype=np.uint8), cv2.IMREAD_COLOR)

            image = bridge.compressed_imgmsg_to_cv2(image_message, desired_encoding='passthrough') if robot_camera_compression else bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
            
            success, detection = compute_qr_code_timestamp(image=image, verbose=False)

            if success:
                n_detections += 1
                sec, nsec = detection.split('.')
                detection = 1e9 * int(sec) + int(nsec)

                detections.append((detection, timestamp))
                if n_detections > n_detections_threshold:
                    print(f'Detected {n_detections_threshold} QR codes in robot camera data.')
                    detection_successful = True

        # Store sync commands
        elif topic == sync_command_topic:
            sync_command_msg = rclpy.serialization.deserialize_message(raw_data, message_type=sync_command_message_type)
            sync_commands.append((timestamp, sync_command_msg))

    offset = 0
    del reader
    if n_detections > 0:
        for image_timestamp, ros_timestamp in detections:
            offset += image_timestamp - ros_timestamp
        return offset / (len(detections) * 1e9), sync_commands

    else:
        print(f'Time sync QR code detection failed.')
        return None, None             

# Returns the ROS timestamp for the start of the video file
def get_timestamp_association(video_file_uri, offset, skip_seconds, n_detections_threshold):
    """
    Function to process video file and calculate timestamp association.
    Returns the ROS timestamp corresponding to video start timestamp
    """

    video_capture = cv2.VideoCapture(filename=video_file_uri)
    assert video_capture is not None, "Invalid video file"

    fps, n_frames = video_capture.get(cv2.CAP_PROP_FPS), video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video file with fps: {fps} and total expected frames: {n_frames}")

    n_detections = 0
    n_frames_counted = fps * skip_seconds
    empty_frame_count = 0
    detections = []

    target_frame = int(skip_seconds * fps)
    assert target_frame <= n_frames, f"Target frame is outside the video file when skipping {skip_seconds} [s]"
    frame_set_success = video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)

    if frame_set_success:
        print(f"Reading video file {video_file_uri} from {skip_seconds} [s]")

        # Get timestamp mapping for time synchronization
        for _ in tqdm(_helper_video_generator(video_capture=video_capture)):
            ret, frame = video_capture.read()
            video_timestamp = n_frames_counted / fps
            
            # GoPro specific filter, to handle random invalid frames in GoPro videos
            # reference: https://stackoverflow.com/questions/49060054/opencv-videocapture-closes-with-videos-from-gopro
            if not ret:
                if n_frames_counted < n_frames:
                    empty_frame_count += 1
                    continue
                else:
                    print(f"Reached end of video file")
                    break

            n_frames_counted += 1

            success, detection = compute_qr_code_timestamp(image=frame, verbose=False)

            if success:
                n_detections += 1
                detection = float(detection)
                detections.append((detection, video_timestamp))
                if n_detections > n_detections_threshold:
                    print(f'Detected {n_detections_threshold} QR codes in BEV camera data.')
                    break

        if n_detections > 0:
            ros_timestamps, video_timestamps = zip(*detections)
            ros_timestamp = sum(ros_timestamps) / len(ros_timestamps) 
            assoc_video_timestamp = sum(video_timestamps) / len(video_timestamps) - offset
            return ros_timestamp - assoc_video_timestamp
        else:
            print(f"Could not detect QR code timestamp in video file.")
    
    else:
        print(f"Failed to set to frame {target_frame} after skipping first {skip_seconds} [s] in video.")

# Helper function for tqdm
def _helper_rosbag_generator(reader):
    while reader.has_next():
        yield

# Helper function for tqdm
def _helper_video_generator(video_capture):
    while video_capture.isOpened():
        yield

def get_apriltag_pose(frame: np.ndarray, intrinsics: CameraIntrinsics, tag_size=0.9, tag_family="tag36h11", verbose=False):
    """
    Detects apriltags in `frame`, and computes their poses using the provided camera `intrinsics`.

    Parameters:
        frame (numpy.ndarray): image frame
        intrinsics (CameraIntrinsics)
        tag_size (float): The size of the AprilTag [m]
        tag_family (str): The family of the AprilTag to detect (default: "tag36h11")

    Returns:
        frame (with annotated detection(s) if detected)
        None if no detections
    """

    # Initialize the AprilTag detector
    detector_options = apriltag.DetectorOptions(families=tag_family)
    detector = apriltag.Detector(detector_options)

    # Detect AprilTags in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    if not detections:
        # print("No AprilTags detected in the image.")
        return frame, None

    # Camera parameters
    camera_params = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]

    for detection in detections:
    
        (ptA, ptB, ptC, ptD) = detection.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the frame
        tagFamily = detection.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Compute the pose of the detected tag
        pose, e0, e1 = detector.detection_pose(detection, camera_params, tag_size)

        # Extract translation and rotation components
        translation = pose[:3, 3]
        rotation_matrix = pose[:3, :3]

        # Print the results if verbose argument
        if verbose:
            print(f"Detected tag ID: {detection.tag_id}")
            print(f"Translation (x, y, z): {translation}")
            print("Rotation matrix:")
            print(rotation_matrix)
            print("---")
        rotation_ = R.from_matrix(rotation_matrix)
        quaternion = rotation_.as_quat()
        camera_optical_to_tag_transform = FrameTransform(
            translation=Translation(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
            rotation=Rotation(x=float(quaternion[0]), y=float(quaternion[1]), z=float(quaternion[2]), w=float(quaternion[3]))
        )

        return frame, camera_optical_to_tag_transform

def get_position_sync_command_timestamps(sync_commands):
    sync_timestamps = []
    for timestamp, sync_command in sync_commands:
        if sync_command.data == SyncCommandMap.SYNC.value:
            sync_timestamps.append(timestamp)

    if len(sync_timestamps) > 0:
        return sync_timestamps
    else:
        return None

def get_tag_poses_wrt_robot(rosbag_uri: str, rosbag_storage_id: str, robot_camera_topic: str, camera_info_topic: str, sync_timestamp):
    # Create ROSbag reader
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=rosbag_uri,
        storage_id=rosbag_storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    metadata = reader.get_metadata()
    start_time = metadata.starting_time
    end_time = metadata.starting_time + metadata.duration
    print(f"Opened bag with start time: {start_time.nanoseconds} and end time: {end_time.nanoseconds}")

    reader.set_filter(rosbag2_py.StorageFilter(topics=[robot_camera_topic, camera_info_topic]))
    tag_poses = []
    
    time_threshold = 2 * 1e9    # in milliseconds
    read_camera_info = False

    for _ in tqdm(_helper_rosbag_generator(reader=reader)):
        topic, raw_data, timestamp  = reader.read_next()

        if topic == camera_info_topic and not read_camera_info:
            # Read and store camera info
            camera_info = rclpy.serialization.deserialize_message(raw_data, message_type=CameraInfo)
            K = camera_info.k
            camera_intrinsics = CameraIntrinsics(fx=K[0], fy=K[4], cx=K[2], cy=K[5])
            read_camera_info = True

        elif topic == robot_camera_topic:
            # Process image data
            if timestamp < sync_timestamp - time_threshold:
                continue
            elif abs(timestamp - sync_timestamp) <= time_threshold: 
                # Process images
                compressed_image_msg = rclpy.serialization.deserialize_message(raw_data, message_type=CompressedImage)
                image = cv2.imdecode(np.frombuffer(compressed_image_msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                frame, camera_optical_to_tag_transform = get_apriltag_pose(frame=image, intrinsics=camera_intrinsics, )

                if camera_optical_to_tag_transform is not None:
                    tag_poses.append((camera_optical_to_tag_transform, timestamp))
            else:
                print(f"Searched beyond sync timestamp {sync_timestamp}")
                break
        
    if len(tag_poses) > 0:
        return tag_poses
    else:
        print(f"Could not detect AprilTag in specified timestamp range")

def get_tag_poses_wrt_bev(video_file_uri, sync_timestamp, bev_camera_intrinsics):
    video_capture = cv2.VideoCapture(filename=video_file_uri)
    assert video_capture is not None, "Invalid video file"

    fps, n_frames = video_capture.get(cv2.CAP_PROP_FPS), video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video file with fps: {fps} and total expected frames: {n_frames}")

    empty_frame_count = 0
    tag_poses = []

    # Set to read frames around 2 sec interval for the sync timestamp
    detection_time_interval = 2
    target_frame = int((sync_timestamp - detection_time_interval) * fps)
    assert target_frame <= n_frames, "target frame is outside the video file"
    success = video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    
    if success:

        n_frames_counted = target_frame - 1
        # Get timestamp mapping for time synchronization
        for _ in tqdm(_helper_video_generator(video_capture=video_capture)):
            ret, frame = video_capture.read()
            video_timestamp = n_frames_counted / fps

            if video_timestamp < sync_timestamp - detection_time_interval:
                n_frames_counted += 1
                continue

            if video_timestamp > sync_timestamp + detection_time_interval:
                print(f"Reached beyond target timestamp.")
                break
            
            # GoPro specific filter, to handle random invalid frames in GoPro videos
            # reference: https://stackoverflow.com/questions/49060054/opencv-videocapture-closes-with-videos-from-gopro
            if not ret:
                if n_frames_counted < n_frames:
                    empty_frame_count += 1
                    continue
                else:
                    print(f"Reached end of video file")
                    break

            n_frames_counted += 1

            frame, camera_optical_to_tag_transform = get_apriltag_pose(frame=frame, intrinsics=bev_camera_intrinsics, tag_size=0.72)

            if camera_optical_to_tag_transform is not None:
                tag_poses.append((camera_optical_to_tag_transform, video_timestamp))


        if len(tag_poses) > 0:
            return tag_poses
        else:
            print(f"Could not detect QR code timestamp in video file.")
            return None
    
    else:
        print(f"Failure in processing video file.")

def main():

    # Input parameters
    # rosbag_uri = "/home/pranav/go2_rosbags/dataset/apr_02/terrace_new_3_with_overhead_2025-04-02_11-43-48"                 # Path to rosbag file
    # rosbag_storage_id = "sqlite3"   # {sqlite3, mcap}
    # rosbag_compression = False
    # robot_camera_topic = "/camera/camera/color/image_raw/compressed"
    # robot_camera_compression = True
    # camera_info_topic = "/camera/camera/color/camera_info"
    # sync_command_topic = "/sync_command"
    # video_file_uri = "/home/pranav/go2_bev_videos/apr_02/terrace_3_4/cam_01/GX010100.MP4"             # Path to video file
    # sync_command_message_type = String
    # bev_skip_time_seconds = 200
    # bev_camera_intrinsics = CameraIntrinsics(fx = 1.00555621e+03, fy = 1.00425365e+03, cx = 9.66122376e+02, cy = 5.36938213e+02)

    params = load_input_params(file_path="input_params.yaml")
    assert none_check(params), "Invalid params, please check for None values"

    # Populate params
    rosbag_uri = params.get("rosbag_uri")
    rosbag_storage_id = params.get("rosbag_storage_id")
    rosbag_compression = params.get("rosbag_compression")
    robot_camera_topic = params.get("robot_camera_topic")
    robot_camera_info_topic = params.get("robot_camera_info_topic")
    robot_camera_compression = params.get("robot_camera_compression")
    sync_command_topic = params.get("sync_command_topic")

    video_file_uri = params.get("video_file_uri")
    bev_skip_time_seconds = params.get("bev_skip_time_seconds")

    sync_command_message_type = sync_command_message_map.get(params.get("sync_command_message_type"), None)
    assert sync_command_message_type is not None, "Please specify valid sync command message type"
    
    bev_camera_intrinsics = CameraIntrinsics(
        fx=params.get("bev_camera_intrinsics").get("fx"),
        fy=params.get("bev_camera_intrinsics").get("fy"),
        cx=params.get("bev_camera_intrinsics").get("cx"),
        cy=params.get("bev_camera_intrinsics").get("cy")
    )
    
    # """
    # Calculate time synchrnonization
    offset, sync_commands = get_time_sync_offset_and_sync_commands(rosbag_uri=rosbag_uri, rosbag_storage_id=rosbag_storage_id, rosbag_compression=rosbag_compression, robot_camera_topic=robot_camera_topic, robot_camera_compression=robot_camera_compression, n_detections_threshold=5, sync_command_topic=sync_command_topic, sync_command_message_type=sync_command_message_type)
    print(f"Offset between ROS bagging time and QR code time: {offset} [s]")

    # ROS time corresponding to start of video file
    video_start_ros_timestamp = get_timestamp_association(video_file_uri=video_file_uri, offset=offset, skip_seconds=bev_skip_time_seconds, n_detections_threshold=5)
    print("ROS timestamp for video start: ", video_start_ros_timestamp)

    # Position synchrnonization
    position_sync_command_timestamps = get_position_sync_command_timestamps(sync_commands)    

    for sync_timestamp in position_sync_command_timestamps:

        # Search for AprilTag around these timestamps in both ROSbag and video file. Searches for all timestamps since there may be sync commands for multiple BEV cameras in the same rosbag.
        tag_poses_wrt_robot = get_tag_poses_wrt_robot(rosbag_uri=rosbag_uri, rosbag_storage_id=rosbag_storage_id, robot_camera_topic=robot_camera_topic, camera_info_topic=robot_camera_info_topic, sync_timestamp=sync_timestamp)

        video_timestamp = sync_timestamp / 1e9 - video_start_ros_timestamp

        print(f"Searching around video timestamp {video_timestamp}")

        tag_poses_wrt_bev = get_tag_poses_wrt_bev(video_file_uri=video_file_uri, sync_timestamp=video_timestamp, bev_camera_intrinsics=bev_camera_intrinsics)

        if tag_poses_wrt_bev is not None:

            robot_timestamps = np.array([data[-1] / 1e9 for data in tag_poses_wrt_robot])   
            bev_timestamps = np.array([data[-1] + video_start_ros_timestamp for data in tag_poses_wrt_bev])

            association_time_difference_tolerance = 0.5
            least_diff = None
            for index, timestamp in enumerate(robot_timestamps):
                min_diff = np.min(abs(bev_timestamps - timestamp))
                
                if least_diff is None:
                    least_diff = min_diff
                
                if min_diff < association_time_difference_tolerance and min_diff < least_diff:
                    least_diff = min_diff
                    tag_pose_robot = tag_poses_wrt_robot[index][0]
                    tag_pose_robot_timestamp = float(tag_poses_wrt_robot[index][1])
                    tag_pose_bev = tag_poses_wrt_bev[np.argmin(abs(bev_timestamps - timestamp))][0]
                    tag_pose_bev_timestamp = float(bev_timestamps[np.argmin(abs(bev_timestamps - timestamp))])

    # """
    """
    tag_pose_bev = FrameTransform(
        translation=Translation(x=1.0, y=2.0, z=3.0),
        rotation=Rotation(x=0.0, y=0.0, z=0.0, w=1.0)
    )

    tag_pose_robot = FrameTransform(
        translation=Translation(x=1.0, y=2.0, z=3.0),
        rotation=Rotation(x=0.0, y=0.0, z=0.0, w=1.0)
    )

    tag_pose_robot_timestamp = -1
    tag_pose_bev_timestamp = -1
    video_start_ros_timestamp = -1
    """

    if tag_pose_robot is not None and tag_pose_bev is not None:

        output_params = {
            "rosbag_uri": rosbag_uri,
            "video_uri": video_file_uri,
            "video_start_ros_timestamp": video_start_ros_timestamp,
            "tag_pose_wrt_robot" : {
                "pose" : dataclass_to_dict(tag_pose_robot),
                "timestamp" : tag_pose_robot_timestamp / 1e9
            },
            "tag_pose_wrt_bev" : {
                "pose" : dataclass_to_dict(tag_pose_bev),
                "timestamp" : tag_pose_bev_timestamp
            },
        }   

        output_yaml_path = f"{os.path.dirname(video_file_uri)}/" + f"{os.path.splitext(os.path.basename(video_file_uri))[0]}_sync_info.yaml"

        write_to_yaml(file_path=output_yaml_path, dict_to_write=output_params)
        print(f"Wrote output: {output_params} to {output_yaml_path}")

    else:
        print(f"Failed to process synchronization.")

if __name__ == '__main__':
    main()

