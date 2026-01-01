#!/usr/bin/env python3

import numpy as np
import time
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Joy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
import rospy
import cv2
import message_filters
import os
import pickle
from termcolor import cprint
import subprocess
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from parse_utils import BEVLidar
import yaml
import rosbag
import tf2_ros
import math
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

def get_affine_mat(x, y, theta):
    """
    Returns the affine transformation matrix for the given parameters.
    """
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


class ListenRecordData:
    def __init__(self, rosbag_play_process, config_path, viz_lidar, joy_msgs, joy_time_stamps, odom_topic,lidar_topic, odom_msgs=None, odom_time_stamps=None):
        self.rosbag_play_process = rosbag_play_process
        self.data = []
        self.recorded_odom_msgs = odom_msgs
        self.recorded_odom_time_stamps = np.asarray(odom_time_stamps)
        self.recorded_joy_msgs = joy_msgs
        self.recorded_joy_time_stamps = joy_time_stamps

        print('Subscribing to topics: ', odom_topic, ' and ', lidar_topic)
        self.lidar_dummy_sub = rospy.Subscriber(
            lidar_topic, PointCloud2, self.lidar_callback, queue_size=10) 
        
        # self.odom_dummy_sub = rospy.Subscriber(
        #     odom_topic, Odometry, self.odom_callback, queue_size=10) 
        
    def lidar_callback(self, lidar_msg):
        print("###LIDAR CALLBACK")
    
   
    def odom_callback(self, odom_msg):
        print("###ODOM CALLBACK")


if __name__ == '__main__':
    rospy.init_node('listen_record_data', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    robot_name = rospy.get_param('robot_name')
    save_data_path = rospy.get_param('save_data_path')
    viz_lidar = rospy.get_param('viz_lidar')
    odom_topic = rospy.get_param('odom_topic')
    lidar_topic = rospy.get_param('lidar_topic')
    # check if the rosbag path exists
    if not os.path.exists(rosbag_path):
        cprint('rosbag path : ' + str(rosbag_path), 'red', attrs=['bold'])
        raise FileNotFoundError('ROS bag file not found')

    # check if the save_data_path exists
    # create directory if needed
    if not os.path.exists(save_data_path):
        cprint('Creating directory : ' +
               save_data_path, 'blue', attrs=['bold'])
        os.makedirs(save_data_path)
    else:
        cprint('Directory already exists : ' +
               save_data_path, 'blue', attrs=['bold'])

    # parse the rosbag file and extract the odometry data
    cprint('First reading all the odom messages and timestamps from the rosbag',
           'green', attrs=['bold'])

    rosbag = rosbag.Bag(rosbag_path)
    info_dict = yaml.safe_load(rosbag._get_yaml_info())
    print('Conversion done!')
    # read all the odometry messages
    odom_msgs, odom_time_stamps = [], []
    for topic, msg, t in tqdm(rosbag.read_messages(topics=[odom_topic])):
        odom_msgs.append(msg)
        if len(odom_time_stamps) == 0:
            odom_time_stamps.append(0.0)
            start_time = t.to_sec()
        else:
            odom_time_stamps.append(t.to_sec())
    cprint('Done reading odom messages from the rosbag !!!',
           color='green', attrs=['bold'])

    joy_msgs, joy_time_stamps = [], []
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(
        package_root, 'config/'+str(robot_name)+'.yaml')

    duration = info_dict['end'] - info_dict['start']
    print('rosbag_length: ', duration)
    play_duration = str(int(math.floor(duration) - 6))
    print('play duration: {}'.format(play_duration))

    rosbag_play_process = subprocess.Popen(
        ['rosbag', 'play', rosbag_path, '-r', '1.0', '--clock','-q', '-u', play_duration])

    datarecorder = ListenRecordData(rosbag_play_process=rosbag_play_process,
                                    config_path=config_file_path,
                                    viz_lidar=viz_lidar,
                                    odom_msgs=odom_msgs,
                                    odom_time_stamps=odom_time_stamps,
                                    odom_topic=odom_topic,
                                    lidar_topic=lidar_topic,
                                    joy_msgs=joy_msgs,
                                    joy_time_stamps=joy_time_stamps)

    # while not rospy.is_shutdown():
    #     # check if the python process is still running
    #     if rosbag_play_process.wait() is not None:
    #         print('rosbag process has stopped')
    #         # Shutdown subscribers and wait for callbacks to finish
    #         datarecorder.shutdown_subscribers()
    #         # Now save the data
    #         datarecorder.save_data(rosbag_path, save_data_path)
    #         print('Data was saved in :: ', save_data_path)
    #         exit(0)

    rospy.spin()
