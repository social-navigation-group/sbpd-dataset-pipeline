#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import math
import sys

#!/usr/bin/env python


class PointCloudRotator:
    def __init__(self):
        rospy.init_node('rotate_point_cloud', anonymous=True)
        
        # Get parameters
        self.input_topic = rospy.get_param('~lidar_topic', '/input_cloud')
        self.output_topic = rospy.get_param('~rotated_lidar_topic', '/output_cloud')
        self.target_frame = rospy.get_param('~lidar_frame', 'rotated_frame')
        
        # Get RPY angles from command line arguments or parameters
        self.roll = float(sys.argv[1]) * math.pi/180.0
        self.pitch = float(sys.argv[2]) * math.pi/180.0
        self.yaw = float(sys.argv[3]) * math.pi/180.0

        rospy.loginfo(f"Rotating point cloud by Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}")
        
        # Create transform
        self.transform = self.create_transform()
        
        # Publishers and subscribers
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback)
        
        rospy.loginfo(f"Subscribed to: {self.input_topic}")
        rospy.loginfo(f"Publishing to: {self.output_topic}")
    
    def create_transform(self):
        # Convert RPY to quaternion
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cp = math.cos(self.pitch * 0.5)
        sp = math.sin(self.pitch * 0.5)
        cr = math.cos(self.roll * 0.5)
        sr = math.sin(self.roll * 0.5)
        
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = sy * cp * sr + cy * sp * cr
        qz = sy * cp * cr - cy * sp * sr
        
        # Create transform
        transform = TransformStamped()
        transform.header.frame_id = "original_frame"
        transform.child_frame_id = self.target_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        
        return transform
    
    def cloud_callback(self, cloud_msg):
        try:
            # Update transform timestamp
            self.transform.header.stamp = rospy.Time.now()
            
            # Transform the point cloud
            transformed_cloud = do_transform_cloud(cloud_msg, self.transform)
            
            # Update frame_id
            transformed_cloud.header.frame_id = self.target_frame
            transformed_cloud.header.stamp = cloud_msg.header.stamp
            
            # Publish transformed cloud
            self.pub.publish(transformed_cloud)
            
        except Exception as e:
            rospy.logerr(f"Error transforming point cloud: {e}")

if __name__ == '__main__':
    try:
        rotator = PointCloudRotator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass