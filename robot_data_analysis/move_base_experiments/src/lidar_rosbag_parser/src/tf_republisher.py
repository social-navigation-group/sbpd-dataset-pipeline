#!/usr/bin/env python3
import tf2_ros
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from termcolor import cprint
import message_filters
from sensor_msgs.msg import PointCloud2, Joy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
class TFRepublisher:
	def __init__(self,topic_name):
		# setup tf2 publisher
		self.tf2_pub = tf2_ros.TransformBroadcaster()

		# odom callback
		self.odom_sub = rospy.Subscriber(topic_name, Odometry, self.odom_callback)

		cprint('Initialized TF Republisher', 'green', attrs=['bold'])
  
  
	def odom_callback(self, odom):
		tf = TransformStamped()
		tf.header.stamp = odom.header.stamp #rospy.Time.now()
		tf.header.frame_id = 'odom'
		tf.child_frame_id = 'base_link'
		tf.transform.translation.x = odom.pose.pose.position.x
		tf.transform.translation.y = odom.pose.pose.position.y
		tf.transform.translation.z = 0.0
		tf.transform.rotation.x = odom.pose.pose.orientation.x
		tf.transform.rotation.y = odom.pose.pose.orientation.y
		tf.transform.rotation.z = odom.pose.pose.orientation.z
		tf.transform.rotation.w = odom.pose.pose.orientation.w
		# print("Publishing TF")
		self.tf2_pub.sendTransform(tf)


if __name__ == '__main__':
	# init node
	rospy.init_node('tf_republisher')
	# create tf broadcaster
	topic_name = rospy.get_param('odom_topic', '/odom')
	tf_republisher = TFRepublisher(topic_name)

	while not rospy.is_shutdown():
		rospy.spin()


