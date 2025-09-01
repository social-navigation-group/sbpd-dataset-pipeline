import rosbag
import numpy as np
from datetime import datetime
import sys

#!/usr/bin/env python3

import matplotlib.pyplot as plt

def plot_rosbag_timestamps(bag_file_path):
    """
    Read a ROS bag file and plot timestamp differences for pointcloud and odom topics
    """
    # Lists to store timestamps
    pointcloud_timestamps = []
    odom_timestamps = []
    
    # Open the bag file
    with rosbag.Bag(bag_file_path, 'r') as bag:
        # Get bag info
        info = bag.get_type_and_topic_info()
        topics = info.topics.keys()
        
        print("Available topics:")
        for topic in topics:
            print(f"  {topic}")
        
        # Read messages and extract timestamps
        for topic, msg, t in bag.read_messages():
            timestamp = t.to_sec()
            
            # Check for pointcloud topics (common names)
            if any(keyword in topic.lower() for keyword in ['pointcloud', 'velodyne', 'lidar', 'scan']):
                pointcloud_timestamps.append(timestamp)
            
            # Check for odometry topics
            elif any(keyword in topic.lower() for keyword in ['odom', 'odometry']):
                odom_timestamps.append(timestamp)
    
    # Convert to numpy arrays and calculate differences
    pointcloud_diffs = []
    odom_diffs = []
    
    if pointcloud_timestamps:
        pointcloud_timestamps = np.array(pointcloud_timestamps)
        pointcloud_diffs = np.diff(pointcloud_timestamps)
    
    if odom_timestamps:
        odom_timestamps = np.array(odom_timestamps)
        odom_diffs = np.diff(odom_timestamps)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    if len(pointcloud_diffs) > 0:
        plt.subplot(2, 1, 1)
        plt.plot(pointcloud_diffs, 'bo-', markersize=3, linewidth=1)
        plt.title('PointCloud Timestamp Differences')
        plt.xlabel('Message Index')
        plt.ylabel('Time Difference (seconds)')
        plt.grid(True)
        
        # Calculate and display statistics
        mean_diff = np.mean(pointcloud_diffs)
        std_diff = np.std(pointcloud_diffs)
        plt.text(0.02, 0.98, f'Mean: {mean_diff:.4f}s\nStd: {std_diff:.4f}s\nFreq: {1/mean_diff:.2f} Hz', 
                transform=plt.gca().transAxes, verticalalignment='top')
    
    if len(odom_diffs) > 0:
        plt.subplot(2, 1, 2)
        plt.plot(odom_diffs, 'ro-', markersize=3, linewidth=1)
        plt.title('Odometry Timestamp Differences')
        plt.xlabel('Message Index')
        plt.ylabel('Time Difference (seconds)')
        plt.grid(True)
        
        # Calculate and display statistics
        mean_diff = np.mean(odom_diffs)
        std_diff = np.std(odom_diffs)
        plt.text(0.02, 0.98, f'Mean: {mean_diff:.4f}s\nStd: {std_diff:.4f}s\nFreq: {1/mean_diff:.2f} Hz', 
                transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("rosbag_timestamp_differences.png")
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"PointCloud messages: {len(pointcloud_timestamps)}")
    print(f"Odometry messages: {len(odom_timestamps)}")
    if len(pointcloud_diffs) > 0:
        print(f"PointCloud avg time diff: {np.mean(pointcloud_diffs):.4f}s")
    if len(odom_diffs) > 0:
        print(f"Odometry avg time diff: {np.mean(odom_diffs):.4f}s")

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python check_rosbag_timestamps.py <path_to_bagfile>")
        sys.exit(1)
    
    bag_file = sys.argv[1]
    plot_rosbag_timestamps(bag_file)