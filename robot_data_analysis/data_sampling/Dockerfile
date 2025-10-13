FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install additional ROS2 dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    ros-humble-nav2-common \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-tf2-tools \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-laser-geometry \
    ros-humble-message-filters \
    ros-humble-pcl-conversions \
    ros-humble-pcl-ros \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs \
    ros-humble-velodyne-msgs \
    ros-humble-rosbag2-py \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rclpy \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for process_bags_ros2.py
RUN pip3 install \
    rosbags \
    termcolor \
    scipy \
    numpy \
    tqdm \
    pyyaml \
    opencv-python \
    pillow \
    open3d \
    pypcd \
    ros2-numpy \
    velodyne-decoder \
    ipython

# Initialize rosdep (if not already done)
RUN rosdep init || true
RUN rosdep update

# Update package cache and install missing dependencies
# RUN apt-get update && apt-get install -y libnetpbm10-dev netpbm libnetpbm-dev

# Set up ROS2 environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create directories for rosbag data and output
RUN mkdir -p /rosbags /output /workspace

# Set working directory for the bag processing scripts
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]