FROM osrf/ros:noetic-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-catkin-tools \
    python3-pip \
    ros-noetic-navigation \
    ros-noetic-move-base \
    ros-noetic-tf \
    ros-noetic-laser-geometry \
    ros-noetic-message-filters \
    ros-noetic-nodelet \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    ros-noetic-costmap-2d \
    ros-noetic-nav-core \
    ros-noetic-nav-msgs \
    ros-noetic-pluginlib \
    ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install rosbags-convert for ROS2 to ROS1 conversion
RUN pip3 install rosbags termcolor scipy numpy tqdm

# Create catkin workspace
WORKDIR /catkin_ws

# Initialize rosdep (if not already done)
RUN rosdep init || true
RUN rosdep update

# Update package cache and install missing dependencies
RUN apt-get update && apt-get install -y libnetpbm10-dev || apt-get install -y netpbm libnetpbm-dev

# Note: Dependencies and build will be handled at runtime since src/ is volume mounted
WORKDIR /catkin_ws
# Set up environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# Create directories for rosbag data and output
RUN mkdir -p /rosbags /output /ros1_bags

# Note: run.sh will be available through volume mount

# Set working directory

# Create src directory for catkin workspace
RUN mkdir -p /catkin_ws/src

# Default command
WORKDIR /catkin_ws
CMD ["/bin/bash"]