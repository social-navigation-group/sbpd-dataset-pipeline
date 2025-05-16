# Use Ubuntu 22.04 as the base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install prerequisites including python3-pip
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    python3-pip \
    wget \
    apt-utils \
    git-all

# Add the ROS2 apt repository key and repository
RUN curl -sSL http://repo.ros2.org/repos.key | apt-key add - \
    && echo "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu/main $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

# Update and install ROS2 Humble Desktop (adjust if needed)
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosbag2-py \
    ros-humble-vision-msgs

# Copy the requirements file from your bag_process folder into the image
COPY requirements.txt /tmp/requirements.txt

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Create a workspace directory
RUN mkdir -p /bag_ws
WORKDIR /bag_ws

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install -r /tmp/requirements.txt

# Install Detectron2
RUN git clone https://github.com/allanwangliqian/detectron2.git
RUN pip3 install -e detectron2

# Install ByteTrack
RUN git clone https://github.com/allanwangliqian/ByteTrack.git
WORKDIR /bag_ws/ByteTrack
RUN pip3 install -r requirements.txt
RUN python3 setup.py develop
RUN pip3 install cython cython_bbox gdown

# Download ByteTrack pretrained model
RUN mkdir -p /bag_ws/ByteTrack/pretrained
RUN gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O /bag_ws/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar

WORKDIR /bag_ws

# RUN rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script into the container and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the default entrypoint to source the ROS2 environment
ENTRYPOINT ["/entrypoint.sh"]

# Default command to run bash
CMD ["bash"]

