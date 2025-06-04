FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies in one layer and clean up afterwards.
RUN apt update && \
    apt install -y --no-install-recommends \
      curl \
      gnupg2 \
      lsb-release \
      build-essential \
      python3-pip \
      wget \
      apt-utils \
      software-properties-common \
      git-all && \
    add-apt-repository universe && \
    rm -rf /var/lib/apt/lists/*

# Add the ROS2 apt repository key and repository
# RUN curl -sSL http://repo.ros2.org/repos.key | apt-key add - \
#    && echo "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu/main $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    apt install /tmp/ros2-apt-source.deb

# Update and install ROS2 Humble Desktop (adjust if needed)
RUN apt update && apt-get install -y \
        ros-humble-desktop \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-rosbag2-py \
        ros-humble-vision-msgs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements, create workspace and install Python dependencies in one layer.
COPY requirements.txt /tmp/requirements.txt

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN mkdir -p /bag_ws && \
    cd /bag_ws && \
    pip3 install --upgrade pip && \
    pip3 install --ignore-installed torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip

WORKDIR /bag_ws

# Install ByteTrack in one layer and remove its build files when done.
RUN git clone https://github.com/allanwangliqian/ByteTrack.git && \
    cd ByteTrack && \
    pip3 install -r requirements.txt && \
    python3 setup.py develop && \
    pip3 install cython cython_bbox pycocotools gdown && \
    mkdir -p pretrained && \
    gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O pretrained/bytetrack_x_mot17.pth.tar && \
    cd ..
    # rm -rf ByteTrack

WORKDIR /bag_ws

# Copy the entrypoint script and make it executable.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]