#!/bin/bash

# Setup script for move_base_experiments Docker environment
# This script helps your teammate get started quickly

set -e

echo "🚀 Setting up Move Base Experiments Docker Environment"
echo "======================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create output directory
echo "📁 Creating output directory..."
mkdir -p output

echo "✅ Directory created:"
echo "   - output/   (experiment results will be saved here)"
echo ""
echo "📝 Note: You'll need to set ROSBAG_DIR environment variable or"
echo "    edit docker-compose.yml to point to your ROS2 rosbag directory"

# Build the Docker image
echo "🔨 Building Docker image (this may take a few minutes)..."
if docker-compose build; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set ROSBAG_DIR environment variable: export ROSBAG_DIR=/path/to/your/ros2/rosbags"
echo "2. Run: docker-compose up"
echo "3. In the container, run: ./run.sh /rosbags /ros1_bags /output [odom_topic] [lidar_topic] [viz]"
echo ""
echo "Example with custom topics:"
echo "   ./run.sh /rosbags /ros1_bags /output /your/odom/topic /your/lidar/topic true"
echo ""
echo "For more details, see README.md"