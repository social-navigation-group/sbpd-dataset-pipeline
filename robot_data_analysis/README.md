This directory contains all the scripts for running the analysis for the data collected from onboard the robot. 

### Parsing bag files

To extract basic info from the bag files including synchonized images, odometry and lidar point cloud, we use the setup from [ViNT](https://github.com/robodhruv/visualnav-transformer) codebase. The original codebase only supported ROS1 bag files (as used in SCAND and MuSoHu datasets). Processing Bag files is the first step before performing any other analysis.

We support both ROS1 and ROS2 bag format. To process a folder containing bag files from a dataset:

- bag_processing/process_bags_config.yaml contains the specification for processing each datatype for each dataset.
- process_bags_ros `<version>`.py can be run with the config file path specified to process ROS1 or ROS2 bag files.

Once the basic data from the bag files are extracted, we can extract additional features from the processed data (Note that this only works on unanonymized data. For the anonymized data, the people detection is provided directly):

1. Human 2D position Tracking in FPV RGB with Yolo:
2. Human 3D position + Vecloity Tracking in FPV with Yolo:
