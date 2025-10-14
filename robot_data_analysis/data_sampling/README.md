# ROS2 Bag Processing Docker Container

This Docker container processes ROS2 bag files to extract synchronized image, odometry, point cloud, and tracking data for robotics datasets.

## Setup

### Prerequisites

- Docker and Docker Compose installed
- ROS2 bag files to process

### Build the Container

* Remember to update docker-compose.yml Line#10 to point to the volume containing your ros bags

```bash
docker-compose build
```

## Configuration

### Config File Structure

The `process_bags_config.yaml` file defines dataset-specific configurations. Each dataset entry contains:

- `odomtopics`: List of odometry topics (e.g., `["/odom"]`)
- `imtopics`: List of image topics (e.g., `["/image_raw/compressed"]`)
- `pcltopics`: List of point cloud topics (e.g., `["/velodyne_points"]`)
- `scantopics`: You can leave this an empty list
- `trackingtopics`: List of tracking topics
- `masktopics`: List of mask topics (if applicable)
- `speech_topic`: Speech topic for audio data
- Processing flags:
  - `process_pcl`: Enable/disable point cloud processing (Should be enabled)
  - `process_scan`: Enable/disable laser scan processing (leave as False)
  - `process_tracking`: Enable/disable tracking data processing (Should be enabled if there's a tracking topic available)
  - `process_masks`: Enable/disable mask processing (Should be enabled if there's a mask topic available)
- Processing functions for different data types
- `start_slack` and `end_slack`: In order to cut off parts of the start or end of a bag, add in the number of timesteps to cut here. For example, a start slack of 10 means the first 10 timesteps * sample-rate (default 4Hz) = 2.5 seconds will be skipped (similarly for the end_slack)

### Example Dataset Configuration

```yaml
scand_spot:
  odomtopics: ["/odom"]
  imtopics: ["/image_raw/compressed"]
  pcltopics: ["/velodyne_points"]
  scantopics: []
  trackingtopics: ["/tracks_image_raw_compressed"]
  process_pcl: true
  process_scan: false
  process_tracking: true
  process_masks: false
```

## Running the Container

### Using Docker Compose

1. **Update volume paths in `docker-compose.yml`**:

   ```yaml
   volumes:
      # Edit the left path to the root of your ros2 bags. its okay if they are in sub folders, the script will recursively search for bags.
     - /path/to/your/ros2bags:/ros2_bags:ro
      # Edit left path to where the results should be stored.
     - ./processed:/processed
   ```
2. **Start the container**:

   ```bash
   docker-compose up -d
   ```
3. **Execute the processing script**:

   ```bash
   docker-compose exec data_sampling attach
   python3 process_bags_ros2.py -d <dataset_name specified in process_bags_config.yaml>
   ```

### Direct Docker Run

```bash
docker run -it --rm \
  -v /path/to/rosbags:/ros2_bags:ro \
  -v $(pwd)/processed:/processed \
  -v $(pwd):/workspace \
  data_sampling_data_sampling \
  python3 process_bags_ros2.py -d <dataset_name specified in process_bags_config.yaml>
```

## Script Arguments

### Required Arguments

- `--dataset-name, -d`: Dataset name defined in config file (e.g., `scand_spot`, `miraikan`)

### Optional Arguments

- `--input-dir, -i`: Input directory containing bag files (default: `/ros2_bags`)
- `--input-bag, -b`: Process specific bag file instead of directory
- `--output-dir, -o`: Output directory for processed data (default: `/processed`)
- `--config-file, -c`: Config file path (default: `process_bags_config.yaml`)
- `--num-trajs, -n`: Maximum number of trajectories to process (default: -1 for all) (optional)
- `--sample-rate, -s`: Sampling rate in Hz (default: 4.0) LEAVE AS IS
- `--num-workers, -w`: Number of worker processes (default: 4)
- `--contin, -cont`: Continue processing, skip already processed bags (optional)
- `--file-list`: Process specific list of files (optional)

## Usage Examples

### Process all bags from scand_spot dataset

```bash
python3 process_bags_ros2.py --dataset-name scand_spot
```

### Process a specific bag file

```bash
python3 process_bags_ros2.py \
  --dataset-name miraikan \
  --input-bag /ros2_bags/specific_bag.bag \
  --sample-rate 2.0
```

### Process with custom output directory and continue mode

```bash
python3 process_bags_ros2.py \
  --dataset-name scand_spot \
  --output-dir /custom_output \
  --contin \
  --num-workers 8
```

### Process limited number of trajectories

```bash
python3 process_bags_ros2.py \
  --dataset-name miraikan \
  --num-trajs 10 \
  --sample-rate 1.0
```

## Output Structure

The script creates the following output structure:

```
processed/
├── trajectory_name_0/
│   ├── imgs/
│   ├── masks/
    ├── pcd/ 
    ├── traj_data.pkl
    ├── tracks.pkl
    ├── timestamps.txt
    ├── speech.txt
│ 
└── ...
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure Docker has access to input/output directories
2. **Memory issues**: Adjust `--num-workers` parameter for large datasets
3. **Missing topics**: Check topic names in config file match bag file topics if you see output like "\<bag name> didn't have the topics, Skipping..."

### Debug Mode

For debugging, run the container interactively:

```bash
docker-compose exec data_sampling /bin/bash
```

Then run the script with additional debugging options or explore the data structure.
