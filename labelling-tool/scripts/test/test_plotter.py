import os
from utils.logging_utils import log_info
from .trajectory_plotter import TrajectoryPlotter

resources_path = os.path.join(os.path.dirname(__file__), "..", "resources")
video_path = os.path.join(resources_path, "videos", "example_raw.avi") 
save_path = os.path.join(resources_path, "output", "test_output.avi")  
human_config_path = os.path.join(resources_path, "config", "human_config.toml")

output_dir = os.path.dirname(save_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(human_config_path):
    log_info("Human config file not found! Test cannot proceed.")
    exit(1)

import toml
human_config = toml.load(human_config_path)

# Run the trajectory plotter
log_info("Running trajectory plotter for testing...")
TrajectoryPlotter.plot_trajectories(video_path, save_path, human_config)

log_info(f"Output saved to {save_path}")
