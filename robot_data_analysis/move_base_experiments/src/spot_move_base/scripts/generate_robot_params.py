#!/usr/bin/env python3
"""
Generates move_base parameter files from robot configuration.
Usage: python generate_robot_params.py <robot_name>
"""

import sys
import os
import yaml
from pathlib import Path


def load_robot_config(robot_name):
    """Load robot configuration from YAML file."""
    script_dir = '/catkin_ws/src/'
    config_path = Path(script_dir+"lidar_rosbag_parser/config/"+f"{robot_name}.yaml")

    if not config_path.exists():
        print(f"Error: Robot config file not found: {config_path}")
        print(f"Available configs:")
        config_dir = config_path.parent
        if config_dir.exists():
            for f in sorted(config_dir.glob("*.yaml")):
                print(f"  - {f.stem}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_costmap_params(robot_config, output_path):
    """Generate costmap_common_params.yaml with robot footprint."""

    # Extract footprint from config, or use a default based on robot dimensions
    if 'footprint' in robot_config:
        footprint = robot_config['footprint']
    else:
        print("Warning: 'footprint' not found in config, using default")
        footprint = [[-0.35, -0.165], [-0.35, 0.165], [0.35, 0.165], [0.35, -0.165]]

    params = {
        'map_type': 'costmap',
        'origin_z': 0.0,
        'z_resolution': 1,
        'z_voxels': 2,
        'cost_scaling_factor': 60,
        'inflation_radius': 10.0,
        'obstacle_range': 10.0,
        'raytrace_range': 12.5,
        'publish_voxel_map': False,
        'transform_tolerance': 0.5,
        'meter_scoring': True,
        'footprint': footprint,
        'footprint_padding': 0.1,
        'plugins': [
            {'name': 'obstacles_layer', 'type': 'costmap_2d::ObstacleLayer'},
            {'name': 'inflation_layer', 'type': 'costmap_2d::InflationLayer'}
        ],
        'obstacles_layer': {
            'observation_sources': 'scan',
            'scan': {
                'sensor_frame': 'lidar_link_remapped',
                'data_type': 'LaserScan',
                'topic': 'scan',
                'marking': True,
                'clearing': True,
                'min_obstacle_height': -100000.0,
                'max_obstacle_height': 100000.0,
                'obstacle_range': 10.0,
                'raytrace_range': 12.5
            }
        },
        'inflater_layer': {
            'inflation_radius': 0.3
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Generated: {output_path}")


def generate_planner_params(robot_config, output_path):
    """Generate base_local_planner_params.yaml with velocity/acceleration limits."""

    # Extract parameters from config with defaults
    params = {
        'TrajectoryPlannerROS': {
            'max_vel_x': robot_config.get('max_vel_x', 1.6),
            'min_vel_x': robot_config.get('min_vel_x', 0.2),
            'max_vel_theta': robot_config.get('max_vel_theta', 1.57),
            'min_in_place_vel_theta': robot_config.get('min_in_place_vel_theta', 0.314),
            'acc_lim_theta': robot_config.get('acc_lim_theta', 20.0),
            'acc_lim_x': robot_config.get('acc_lim_x', 2.5),
            'acc_lim_y': robot_config.get('acc_lim_y', 2.5),
            'holonomic_robot': robot_config.get('holonomic_robot', False)
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Generated: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_robot_params.py <robot_name>")
        print("Example: python generate_robot_params.py spot")
        sys.exit(1)

    robot_name = sys.argv[1]

    print(f"🤖 Generating move_base parameters for robot: {robot_name}")

    # Load robot configuration
    robot_config = load_robot_config(robot_name)
    print(f"📄 Loaded config from: lidar_rosbag_parser/config/{robot_name}.yaml")

    # Output directory
    script_dir = Path(__file__).parent.parent
    params_dir = script_dir / "params"
    params_dir.mkdir(exist_ok=True)

    # Generate parameter files
    costmap_path = params_dir / "costmap_common_params.yaml"
    planner_path = params_dir / "base_local_planner_params.yaml"

    generate_costmap_params(robot_config, costmap_path)
    generate_planner_params(robot_config, planner_path)

    print(f"✨ Successfully generated move_base parameters for {robot_name}")


if __name__ == "__main__":
    main()
