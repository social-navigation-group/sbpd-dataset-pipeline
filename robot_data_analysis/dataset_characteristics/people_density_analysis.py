import os
import pickle
import numpy as np
import argparse
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from IPython import embed
from shapely.geometry import box
from shapely.ops import unary_union
import cv2
from tqdm import tqdm
from itertools import cycle
from matplotlib.gridspec import GridSpec
SAMPLE_RATE = 4 #rate at which the bag was processed

def extract_people_2d(dataset_path,dataset_name,d3=False):
    """
    Iterates through folders in the dataset path, processes 'tracks.pkl' and 'pedestrian_3d.pkl' files,
    and calculates statistics of the number of people across the dataset.
    """
    #total_people_counts = []
    #proportion_area_covered = []
    total_duration = 0
    distance_of_pedestrians_from_robot = []
    closest_pedestrian_distance = np.inf
    farthest_pedestrian_distance = -np.inf
    ped_count = {}
    proportion_area_covered = {}
    # Iterate through all subdirectories in the dataset path    
    for trajectory in tqdm(os.listdir(dataset_path)): 
        ped_count[trajectory] = []
        proportion_area_covered[trajectory] = []
        #get the image size
        imgs = os.listdir(os.path.join(dataset_path,trajectory,'imgs'))
        assert len(imgs)>0, f"Error: No images found in {os.path.join(dataset_path,trajectory,'imgs')}"
        img = cv2.imread(os.path.join(dataset_path,trajectory,'imgs',imgs[0]))
        h,w = img.shape[:2]
        
        tracks_file_path = os.path.join(dataset_path, trajectory, 'tracks.pkl')
        ped_3d_file_path = os.path.join(dataset_path, trajectory, 'pedestrians_3d.pkl')
        if os.path.isfile(tracks_file_path):
            
            if d3: 
                if not os.path.isfile(ped_3d_file_path):
                    print(f"Warning: No pedestrians_3d.pkl file found in {ped_3d_file_path}.")
                    continue
                with open(ped_3d_file_path, 'rb') as f:
                    pedestrians_3d = pickle.load(f)
            
            
            with open(tracks_file_path, 'rb') as f:
                tracks = pickle.load(f)
            
            
            for timestep in range(len(tracks)):
                pedestrian_screen_area = 0
                people_count = 0
                if tracks[timestep][0] is not None and len(tracks[timestep][0]) > 0:
                    detections = np.array(tracks[timestep][0])
                    try:
                        # detections: [x, y, w, h] for each detection
                        boxes = [box(max(0,det[0]), max(0,det[1]), min(det[0] + det[2],w), min(det[1] + det[3],h)) for det in detections]
                        if boxes:
                            union_area = unary_union(boxes).area
                        else:
                            union_area = 0
                        pedestrian_screen_area = union_area / (h*w)
                        
                        if pedestrian_screen_area>1.0:
                            embed()
                    except IndexError:
                        print(tracks[timestep][0])
                        print(f"Error: IndexError in timestep {timestep} of trajectory {trajectory}.")
                        exit(0)
                    people_count = detections.shape[0]
                #else:
                #    print(f"Warning: No detections found in timestep {timestep} of trajectory {trajectory}.")
            
                ped_count[trajectory].append(people_count)
                
                proportion_area_covered[trajectory].append(pedestrian_screen_area)  # Proportion of the frame covered by people
                #total_people_counts.append(people_count)
                if d3:
                    if timestep in pedestrians_3d:
                        ped_count_3d = np.array([peds[0] for peds in pedestrians_3d[timestep]]).reshape(-1,3)
                        distance_of_pedestrians_from_robot= np.append(distance_of_pedestrians_from_robot,
                                                                    np.linalg.norm(ped_count_3d,axis =1))
                    else: 
                        print(f"Warning: No 3D pedestrians found in timestep {timestep} of trajectory {trajectory}.")
                        
                total_duration+=1
    return dict(
        total_duration = total_duration,
        proportion_area_covered = proportion_area_covered,
        pedestrian_count = ped_count,
    )
    
if __name__ == "__main__":
    
    datasets = {
        "scand_spot": "/media/shashank/T/processed/scand_spot",
        "scand_jackal": "/media/shashank/T/processed/scand_jackal",
        "go2nus": "/media/shashank/T/processed/go2nus",
        "musohu": "/media/shashank/T/processed/musohu",
    }
    total_people_counts = {}
    proportion_area_covered = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"Processing dataset: {dataset_name}")
        # stats = extract_people_2d(dataset_path, dataset_name, False)
        # with open(f'people_stats_{dataset_name}.pkl', 'wb') as f:
        #     pickle.dump(stats, f)
        with open(f'rgb_results/people_stats_{dataset_name}.pkl', 'rb') as f:
            stats = pickle.load(f)
        
        total_duration = stats['total_duration']
        proportion_area_covered_per_trajectory = stats['proportion_area_covered']
        ped_count_per_trajectory = stats['pedestrian_count']
        
        total_people_counts[dataset_name] = []
        for k,v in ped_count_per_trajectory.items():
            total_people_counts[dataset_name].extend(v)
        total_people_counts[dataset_name] = np.array(total_people_counts[dataset_name])
        
        proportion_area_covered[dataset_name] = []
        for k,v in proportion_area_covered_per_trajectory.items():
            proportion_area_covered[dataset_name].extend(v)
        proportion_area_covered[dataset_name] = np.array(proportion_area_covered[dataset_name])
            
        print(f"=====Dataset {dataset_name} statistics=====")
        print(f"Total duration (at 4hz sample rate):{total_duration/(SAMPLE_RATE*3600)} hours") #frames/fps
        print(f"""Duration with no (detected) people:{np.sum(total_people_counts[dataset_name] == 0) / len(total_people_counts[dataset_name]) * 100}%""")
        print(f"Average number of people per timestep:{total_people_counts[dataset_name].mean()}")
        print(f"Average proportion of the frame covered by people: {proportion_area_covered[dataset_name].mean()*100.0}%")
        #print(stats)
    
    # find 2 timsteps in the go2nus dataset with 10 people detected and very different proportion of area covered
    # Find 2 timesteps in go2nus dataset with 10 people and very different area coverage
    go2nus_stats = {}
    with open(f'rgb_results/people_stats_go2nus.pkl', 'rb') as f:
        go2nus_stats = pickle.load(f)

    ped_count_per_traj = go2nus_stats['pedestrian_count']
    area_covered_per_traj = go2nus_stats['proportion_area_covered']

    timesteps_with_10_people = []
    for traj_name, counts in ped_count_per_traj.items():
        for timestep, count in enumerate(counts):
            if count == 8:
                area_coverage = area_covered_per_traj[traj_name][timestep]
                timesteps_with_10_people.append((traj_name, timestep, area_coverage))

    if len(timesteps_with_10_people) >= 2:
        # Sort by area coverage to find most different
        timesteps_with_10_people.sort(key=lambda x: x[2])
        min_coverage = timesteps_with_10_people[0]
        max_coverage = timesteps_with_10_people[-1]
        
        print(f"Timestep with 10 people and minimum area coverage:")
        print(f"  Trajectory: {min_coverage[0]}, Timestep: {min_coverage[1]}, Area coverage: {min_coverage[2]:.4f}")
        print(f"Timestep with 10 people and maximum area coverage:")
        print(f"  Trajectory: {max_coverage[0]}, Timestep: {max_coverage[1]}, Area coverage: {max_coverage[2]:.4f}")
    else:
        print(f"Found only {len(timesteps_with_10_people)} timesteps with exactly 10 people")
    
    
        
    #combine scand_spot and scand_jackal datasets
    total_people_counts['scand'] = np.concatenate(
        [total_people_counts['scand_spot'], total_people_counts['scand_jackal']]
    )
    proportion_area_covered['scand'] = np.concatenate(
        [proportion_area_covered['scand_spot'], proportion_area_covered['scand_jackal']]
    )
    # Remove individual datasets
    del total_people_counts['scand_spot']
    del total_people_counts['scand_jackal']
    del proportion_area_covered['scand_spot']
    del proportion_area_covered['scand_jackal']

    # Plot combined stats with 5 subplots: 
    # 1. Pedestrian counts histogram/density
    # 2. Proportion area covered density

    fig = plt.figure(figsize=(20, 10))
    axs = fig.subplots(1, 2)

    # Assign a color to each dataset for consistency
    colors = plt.cm.tab10.colors
    dataset_names = list(total_people_counts.keys())
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(dataset_names)}
    
    # 1. Pedestrian counts histogram - stacked
    max_count = max([counts.max() for counts in total_people_counts.values()])
    bins = np.arange(0, max_count + 2)

    # Create histograms for each dataset
    for dataset_name, counts in total_people_counts.items():
        hist, _ = np.histogram(counts, bins=bins, density=True)
        hist_percent = hist * 100  # Convert to percentage
        axs[0].step(bins[:-1], hist_percent, where='post', lw=2, 
                    label=f"{dataset_name}", color=color_map[dataset_name])
        
        # Add arrows and text for values that exceed ylim
        ylim_max = 20
        for i, (bin_start, value) in enumerate(zip(bins[:-1], hist_percent)):
            if value > ylim_max:
                # Add arrow pointing up at the top of the bin
                axs[0].annotate(f'{value:.1f}', 
                               xy=(bin_start, ylim_max), 
                               xytext=(bin_start, ylim_max - 2),
                               ha='center', va='top',
                               fontsize=10,
                               color=color_map[dataset_name],
                               arrowprops=dict(arrowstyle='->', 
                                             color=color_map[dataset_name],
                                             lw=1.5))
    
    axs[0].set_ylabel("% of Dataset", fontsize=16)
    axs[0].set_title("Number of Pedestrian Detections in Image", fontsize=18)
    axs[0].legend(fontsize=13)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].set_xlabel("(a)", fontsize=16)
    axs[0].set_ylim(0,20)

    # 2. Proportion area covered density (spanning last column of top row)
    max_prop = max([props.max() for props in proportion_area_covered.values()])
    x_vals_prop = np.linspace(0, max_prop, 200)
    for dataset_name, props in proportion_area_covered.items():
        if len(props) > 1:
            density = gaussian_kde(props)
            density_values = density(x_vals_prop)
            axs[1].plot(
                x_vals_prop, density_values, lw=1.5, label=f"{dataset_name}", 
                color=color_map[dataset_name]
            )
            
            # Add arrows and text for values that exceed ylim
            ylim_max = 20
            for i, (x_val, y_val) in enumerate(zip(x_vals_prop, density_values)):
                if y_val > ylim_max and x_val == 0:
                    # Add arrow pointing up at the peak
                    axs[1].annotate(f'{y_val:.1f}', 
                                   xy=(x_val, ylim_max), 
                                   xytext=(x_val -0.01, ylim_max - 2),
                                   ha='center', va='top',
                                   fontsize=10,
                                   color=color_map[dataset_name],
                                   arrowprops=dict(arrowstyle='->', 
                                                 color=color_map[dataset_name],
                                                 lw=1.5))
    #axs[1].set_xlabel("Proportion of frame covered by people", fontsize=16)
    axs[1].set_ylabel("% of Dataset", fontsize=16)
    axs[1].set_title("Proportion of Image Area covered by Detected Pedestrians", fontsize=18)
    axs[1].legend(fontsize=13)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].set_xlabel("(b)", fontsize=16)
    axs[1].set_ylim(0,20)
    plt.savefig('combined_trimmed_people_density_analysis.svg', bbox_inches='tight', dpi=300)
    # plt.show()    