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
        x = np.arange(0, np.max(total_people_counts[dataset_name]) + 1)
        y1 = np.bincount(total_people_counts[dataset_name]) * 100.0 / len(total_people_counts[dataset_name])
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Left subplot: Number of people per frame
        bars = axs[0].bar(x, y1)
        axs[0].set_xlabel("Number of people (detected) in the frame")
        axs[0].set_ylabel("% of dataset")
        axs[0].set_title("Distribution of Pedestrian Counts per Frame")
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Right subplot: Proportion of frame covered by people
        # Plot as a line graph (density curve) instead of histogram

        density = gaussian_kde(proportion_area_covered[dataset_name])
        xs = np.linspace(0, proportion_area_covered[dataset_name].max(), 200)
        axs[1].plot(xs, density(xs), color='green', lw=2)
        axs[1].set_xlabel("Proportion of frame covered by people")
        axs[1].set_ylabel("% of dataset")
        axs[1].set_title("Distribution of Proportion Area Covered")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"people_density_2d_analysis_{dataset_name}.png")
        
        # if d3:
        #     fig,axs = plt.subplots(1, 1, figsize=(14, 6))
        #     bins = np.linspace(0, np.max(distance_of_pedestrians_from_robot), 30)
        #     axs.hist(distance_of_pedestrians_from_robot, bins=bins, color='orange', alpha=0.7, edgecolor='black')
        #     axs.set_xlabel("Distance of pedestrians from robot (meters)")
        #     axs.set_ylabel("Count")
        #     axs.set_title("Distribution of Pedestrian Distances")
        #     axs.grid(axis='y', linestyle='--', alpha=0.7)
        #     plt.tight_layout()
        #     plt.savefig(f"people_distance_3d_analysis_{dataset_name}.png")
        
        print(f"Total duration (at 4hz sample rate):{total_duration/(SAMPLE_RATE*3600)} hours") #frames/fps
        print(f"""Duration with no (detected) people:{np.sum(total_people_counts[dataset_name] == 0) / len(total_people_counts[dataset_name]) * 100}%""")
        print(f"Average number of people per timestep:{total_people_counts[dataset_name].mean()}")
        print(f"Average proportion of the frame covered by people: {proportion_area_covered[dataset_name].mean()*100.0}%")
        #print(stats)
    
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
    # 3. Average number of people
    # 4. Average proportion area covered
    # 5. Proportion of frames with no people detected

    # Make the first row much bigger and the bars wider for readability

    fig = plt.figure(figsize=(30, 20))
    gs = GridSpec(2, 3, height_ratios=[2, 1], hspace=0.2, wspace=0.25)

    # Top row: two wide subplots
    ax_top0 = fig.add_subplot(gs[0, :2])
    ax_top1 = fig.add_subplot(gs[0, 2])

    # Assign a color to each dataset for consistency
    colors = plt.cm.tab10.colors
    dataset_names = list(total_people_counts.keys())
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(dataset_names)}

    # 1. Pedestrian counts histogram (spanning first two columns of top row)
    bins = np.arange(0, max([counts.max() for counts in total_people_counts.values()]) + 2) - 0.5
    width = 0.7 / max(1, len(total_people_counts))  # wider bars
    for i, (dataset_name, counts) in enumerate(total_people_counts.items()):
        hist, _ = np.histogram(counts, bins=bins, density=True)
        ax_top0.bar(
            bins[:-1] + i * width, hist*100.0, width=width, alpha=0.85,
            label=f"{dataset_name}", edgecolor='black', color=color_map[dataset_name]
        )
    #ax_top0.set_xlabel("Number of people (detected) in the frame", fontsize=16)
    ax_top0.set_ylabel("% of Dataset", fontsize=16)
    ax_top0.set_title("Number of Pedestrian Detections in Image", fontsize=18)
    ax_top0.legend(fontsize=13)
    ax_top0.grid(axis='y', linestyle='--', alpha=0.7)
    ax_top0.tick_params(axis='both', labelsize=14)
    ax_top0.set_xlabel("(a)", fontsize=16)

    # 2. Proportion area covered density (spanning last column of top row)
    max_prop = max([props.max() for props in proportion_area_covered.values()])
    x_vals_prop = np.linspace(0, max_prop, 200)
    for dataset_name, props in proportion_area_covered.items():
        if len(props) > 1:
            density = gaussian_kde(props)
            ax_top1.plot(
                x_vals_prop, density(x_vals_prop), lw=1.5, label=f"{dataset_name}", 
                color=color_map[dataset_name]
            )
    #ax_top1.set_xlabel("Proportion of frame covered by people", fontsize=16)
    ax_top1.set_ylabel("% of Dataset", fontsize=16)
    ax_top1.set_title("Proportion of Image Area covered by Detected Pedestrians", fontsize=18)
    ax_top1.legend(fontsize=13)
    ax_top1.grid(axis='y', linestyle='--', alpha=0.7)
    ax_top1.tick_params(axis='both', labelsize=14)
    ax_top1.set_xlabel("(b)", fontsize=16)
    # Bottom row: 3 plots
    axs = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # 3. Average number of people
    avg_people = [counts.mean() for counts in total_people_counts.values()]
    std_people = [counts.std() for counts in total_people_counts.values()]
    axs[0].bar(
        dataset_names, avg_people, color=[color_map[name] for name in dataset_names], width=0.7,
        yerr=std_people, capsize=5
    )
    axs[0].set_ylabel("Average # people", fontsize=14)
    axs[0].set_title("Average Number of Pedestrian Detections per Image frame", fontsize=15)
    for i, p in enumerate(avg_people):
        axs[0].text(i, p + 0.05, f"{p:.2f}", ha='center', va='bottom', fontsize=12)
    axs[0].tick_params(axis='x', labelsize=13)
    axs[0].tick_params(axis='y', labelsize=13)
    axs[0].set_ylim(top=max(avg_people) + 1)  # Increase top limit for more space above bars/text
    axs[0].set_xlabel("(c)", fontsize=16)
    # 4. Average proportion area covered
    avg_prop = [props.mean() * 100 for props in proportion_area_covered.values()]
    std_prop = [props.std() * 100 for props in proportion_area_covered.values()]
    axs[1].bar(
        dataset_names, avg_prop, color=[color_map[name] for name in dataset_names], width=0.7,
        yerr=std_prop, capsize=5
    )
    axs[1].set_ylabel("Average % area covered", fontsize=14)
    axs[1].set_title("Average Proportion of Image Area Covered by Detected Pedestrians", fontsize=15)
    for i, a in enumerate(avg_prop):
        axs[1].text(i, a + 0.05, f"{a:.2f}%", ha='center', va='bottom', fontsize=12)
    axs[1].tick_params(axis='x', labelsize=13)
    axs[1].tick_params(axis='y', labelsize=13)
    axs[1].set_ylim(top=max(avg_prop) + 3)  # Increase top limit for more space above bars/text
    axs[1].set_xlabel("(d)", fontsize=16)
    # 5. Proportion of frames with no people detected
    no_people_prop = [
        (counts == 0).sum() / len(counts) * 100 for counts in total_people_counts.values()
    ]
    axs[2].bar(
        dataset_names, no_people_prop, color=[color_map[name] for name in dataset_names], width=0.7
    )
    axs[2].set_ylabel("% frames with no people", fontsize=14)
    axs[2].set_title("Proportion of Dataset with No People Detected in Image Frame", fontsize=15)
    for i, v in enumerate(no_people_prop):
        axs[2].text(i, v + 0.5, f"{v:.2f}%", ha='center', va='bottom', fontsize=12)
    axs[2].tick_params(axis='x', labelsize=13)
    axs[2].tick_params(axis='y', labelsize=13)
    axs[2].set_ylim(top=max(no_people_prop) + 5)  # Increase top limit for more space above bars/text
    axs[2].set_xlabel("(e)", fontsize=16)
    plt.tight_layout()
    # plt.axis('off')
    plt.savefig("combined_people_density_2d_analysis.svg", format='svg', bbox_inches='tight')
    plt.show()    