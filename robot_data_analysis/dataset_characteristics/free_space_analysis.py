import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
SAMPLE_RATE = 4
DATASET_CONFIG = {
    "go2nus":{
    }
}
def calculate_free_space_statistics(dataset_path,dataset_name = "go2nus"):
    """
    Iterates through folders in the dataset path, processes 'tracks.pkl' files,
    and calculates statistics of the number of people across the dataset.
    """
    proportion_occupied_space = []
    total_duration = 0
    # Iterate through all subdirectories in the dataset path
    for root, dirs, files in os.walk(dataset_path): 
        for trajectory in dirs:
            tracks_file_path = os.path.join(root, trajectory, 'tracks.pkl')
            # Check if 'tracks.pkl' exists in the folder
            if os.path.isfile(tracks_file_path):
                # for each trajectory:
                with open(tracks_file_path, 'rb') as f:
                    tracks = pickle.load(f)
                
                # Extract statistics from each timestep
                for timestep in range(len(tracks)):
                    pedestrian_screen_area = 0
                    people_count = 0
                    
                    if tracks[timestep][0] is not None:
                        detections = np.array(tracks[timestep][0])
                        pedestrian_screen_area = (detections[:,2] * detections[:,3]).sum()/(DATASET_CONFIG[dataset_name]['image_size'][0] * DATASET_CONFIG[dataset_name]['image_size'][1])
                        people_count = detections.shape[0]
                        
                    proportion_area_covered.append(pedestrian_screen_area)  # Proportion of the frame covered by people
                    total_people_counts.append(people_count)
                    total_duration+=1
    proportion_area_covered = np.array(proportion_area_covered)
    total_people_counts = np.array(total_people_counts)
    
    print(f"=====Dataset {dataset_name} statistics=====")
    x = np.arange(0, np.max(total_people_counts) + 1)
    y = np.bincount(total_people_counts)*100.0/len(total_people_counts)
    fig = plt.figure()
    bars = plt.bar(x,y)
    plt.xlabel("Number of people (detected) in the frame")
    plt.ylabel(""" % of dataset """)
    plt.title("Distribution of Pedestrian Counts per Frame")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # for bar, count in zip(bars, y):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{count/(4*60):4f}minutes", 
    #             ha='center', va='bottom', fontsize=10)
    plt.savefig(f"people_density_analysis_{dataset_name}.png")  
    print(f"Total duration (at 4hz sample rate):{total_duration/(SAMPLE_RATE*3600)} hours") #frames/fps
    print(f"""Duration with no (detected) people:{np.sum(total_people_counts == 0) / len(total_people_counts) * 100}%""")
    print(f"Average number of people per timestep:{total_people_counts.mean()}")
    print(f"Average proportion of the frame covered by people: {proportion_area_covered.mean()*100.0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--dataset-dir",
        "-i",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_dir
    if not os.path.isdir(dataset_path):
        print("Invalid dataset path. Please provide a valid directory.")
    else:
        stats = calculate_people_statistics(dataset_path)
        print(stats)