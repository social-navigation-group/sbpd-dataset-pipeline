"""
This script annotates images in multiple folders using a YOLO11 model with pedestrian bounding boxes.
It processes images in folders whose paths are specified in an input text file.
Command-line Arguments:
    --dataset (-d): Name of the dataset to process (required).
    --output_file (-o): Path to the output file to save results (required).
Usage:
    Run the script from the command line with the required arguments:
    python annotate_dataset_imgs.py --dataset <dataset_name> --output_file <output_path> --index <start_index>
"""
import os
from tqdm import tqdm
from ultralytics import YOLO
import argparse
import numpy as np
from PIL import Image
import pickle as pkl
model = YOLO('yolo11n.pt')

# Function to run YOLO on images in a folder
def run_yolo_on_folder(folder_path):
    results = {}
    folder_name = folder_path.split("/")[-1]
    folder_root = folder_path.replace(folder_name, "")
    img_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, img_name)) and img_name.lower().endswith(('.jpg', '.png'))]
    batch_size = 512
    yolo_results = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        images = [np.asarray(Image.open(img_path)) for img_path in batch_paths]
        yolo_results.extend(model(images,classes=0,verbose=False))
    
    for img_path, yolo_result in zip(img_paths, yolo_results):
        img_name = os.path.basename(img_path)
        if len(yolo_result.boxes)>0:
            results[img_name] = yolo_result.boxes
    with open(os.path.join(folder_path,"yolo_results.pkl"), 'wb') as f:
        pkl.dump(results, f)
    return results

# Main function to process all folders
def annotate_folders(args):
    try:
        with open(args.folder_paths) as file:
            folder_names = file.read().splitlines()
    except FileNotFoundError:
        print("Error: The specified file does not exist.")
        return
    
    dataset_results = {}
    for folder_name in tqdm(folder_names):
        if os.path.isdir(folder_name):
            dataset_results[folder_name] = run_yolo_on_folder(folder_name)
        with open(args.output_file, 'wb') as f:
            pkl.dump(dataset_results, f)
    return dataset_results

# Example usage
def main():
    parser = argparse.ArgumentParser(description='Annotate dataset images using YOLO.')
    parser.add_argument('--output_file',"-o", type=str, required=True, help='Path to the output file to save results.')
    parser.add_argument('--folder_paths',"-f", type=str, default="/media/dataset_access/datasets")    
    args = parser.parse_args()
    
    annotate_folders(args)

if __name__ == '__main__':
    main()