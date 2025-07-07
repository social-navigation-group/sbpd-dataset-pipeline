#!/usr/bin/env python3
"""
Convert YOLO JSON annotations into per-image text files.

This script takes a JSON file exported from VIA containing YOLO format annotations
and converts it into individual text files for each image, as required by YOLOv8.

Usage:
    python yolo_json_to_txt.py input.json output_dir

Example:
    python yolo_json_to_txt.py yolo_annotations.json ./labels
"""

import json
import os
import sys
import argparse

def convert_yolo_json(input_file, output_dir):
    """
    Convert YOLO JSON annotations into per-image text files.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the output text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Save classes.txt
    classes_file = os.path.join(output_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('\n'.join(data['classes']))
    print(f"Saved classes to {classes_file}")
    
    # Save individual annotation files
    for image_name, annotations in data['annotations'].items():
        output_file = os.path.join(output_dir, f"{image_name}.txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(annotations))
        print(f"Saved annotations for {image_name} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO JSON annotations into per-image text files')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_dir', help='Directory to save the output text files')
    
    args = parser.parse_args()
    
    try:
        convert_yolo_json(args.input_file, args.output_dir)
        print("Successfully converted YOLO annotations into individual text files")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 