#!/bin/bash

BASE_PATH=/path/to/directory       # change this to path to directory with all anonymized split bags

dir_list=()
for dir in "$BASE_PATH"/*/; do
    # Check if it is a directory
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        dir_list+=("$dir_name")
    fi
done

# Iterate over the list and print each directory
for BAG_FILE_NAME in "${dir_list[@]}"; do
    echo "[annotate.sh] Processing file: $BAG_FILE_NAME"

    mkdir -p ./temp_files/${BAG_FILE_NAME}/clouds
    mkdir -p ./temp_files/${BAG_FILE_NAME}/fpv_images
    mkdir -p ./temp_files/${BAG_FILE_NAME}/cloud_images

    # Write cloud and fpv frames
    python3 generate_video_frames.py ${BASE_PATH} ${BAG_FILE_NAME}

    # Read frame rates from temp file
    source ./temp_files/${BAG_FILE_NAME}/fps.txt

    # Write cloud video from frames
    ffmpeg -framerate ${CLOUD_FRAME_RATE} -i ./temp_files/${BAG_FILE_NAME}/cloud_images/image_%d.png -s 1920x1080 -c:v libx264 -pix_fmt yuv420p -crf 18 ./temp_files/${BAG_FILE_NAME}/cloud.mp4

    # Write cloud video from frames
    ffmpeg -framerate ${FPV_FRAME_RATE} -i ./temp_files/${BAG_FILE_NAME}/fpv_images/image_%d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./temp_files/${BAG_FILE_NAME}/fpv.mp4

    # Combine videos
    ffmpeg -i ./temp_files/${BAG_FILE_NAME}/cloud.mp4 -i ./temp_files/${BAG_FILE_NAME}/fpv.mp4 -filter_complex "[1:v] scale=640:-1 [pip]; [0:v][pip] overlay=0:H-h" -framerate 30 -c:v libx264 -pix_fmt yuv420p -crf 18 -preset veryfast ${BAG_FILE_NAME}.mp4

    # Delete temp files
    rm ./temp_files/${BAG_FILE_NAME}/ -rf
    
    echo "[annotate.sh] Completed processing: $BAG_FILE_NAME"

    # break

done