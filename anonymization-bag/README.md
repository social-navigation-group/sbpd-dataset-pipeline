# Anonymization on bag files

## Overview

The codes in this folder are for filtering topics and anonymization on bag files for most likely FPV videos. There are two types of anonymization provided: **detection-based** bbox blurring anonymization and **instance segmentation-based** masking anonymization. BEV videos should work too, but it is recommended to extract them from the bag files first and use the anonymization tool [here](https://github.com/social-navigation-group/sbpd-dataset-pipeline/tree/main/anonymization) to perform anonymization, because instance segmentation-based anonymization is not useful for BEV videos. 

Once everything is configured, only a single `./run_filter.sh` is needed to process all the bag files in a given directory. The script will anonymize your video. However, to maximize the values of the anonymized videos, we will also run 2D bounding box tracking and skeleton keypoint detection on your video. So it may take a long time to finish processing the videos...

This script will **reursively** search for all the bag files (that do not end with '_filtered' or '_merged') inside a given directory and perform topic filtering and anonymization. For each bag file named `<bag-name>`, the script will generate on its base path a `<bag-name>_filtered` with intermediate bag files, extracted FPV videos `<fpv-name>`, and the corresponding anonymized `<fpv-name>_processed` for each image topic in the original bag file. It will also generate a `<bag-name>_merged` with the finalized bag files. Please share the `<bag-name>_merged` bag files with us. If you run into disk space problems, feel free to delete `<bag-name>_filtered` when you are finished.

## Dependencies
Make sure you have ROS2 and CUDA installed. If not, you can use Docker to install.

## Install

You can choose to install either natively or via Docker. However, if your Ubuntu system is lower than 22.04, because ROS2 is not supported, you will have to install via Docker.

Regardless of which way, first
```
sudo chmod +x scripts/run_filter.sh
```

### Native install
Create a virtual environment or activate your existing virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies
```
pip3 install -r requirements.txt
```

Install pytorch (You need to install [CUDA](https://developer.nvidia.com/cuda-downloads) first, and install the correct [PyTorch](https://pytorch.org/get-started/previous-versions/) version that matches your CUDA)
```
pip3 install torch torchvision torchaudio
```

Install Ultralytics
```
pip3 install ultralytics
```

Install the slightly modified [ByteTrack](https://github.com/allanwangliqian/ByteTrack.git)
```
git clone https://github.com/allanwangliqian/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython pycocotools
pip3 install --upgrade 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

Download ByteTrack pretrained model
```
mkdir pretrained
pip3 install gdown
gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O ./pretrained/bytetrack_x_mot17.pth.tar
```

### Docker Install
Modify the `volumes` in `docker-compose.yaml` first. Change `/path/to/your/data` to the path where you saved all your bag files.

Build
```
docker-compose build
```

Launch
```
docker-compose up -d
```

Go to the container
```
docker-compose exec bag_processor bash
```

This should lead you to the workspace `bag_ws` in the container. The bag file data will be linked inside `data`, and the codes will be in `scripts`.

Source ROS setup bash script inside the container again
```
source /opt/ros/humble/setup.bash
```

## Usage

### Installed natively
Go to scripts
```
cd scripts
```

Modify `arguments.py`

Modify the default values of `--bytetrack-model`, and `--bytetrack_config` to be the actual paths of the detectron config path, bytetrack checkpoints path, and bytetrack config path. Use the given default values as examples to locate them.

Begin the topic filtering and anonymization process (detection-based bbox blurring).
```
./run_filter.sh -p <path to base path of all bag files> -a -b
```
Or, use instance segmentation-based masking instead. 
```
./run_filter.sh -p <path to base path of all bag files> -a -m
```

### Installed via Docker
Go to scripts
```
cd scripts
```

Begin the topic filtering and anonymization process (detection-based bbox blurring).
```
./run_filter.sh -a -b
```
Or, use instance segmentation-based masking instead. 
```
./run_filter.sh -a -m
```

### Description of flags
`-a`: Do not perform topic filtering. Simply save all topics. If not provided, it will only save the topics provided in `DEFAULT_TOPICS`. If you wish to filter topics, please modify `DEFAULT_TOPICS` in `run_filter.sh` manually.

`-p`: Specify the base path where all the bag files are stored. If not provided, it will use the path in `DEFAULT_BASE_PATH`.

`-b`: Use detection-based bbox blurring to perform anonymization.

`-m`: Use instance segmentation-based masking to perform anonymization.

If both `-b` and `-m` are provided, only `-m` will run. If neither is provided, anonymization will not be performed.

Note: If you run `./run_filter.sh -a`, this will not anonymize the videos, but it will still generate the 2D bounding box tracking and skeleton keypoint detections, and integrate them into the bag files.

### Other notes
If you use masking to anonymize data, feel free to modify the confidence threshold in `arguments.py` if there are too many false positives or if some humans are not picked up by the instance segmentation model.
