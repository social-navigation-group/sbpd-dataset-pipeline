# Human Data Labeling Tool

## Overview
The **Human Data Labeling Tool** is a PyQt6-based application designed for annotating human movement trajectories in video files. It provides an interactive UI for loading videos, importing trajectory files, labeling human data, and managing trajectory modifications.

This project is a Python conversion of an open-source [MATLAB-based tool](https://github.com/CMU-TBD/tbd_label_correction_UI) and has been tested on Linux systems.

## Features
- **Video Playback & Control**: Load and play videos with standard playback controls.
- **Trajectory Import & Editing**: Import trajectory files and modify data using an intuitive UI.
<!-- - **Human Labeling**: Assign context labels (e.g., Adults, Children, Elderly) to detected trajectories. -->
- **Undo Support**: Reverse modifications to trajectory data when needed.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```sh
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS

pip install --upgrade pip

# Install dependencies
- toml
- opencv-python
- numpy
- pyqt6
```

> **Note:** This application has been tested on **Linux**. Compatibility with Windows or macOS has not been verified.


## File Structure
```
project_root/
│── src/
│   ├── main.py  # Entry point
│   ├── logs/    # Log files for debugging
│   ├── ui/      # UI components (MainWindow, VideoControls, TabDialog)
│   ├── video_proc_comps/  # Video player and trajectory handlers
│   ├── utils/   # Logging, resource management, and file utilities
│── resources/   # Stylesheets, icons, and default config files
│── README.md    # Project documentation
```

### Usage
Run the main application script:
```sh
python3 path/to/main.py  # Use python or python3 depending on your system
```

## Workflow
1. **Launch the application**
2. **Load a video file** via the dropdown menu.
3. **Import a trajectory file** to begin (or continue) labeling.
4. **Modify trajectories**: Relabel, delete, join, break, or disentangle trajectories.
<!-- 5. **Assign human context labels** to each detected trajectory. -->
5. **Save progress** via the dropdown menu (as a TOML file).

## Button Functions

### Trajectory Editing Buttons

#### **Relabel**
Allows redrawing an existing trajectory. 
1. Select the trajectory you want to relabel:
   - Input the trajectory ID manually in the input field, or 
   - Click directly on the trajectory in the video. 
2. Click the **Select** button. 
3. Redraw the desired trajectory by clicking in the video. 
   - For better accuracy, focus on the person's feet (when possible).

[Watch Relabel Video](videos/relabel.mp4)

---

#### **Missing**
Draws a missing trajectory where none currently exists. 
1. Draw the desired trajectory by clicking in the video. 
   - For better accuracy, focus on the person's feet (when possible). 

[Watch Missing Video](videos/missing.mp4)

---

#### **Break**
Splits a selected trajectory into two at the current frame. 
1. Click directly on the point where you want to split the trajectory. 
2. Click **Apply** to finalize the action. 
   - The original trajectory is divided, and a new one starts from the split point. 

[Watch Break Video](videos/break.mp4)

---

#### **Join**
Merges two selected trajectories into one, interpolating where necessary. 
1. Input the first trajectory ID manually in the input field and click **Select**. 
2. Input the second trajectory ID (to merge into the first one) and click **Select**. 
3. Click **Apply** to finalize the action. 

[Watch Join Video](videos/join.mp4)

---

#### **Delete**
Removes the selected trajectory from the dataset. 
1. Select the trajectory you want to delete:
   - Input the trajectory ID manually in the input field, or 
   - Click directly on the trajectory in the video. 
2. Click the **Select** button. 
3. Click **Apply** to finalize the action. 

[Watch Delete Video](videos/delete.mp4)

---

#### **Disentangle**
Swaps two overlapping trajectories after a certain frame to correct misidentified paths. 
1. Select **Trajectory 1**:
   - Input the trajectory ID manually in the input field, or 
   - Click directly on the trajectory in the video. 
2. Click the **Select** button. 
3. Select **Trajectory 2**:
   - ONLY click directly on the second trajectory at the point where you want to disentangle. 
4. Click **Apply** to finalize the action. 

[Watch Disentangle Video](videos/disentangle.mp4)

---

#### **Undo**
Reverts the last modification to the trajectory data. 
- Each click reverts the most recent unsaved modification. 
- Once saved, modifications can no longer be undone using this button. 

[Watch Undo Video](videos/undo.mp4)


## Error Logging
The application allows users to manually log errors encountered during labeling. Users can enter error descriptions through the UI, which will be recorded in the log file for review. This helps in debugging labeling inconsistencies or software-related issues. ubmitting any errors seen in the video that cannot be fixed with the above transformations. The user will describe the error in the textbox and then press the Submit button, which will record their message along with the current frame and the currently active trajectories of that frame


## License
This project is licensed under the MIT License. See `LICENSE` for details.
