# Human Data Labeling Tool

The **Human Data Labeling Tool** is a PyQt6-based application designed for annotating human movement trajectories in video files. It provides an interactive UI for loading videos, importing trajectory files, labeling human data, and managing trajectory modifications.
This project is a Python conversion of an open-source [MATLAB-based tool](https://github.com/CMU-TBD/tbd_label_correction_UI) and has been tested on a Linux system.

## Features
- **Video Playback & Control**: Load and play videos with standard playback controls.
- **Trajectory Import & Editing**: Import trajectory files and modify data using an intuitive UI.
- **Undo Support**: Reverse modifications to trajectory data when needed.

## Installation
### Prerequisites
Ensure you have **Python updated**. It is recommended to use a virtual environment:

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

Create the following folders
```
mkdir resources/config/original_data resources/videos
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
python3 path/to/main.py  # Use python or python3 
```
> To display videos in the dropdown menu of the UI, add them to the `resources/config/videos` folder. The trajectories can be placed in any directory.
> 🔵 In current version the tab **"Human"** in the UI is under construction.

## Workflow
1. **Launch the application**
2. **Load a video file** via the dropdown menu.
3. **Import a trajectory file** to begin (or continue) labeling.
4. **Modify trajectories**: Relabel, delete, join, break, or disentangle trajectories.
5. 🔴 **SAVE PROGRESS**! via the dropdown menu (as a TOML file).
   
> 🔴 **After making a modification, it is recommended to push the video forward to verify if the changes produce the desired behavior in future frames. Once confirmed, save the video—either as a new file or by replacing an existing one (user preference).**

## Button Functions

### Trajectory Editing Buttons
#### **Relabel**
##### Summarized Steps
1. Trajectory Selection:
   - **Input the trajectory ID manually in the input field**, or 
   - **Click directly on the trajectory in the video.**
2. Click the **Select** button. 
3. Redraw the desired trajectory by clicking in the video. Each click moves the video 30 frames ahead to the next point.
   - **For better accuracy, focus on the person's feet (when possible).**
4. Click the **Apply** button.
5. To CANCEL, click the **Cancel** button
6. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button allows the user to draw a new trajectory starting at the frame the button was clicked. After pressing the button, the UI will update, prompting an input field to insert the ID of the desired trajectory to modify. This can also be accomplished by clicking directly on the trajectory, which will directly input the trajectory ID into the input field. After clicking, the user should click on the **"Select"** button to select the trajectory. After selecting the trajectory, the user must click directly at the video to draw the trajectory - each click will make the video jump 30 frames ahead so that the user can keep clicking the next point of the trajectory. This continues until there are no more frames or the user manually stops the relabeling. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the relabel button, the user can cancel it, enabling the other modifiers. After relabeling has been applied, the points are joined together into a single solid trajectory.

[Watch example of the relabel modifier](videos/relabel.mp4)

---

#### **Missing**
##### Summarized Steps
1. Draw the desired trajectory by **clicking in the video** - each click pushes the video 30 frames ahead for the next point.
   - **For better accuracy, focus on the person's feet (when possible).**
2. Click the **Apply** button.
3. To CANCEL, click the **Cancel** button
4. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button allows the user to draw a missing trajectory where none currently exists, starting at the frame where the button was clicked. After pressing the button, the UI will update, showing the "Apply" and "Cancel" buttons. If the user attempts to click **"Apply"** before making any change, a message will be prompted to ensure proper action, be it to cancel or draw the missing trajectory. To draw the trajectory, the user must click directly at the video - each click will make the video jump 30 frames ahead so that the user can keep clicking the next point of the trajectory. This continues until there are no more frames or the user manually stops the relabeling. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the missing button, the user can cancel it, enabling the other modifiers.

[Watch example of the missing modifier](videos/missing.mp4)

---

#### **Break**
##### Summarized Steps
1. Trajectory Selection:
   - **Click directly on the point in the trajectory (in the video).**
2. Click the **Select** button. 
3. Click the **Apply** button.
4. To CANCEL, click the **Cancel** button.
5. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button divides the selected trajectory (determined by the point clicked in the desired trajectory to break) at the current frame into two separate trajectories. The first trajectory begins where the original trajectory began and ends at the current frame, and the second begins at the current frame and ends where the original trajectory ended. After pressing the button, the UI will update, prompting an input field to insert the ID of the desired trajectory to modify. This can also be accomplished by clicking directly on the trajectory, which will directly input the trajectory ID into the input field. After clicking, the user should click the **"Select"** button to select the trajectory. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the **"Break"** button, the user can cancel it, enabling the other modifiers. 

> Even though it is possible to insert it manually in the input box, the function will only work properly when clicked directly at the desired point in the trajectory.

[Watch example of the break modifier](videos/break.mp4)

---

#### **Join**
##### Summarized Steps
1. Trajectory Selection:
   - **Input the trajectory ID manually in the input field**, or 
   - **Click directly on the trajectory in the video.**
2. Click the **Select** button.
3. Repeat steps 1 and 2 for the second trajectory.
4. Click the **Apply** button.
5. To CANCEL, click the **Cancel** button
6. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button connects the two selected trajectories (determined by the values inside "Trajectory 1" and "Trajectory 2"). After pressing the button, the UI will update, prompting an input field to insert the ID of the desired trajectory to modify. This can also be accomplished by clicking directly on the trajectory, which will directly input the trajectory ID into the input field, which next the user must click on the **"Select"** button and repeat the process for the second trajectory. **However, the trajectories can only be clicked when active at the frame, meaning, case the second trajectory is not visible, the user needs to use the slider or click **"Play"** or **"Fast-Forward"** to push the video (frames) forward**. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the **"Join"** button, the user can cancel it, enabling the other modifiers. After joining has been applied, the resulting trajectory begins where "Trajectory 1" started and ends where "Trajectory 2" ends. 

[Watch example of the join modifier](videos/join.mp4)

---

#### **Delete**
##### Summarized Steps
1. Trajectory Selection:
   - **Input the trajectory ID manually in the input field**, or 
   - **Click directly on the trajectory in the video.**
2. Click the **Select** button. 
3. Click the **Apply** button.
4. To CANCEL, click the **Cancel** button
5. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button deletes the selected trajectory (which is determined by the trajectory number inside of the "Trajectory 1" textbox) from the current frame onward. After pressing the button, the UI will update, prompting an input field to insert the ID of the desired trajectory to modify. This can also be accomplished by clicking directly on the trajectory, which will directly input the trajectory ID into the input field. After clicking, the user should click the **"Select"** button to select the trajectory. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the **"Delete"** button, the user can cancel it, enabling the other modifiers.

[Watch example of the delete modifier](videos/delete.mp4)

---

#### **Disentangle**
##### Summarized Steps
1. Trajectory Selection:
   - **Input the trajectory ID manually in the input field**, or 
   - **Click directly on the trajectory in the video.**
2. Click the **Select** button.
3. **Click directly on the point you want to disentangle in the 2nd trajectory (in the video).**
4. Click the **Select** button.
5. Click the **Apply** button.
6. To CANCEL, click the **Cancel** button
7. If the results are as intended **SAVE** the progress through the menu bar.

##### Overview
This button swaps two overlapping trajectories after a certain frame to correct misidentified paths. After pressing the button, the UI will update, prompting an input field to insert the ID of the desired trajectory to modify. This can also be accomplished by clicking directly on the trajectory, which will directly input the trajectory ID into the input field, which next the user must click on the **"Select"** button. **FOR THE 2nd TRAJECTORY** the user must click on the point to disentangle in the the trajectory, which will also include the trajectory ID on the input field, and the user must click the **"Select"** button afterward. To apply the change, the user needs to click the **"Apply"** button. If the user made the mistake of clicking in the **"Disentangle"** button, the user can cancel it, enabling the other modifiers. 

> Even though it is possible to insert it manually in the input box for the 2nd trajectory, the function will only work properly when manually inputting the ID in the input field.

[Watch example of the disentangle modifier](videos/disentangle.mp4)

---

#### **Undo**
##### Summarized Steps
- Each click reverts the most recent unsaved modification. 
- Once saved, modifications can no longer be undone using this button.

---

### 💾 Autosave Feature
After each successful transformation—such as **Relabel**, **Break**, **Join**, **Delete**, etc. — the current state of the trajectories is **automatically saved** to an `autosave.toml` file.

This autosave happens when:
- You click "Apply" for any transformation
- The trajectory selection is cleared

The autosaved file is located in: `resources/config/original_data/autosave.toml`
> 🛑 **Note:** Autosave is session-persistent, but it does not replace manual saving. To export and preserve your final labels, use the **Save** option in the menu bar.

### 🎮 Media Controls & Keyboard Shortcuts

The tool supports convenient keyboard shortcuts for video navigation:

| Key         | Action                            |
|-------------|-----------------------------------|
| `Space`     | Play / Pause                      |
| `S`         | Stop and reset to first frame     |
| `←` (Left)  | Move one frame backward           |
| `→` (Right) | Move one frame forward            |
| `Enter`     | Confirm trajectory selection / Apply changes |
| `Esc`       | Cancel current operation          |

##### Overview
This button reverts the last modification to the trajectory data. The undo starts with the most recent modification to the first one made during the session. After saving the progress, the undo function **WILL NOT REVERT THE MODIFICATION**. By default, the current implementation limits the undos (by default, 10 times only). The user can increase the limit value (if needed).

[Watch example of the undo modifier](videos/undo.mp4)


## Error Logging
The application allows users to manually log errors encountered during labeling. Users can enter error descriptions through the UI, which will be recorded in the log file for review. This helps in debugging labeling inconsistencies or software-related issues. ubmitting any errors seen in the video that cannot be fixed with the above transformations. The user will describe the error in the textbox and then press the Submit button, which will record their message along with the current frame and the currently active trajectories of that frame

## Playground
Users can find an example to test the trajectory modifiers in the `resources/config` directory. In the UI's dropdown menu, select `example_small_data.avi` and then import `example_small_data.toml`.
 - `original_data/example_small_data.toml`
 - `videos/example_small_data.avi`

## Acknowledgments
The icons used in this project are sourced from [ICONS8](https://icons8.jp) under their free license.

## License
[Apache 2.0 License](../LICENSE)
