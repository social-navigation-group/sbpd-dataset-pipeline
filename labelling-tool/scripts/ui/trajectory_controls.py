import os
from PyQt6.QtCore import Qt, pyqtSignal
from datetime import datetime
from utils.logging_utils import log_info
from video_proc_comps.button_controller import ButtonController
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit, QGroupBox, QSlider, QCheckBox
)

class TrajectoryControls(QWidget):
    time_window_changed = pyqtSignal(int)
    time_window_enabled_changed = pyqtSignal(bool)

    def __init__(self, video_player, parent: QWidget):
        super().__init__(parent)
        self.video_player = video_player
        self.is_traj_input_visible = False
        self.button_controller = ButtonController(self)

        # SET THE CONTROLS IN THE LAYOUT
        self.labeling_layout = self.create_labeling_controls()
        self.setLayout(self.labeling_layout)
        
    def create_labeling_controls(self):
        """Creates and returns the labeling control UI."""

        traj_edit_group = QGroupBox("Trajectory Editing: ")
        traj_edit_layout = QVBoxLayout()

        labels = ["Relabel", "Missing", "Break", "Join", "Delete", "Disentangle", "Undo"]
        self.buttons = [QPushButton(label) for label in labels]
        for label, button in zip(labels, self.buttons):
            button.clicked.connect(getattr(self.button_controller, f"on_{label.lower()}_clicked"))
            traj_edit_layout.addWidget(button)
            button.setEnabled(False)

        traj_edit_group.setLayout(traj_edit_layout)

        time_window_group = QGroupBox("Time Window: ")
        time_window_layout = QHBoxLayout()

        self.time_window_checkbox = QCheckBox()
        self.time_window_checkbox.setChecked(True)
        self.time_window_checkbox.setEnabled(False)
        self.time_window_checkbox.stateChanged.connect(self.emit_time_window_enabled_changed)

        self.time_window_label = QLabel()
        self.time_window_label.setText("5 seconds")

        self.time_window_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_window_slider.setValue(5)
        self.time_window_slider.setMinimum(5)
        self.time_window_slider.setMaximum(60)
        self.time_window_slider.setPageStep(5)
        self.time_window_slider.setSingleStep(5)
        self.time_window_slider.setEnabled(False)
        self.time_window_slider.setTickInterval(5)
        self.time_window_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.time_window_slider.sliderReleased.connect(self.snap_to_nearest_step)        

        time_window_layout.addWidget(self.time_window_checkbox)
        time_window_layout.addWidget(self.time_window_slider)
        time_window_layout.addWidget(self.time_window_label)
        time_window_group.setLayout(time_window_layout)

        traj_log_group = QGroupBox("Error Logging: ")
        traj_log_layout = QVBoxLayout()

        self.log_labeling = QTextEdit()
        self.log_labeling.setPlaceholderText("Log any errors seen in the video that cannot be fixed with the above transformations...")
        self.log_labeling.setEnabled(False)
        self.log_labeling.textChanged.connect(self.on_text_changed)

        self.log_submit_button = QPushButton()
        self.log_submit_button.setText("Submit")
        self.log_submit_button.setEnabled(False)
        self.log_submit_button.clicked.connect(self.on_log_submit)

        traj_log_layout.addWidget(self.log_labeling)
        traj_log_layout.addWidget(self.log_submit_button)
        traj_log_group.setLayout(traj_log_layout)

        labeling_layout = QVBoxLayout()
        labeling_layout.addWidget(traj_edit_group)
        labeling_layout.addWidget(time_window_group)
        labeling_layout.addWidget(traj_log_group)

        return labeling_layout

    def on_log_submit(self):
        """Logs errors into a text file, appending under the same date or starting a new section if the date changes."""
        current_frame = self.button_controller.video_player.current_frame
        selected_trajectory = self.button_controller.trajectory_manager.get_active_trajectories(current_frame)
        log_text = self.log_labeling.toPlainText().strip()

        if not log_text:
            return 

        log_entry = f"Current Frame: {current_frame}, Active Trajectories: {selected_trajectory}\nError Description: {log_text}\n\n"

        log_file_path = "trajectory_transformation_error_logs.txt"
        save_directory = os.path.join(self.button_controller.video_player.resource_manager.config_path)
        os.makedirs(save_directory, exist_ok = True)

        file_path = os.path.join(save_directory, log_file_path)
        today = datetime.now().strftime("%Y/%m/%d")

        # Check if the log file exists and contains today's date
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                logs = file.read()
            
            # Append under the existing date if today's section exists
            if f"---------------------------// {today} - LOGS //---------------------------" in logs:
                with open(file_path, "a") as file:
                    file.write(log_entry)
            else:
                # Create a new date section if the day has changed
                with open(file_path, "a") as file:
                    file.write(f"\n\n---------------------------// {today} - LOGS //---------------------------\n\n")
                    file.write(log_entry)
        else:
            # Create a new file with today's log section
            with open(file_path, "w") as file:
                file.write(f"---------------------------// {today} - LOGS //---------------------------\n\n")
                file.write(log_entry)

        log_info(f"Error logs saved successfully in {file_path}")
        self.log_labeling.clear()

    def on_text_changed(self):
        """Enables the submit button when there is text in the QTextEdit, disables it otherwise."""
        if self.log_labeling.toPlainText().strip():
            self.log_submit_button.setEnabled(True)
        else:
            self.log_submit_button.setEnabled(False)
    
    def create_trajID_input(self, labeling_layout, number, mode):
        self.delete_trajID_input(self.labeling_layout, mode)

        if mode in [2, 7]:
            self.create_apply_button(labeling_layout, 0)
            self.create_cancel_button(labeling_layout, 1)
            self.is_traj_input_visible = True
            return labeling_layout
        
        labeling_layout.insertWidget(0, QLabel(f"Trajectory {number}"))
        horizontal_layout = QHBoxLayout()

        trajectory_input = QLineEdit()
        horizontal_layout.addWidget(trajectory_input)
        
        self.select_button = QPushButton("Select")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
            }

            QPushButton:disabled {
                background-color: lightgray;
            }
        """)

        self.select_button.clicked.connect(self.button_controller.on_select_pressed)
        self.select_button.setShortcut(Qt.Key.Key_Return)
        horizontal_layout.addWidget(self.select_button)
        labeling_layout.insertLayout(1, horizontal_layout)

        self.create_apply_button(labeling_layout, 2)
        self.create_cancel_button(labeling_layout, 3)
        
        self.is_traj_input_visible = True
        return labeling_layout
    
    def create_apply_button(self, labeling_layout, widget_position):
        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: green;
            }

            QPushButton:disabled {
                background-color: lightgray;
            }
        """)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.button_controller.on_apply_pressed)
        self.apply_button.setShortcut(Qt.Key.Key_Return)
        labeling_layout.insertWidget(widget_position, self.apply_button)
    
    def create_cancel_button(self, labeling_layout, widget_position):
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("background-color: red;")
        self.cancel_button.clicked.connect(self.button_controller.on_cancel_pressed)
        self.cancel_button.setShortcut(Qt.Key.Key_Escape)
        labeling_layout.insertWidget(widget_position, self.cancel_button)
    
    def delete_trajID_input(self, labeling_layout, mode):
        if self.is_traj_input_visible is False: 
            return labeling_layout

        if mode == 2:
            number_of_widgets = 2
            self.delete_widgets(labeling_layout, number_of_widgets)
            self.is_traj_input_visible = False
            return labeling_layout

        number_of_widgets = 4 
        self.delete_widgets(labeling_layout, number_of_widgets)   
        self.is_traj_input_visible = False

        return labeling_layout
    
    def delete_widgets(self, labeling_layout, number_of_widgets):
        for _ in range(number_of_widgets):
            item = labeling_layout.takeAt(0)  

            if item is None:
                continue

            widget = item.widget()
            if widget:
                widget.deleteLater()  
                continue 

            layout = item.layout()
            if layout:
                while layout.count():  
                    sub_item = layout.takeAt(0)
                    sub_widget = sub_item.widget()

                    if sub_widget:
                        sub_widget.deleteLater()
                layout.deleteLater()  

    def get_time_window_seconds(self):
        return self.time_window_slider.value()
    
    def update_time_window_label(self):
        value = self.time_window_slider.value()
        self.time_window_label.setText(f"{value} seconds")

    def snap_to_nearest_step(self):
        current_value = self.time_window_slider.value()
        snapped_value = round(current_value / 5) * 5
        self.time_window_slider.setValue(snapped_value)  
        self.update_time_window_label()
        self.time_window_changed.emit(snapped_value)

    def is_time_window_enabled(self):
        return self.time_window_checkbox.isChecked()
    
    def emit_time_window_enabled_changed(self):
        enabled = self.time_window_checkbox.isChecked()
        self.time_window_slider.setEnabled(enabled)
        self.time_window_enabled_changed.emit(self.time_window_checkbox.isChecked())

