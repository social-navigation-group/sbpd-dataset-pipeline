import os
import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from .tab_dialog import TabDialog
from .video_controls import VideoControls
from utils.logging_utils import log_info, log_warning
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSplitter, QFileDialog, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self, resource_manager: str):
        super().__init__()
        self.fully_loaded = False
        self.trajectory_file = None
        self.resource_manager = resource_manager

        # LOAD EXTERNAL STYLESHEET
        self.load_stylesheet(self.resource_manager.get_stylesheet())

        # MAIN WINDOW
        self.setWindowTitle("Human Data Labeling")
        self.setFixedSize(1642, 1008)

        menu = self.menuBar()
        self.file_menu = menu.addMenu("File")
        
        self.save_action = QAction("Save Progress", self)
        self.save_action.triggered.connect(self.save_progress)

        import_trajectories = QAction("Import Trajectories", self)
        import_trajectories.triggered.connect(self.import_trajectory_file)

        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(import_trajectories)
        self.file_menu.setEnabled(False)

        # MAIN WIDGET
        container = QWidget()
        main_layout = QVBoxLayout()

        # SPLITTER DIVIDING MEDIA AND LABELING CONTROLS
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # UI COMPONENTS
        self.video_controls = VideoControls(self.resource_manager, self)
        self.tab_dialog = TabDialog(self.video_controls)

        main_splitter.addWidget(self.video_controls)
        main_splitter.addWidget(self.tab_dialog)

        main_layout.addWidget(main_splitter)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_stylesheet(self, path):
        try:
            with open(path, "r") as f:
                self.setStyleSheet(f.read())
                log_info(f"Loaded stylesheet: {path}")
        except FileNotFoundError:
            log_warning(f"Stylesheet {path} not found.")

    def import_trajectory_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dialog.setNameFilters(["TOML Files (*.toml)", "All Files (*)"])

        file_dialog.setDirectory(self.resource_manager.config_path)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            if selected_file:
                try:
                    self.trajectory_file = selected_file
                    self.update_trajectory_in_video_player()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load trajectory file: {e}")

    def update_trajectory_in_video_player(self):
        """Updates the VideoPlayer with the newly imported trajectory file."""
        if self.trajectory_file and hasattr(self.video_controls, "video_player"):
            try:
                self.video_controls.video_player.update_human_config(self.trajectory_file)
                QMessageBox.information(self, "Success", "Trajectory data updated in Video Player.")

                for btn in [self.video_controls.rewind_button, self.video_controls.play_pause_button, self.video_controls.stop_button, self.video_controls.forward_button, self.video_controls.video_dropdown, self.video_controls.frame_forward, self.video_controls.frame_back]: 
                    btn.setEnabled(True)
                self.video_controls.frame_slider.setEnabled(True)
                self.save_action.setEnabled(True)
                
                for btn_functions in self.tab_dialog.trajectory_controls.buttons:
                    btn_functions.setEnabled(True)
                    
                if hasattr(self.tab_dialog, 'human_controls'):
                    self.tab_dialog.human_controls.startIDinput.setEnabled(True)
                    self.tab_dialog.human_controls.startButton.setEnabled(True)
                self.fully_loaded = True
                self.tab_dialog.trajectory_controls.log_labeling.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update video player: {e}")
    
    def save_progress(self):
        """Opens a file dialog to allow user to specify a save filename"""
        save_dir = "saved_modified_trajectories"
        save_path = os.path.join(self.resource_manager.config_path, save_dir)
        os.makedirs(save_path, exist_ok = True)
        
        # Generate the default filename
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y_%m_%d_-_%H:%M")
        sanitized_video_filename = os.path.splitext(os.path.basename(self.resource_manager.video_name))[0]
        default_filename = f"{timestamp}_-_{sanitized_video_filename}.toml"
        default_filepath = os.path.join(save_path, default_filename)

        # Open a file dialog for the user to choose a filename
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setNameFilters(["TOML Files (*.toml)", "All Files (*)"])
        file_dialog.setDefaultSuffix("toml")
        
        file_dialog.selectFile(default_filepath)
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            
            if file_path:  
                try:
                    self.video_controls.video_player.human_config.save_human_config(file_path)
                    QMessageBox.information(self, "Success", f"File saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
        