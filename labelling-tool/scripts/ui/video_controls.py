import os
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from utils.file_utils import list_video_files
from utils.logging_utils import log_info, log_warning
from video_proc_comps.video_player import VideoPlayer
from video_proc_comps.playback_mode import PlaybackMode
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QSlider, QSizePolicy, QLabel, QVBoxLayout, QComboBox, QMessageBox
)

class VideoControls(QWidget):
    def __init__(self, resource_manager: str, parent = None):
        super().__init__(parent)
        self.resource_manager = resource_manager
        self.resources_path = self.resource_manager.resources_path
        self.video_player = VideoPlayer(self, self.resource_manager, self)

        # VIDEO FILE DROPDOWN
        self.video_dropdown = QComboBox()
        self.populate_video_list()
        self.video_dropdown.currentIndexChanged.connect(self.load_video) 

        # SET THE CONTROLS IN THE LAYOUT
        self.setLayout(self.create_video_controls())

    def populate_video_list(self):
        """Scans the default directory and populates the dropdown with available videos."""
        videos_path = os.path.join(self.resources_path, "videos")
        video_files = list_video_files(videos_path)
        self.video_dropdown.clear()

        if not video_files:
            log_info("No video files found in the directory.")
            self.video_dropdown.addItem("No videos found")
        else:
            log_info(f"Found {len(video_files)} video files in {videos_path}")
            self.video_dropdown.addItem("Select a video")
            self.video_dropdown.addItems(video_files)

    def create_button(self, icon_path, callback, size = (50, 22)):
        """Helper function to create buttons with icons."""
        button = QPushButton()
        button.setFixedSize(*size)
        button.setIcon(QIcon(icon_path))
        button.setEnabled(False)
        button.clicked.connect(callback)
        return button

    def create_video_controls(self):
        """Creates and returns the video control UI."""

        # PLAYBACK BUTTONS
        self.rewind_button = self.create_button(self.resource_manager.get_icon("rewind", "rewind-60"), self.toggle_to_rewind)
        self.play_pause_button = self.create_button(self.resource_manager.get_icon("play", "play-60"), self.toggle_to_play)
        self.stop_button = self.create_button(self.resource_manager.get_icon("stop", "stop-60"), self.toggle_to_stop)
        self.forward_button = self.create_button(self.resource_manager.get_icon("fast-forward", "forward-60"), self.toggle_to_forward)
        # self.upload_button = self.create_button(self.resource_manager.get_icon("upload", "upload-60"), self.load_video)

        playback_controls = QHBoxLayout()
        for btn in [self.rewind_button, self.play_pause_button, self.stop_button, self.forward_button, self.video_dropdown]: #, self.upload_button
            playback_controls.addWidget(btn)
            if btn == self.forward_button:
                playback_controls.addStretch(1)

        # FRAME SLIDER
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_moved)

        self.current_frame_label = QLabel()
        self.current_frame_label.setText("0")
        self.max_frame_label = QLabel()
        self.max_frame_label.setText("0")

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.current_frame_label)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.max_frame_label)

        # VIDEO LAYOUT
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_player)
        video_layout.addLayout(playback_controls)
        video_layout.addLayout(slider_layout)

        return video_layout
    
    def load_video(self):
        """Loads the selected video from the dropdown.""" 
        index = self.video_dropdown.currentIndex()
        selected_video = self.video_dropdown.currentText()

        if index == 0 or selected_video == "No videos found":
            log_warning("No valid video selected.")
            return  

        video_path = self.resource_manager.get_video(selected_video)
        log_info(f"User selected video file: {video_path}")
        
        self.video_player.load_video(video_path)
        self.parent().parent().parent().file_menu.setEnabled(True)
        self.parent().parent().parent().save_action.setEnabled(False)

        total_frames = int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            self.frame_slider.setRange(0, total_frames)
            self.max_frame_label.setText(str(total_frames))
        else:
            log_warning(f"Invalid frame count for {video_path}")
    
    def toggle_to_play(self):
        """Toggles between play and pause, ensuring a single click pauses all playback modes."""
        if self.video_player.playback_mode in [PlaybackMode.PLAYING, PlaybackMode.REWINDING, PlaybackMode.FORWARDING]:
            log_info("Video playback paused")
            self.video_player.pause()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("play", "play-60")))
        else:
            log_info("Video playback started")
            self.video_player.play()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("pause", "pause-60")))

    def toggle_to_rewind(self):
        """Toggles between rewind and stop."""
        if self.video_player.playback_mode == PlaybackMode.REWINDING:
            self.video_player.pause()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("play", "play-60")))
        else:
            self.video_player.rewind()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("pause", "pause-60")))

    def toggle_to_forward(self):
        """Toggles between fast forward and stop."""
        if self.video_player.playback_mode == PlaybackMode.FORWARDING:
            self.video_player.pause()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("play", "play-60")))
        else:
            self.video_player.forward()
            self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("pause", "pause-60")))

    def toggle_to_stop(self):
        """Stops playback and resets to the first frame."""
        self.video_player.stop()
        self.video_player.trajectory_worker.overlay_cache.clear()
        self.video_player.trajectory_overlay = np.zeros((self.video_player.video_height, self.video_player.video_width, 3), dtype=np.uint8)
        self.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("play", "play-60")))
        self.frame_slider.setValue(0)

    def slider_moved(self, position):
        """Handles manual frame seeking via the slider."""
        if self.video_player.cap and self.video_player.cap.isOpened():
            self.video_player.show_frame_at(position)
            self.current_frame_label.setText(str(position))

    def get_video_player(self):
        return self.video_player
