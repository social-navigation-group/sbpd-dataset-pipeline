import cv2
import time
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from .playback_mode import PlaybackMode
from PyQt6.QtGui import QPixmap, QImage, QIcon
from .trajectory_worker import TrajectoryWorker
from .trajectory_manager import TrajectoryManager
from utils.human_config_utils import HumanConfigUtils
from .trajectory_click_handler import TrajectoryClickHandler
from utils.file_utils import is_valid_video_file, get_file_size
from utils.logging_utils import log_info, log_warning, log_error
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QWidget, QMessageBox, QSizePolicy, QApplication
)

class VideoPlayer(QWidget):
    def __init__(self, video_controls, resource_manager):
        super().__init__()
        self.cap = None
        self.video_fps = 30  
        self.video_width = 0
        self.video_height = 0
        self.frame_cache = {}
        self.total_frames = 0  
        self.video_path = None
        self.human_config = None
        self.video_controls = video_controls
        self.resource_manager = resource_manager
        self.playback_mode = PlaybackMode.STOPPED 
        self.trajectory_manager = TrajectoryManager()

        self.human_config_path = self.resource_manager.get_human_config()
        self.human_config = HumanConfigUtils(self.human_config_path)

        self.trajectories = np.array([self.human_config.get_element(a, 'trajectories') for a in self.human_config.dict.keys()], dtype = object)
        self.traj_starts = np.array([self.human_config.get_element(a, 'traj_start') for a in self.human_config.dict.keys()])

        self.colors = np.array([
            [255, 255, 255], [0, 255, 255], [255, 0, 255],
            [255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]
        ], dtype = np.uint8)
        self.num_colors = len(self.colors)

        # TIMERS
        self.timer = QTimer()
        self.playback_speed = 1  
        self.timer.timeout.connect(self.update_frame)

        # GRAPHICS SCENE
        self.scene = QGraphicsScene(self)
        self.view = TrajectoryClickHandler(self.trajectory_manager, self.trajectories, self.scene)
        self.view.setScene(self.scene) 

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.view.setMinimumSize(1092, 888) 
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.view.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
    
    def load_video(self, path):
        """Loads the video file and initializes the UI elements."""
        if not is_valid_video_file(path):
            log_warning(f"Attempted to load invalid video file: {path}")
            QMessageBox.warning(self, "Invalid File", "The selected file is not a valid video format.")
            return

        file_size = get_file_size(path)
        log_info(f"Loading video: {path} (Size: {file_size})")

        if self.cap:
            self.cap.release()  

        self.video_path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            log_error(f"Failed to open video file: {path}")
            QMessageBox.warning(self, "Error", f"Failed to open video file from {path}")
            return

        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) 
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        log_info(f"Video Loaded: {path} | FPS: {self.video_fps} | Resolution: {self.video_width}x{self.video_height} | Total Frames: {self.total_frames}")
        self.frame_cache.clear()

        self.video_controls.max_frame_label.setText(str(self.total_frames))
        self.video_controls.frame_slider.setRange(0, self.total_frames)

        if hasattr(self, "trajectory_worker"):
            self.trajectory_worker.stop()
            self.trajectory_worker.wait()

        self.trajectory_worker = TrajectoryWorker(
            self.trajectories, 
            self.traj_starts, 
            self.colors, 
            self.video_width,  
            self.video_height,  
            self.total_frames,  
            self.video_fps,
            cache_size = 30  # Preload next 30 frames only
        )
        self.trajectory_worker.update_overlay.connect(self.update_trajectory_overlay)
        self.trajectory_worker.start() 

        self.trajectory_overlay = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        self.show_frame_at(0)

    def update_frame(self):
        """Handles frame updates for play, rewind, and forward."""
        if not self.cap or not self.cap.isOpened():
            self.timer.stop()
            return

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if self.playback_mode == PlaybackMode.PLAYING:
            new_frame = min(self.total_frames - 1, current_frame + self.playback_speed)
        elif self.playback_mode == PlaybackMode.REWINDING:
            new_frame = max(0, current_frame - self.playback_speed)
        elif self.playback_mode == PlaybackMode.FORWARDING:
            new_frame = min(self.total_frames - 1, current_frame + (self.playback_speed * 2))
        else:
            self.timer.stop()
            return

        self.show_frame_at(new_frame)

        # STOP IF AT BOUNDARIES
        if new_frame == 0 or new_frame == self.total_frames - 1:
            self.timer.stop()
            self.playback_mode = PlaybackMode.STOPPED
            self.video_controls.play_pause_button.setIcon(QIcon(self.resource_manager.get_icon("play", "play-60")))

    def show_frame_at(self, frame_number):
        """Displays the frame at a given position using caching"""
        self.video_controls.current_frame_label.setText(str(frame_number))
        self.video_controls.frame_slider.setValue(frame_number)
        self.trajectory_worker.update_frame(frame_number)

        if frame_number in self.frame_cache and self.frame_cache[frame_number] is not None:
            frame = self.frame_cache[frame_number]
        else:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) 
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    self.frame_cache[frame_number] = frame  
                else:
                    log_warning(f"Failed to read frame {frame_number}")
                    return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        if frame is not None:
            self.display_frame(frame)
        else:
            log_warning(f"Frame {frame_number} is None. Video might be corrupted or out of range.")

    def update_trajectory_overlay(self, overlay):
        log_info("[DEBUG] Overlay updated")
        self.trajectory_overlay = overlay
        self.update()
    
    def display_frame(self, frame):
        """Converts the OpenCV frame to a QPixmap and displays it."""
        if frame is None:
            log_warning("display_frame() received None frame. Skipping frame update.")
            return 

        try:
            start_time = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlayed_frame = cv2.addWeighted(frame, 1, self.trajectory_overlay, 0.5, 0)

            height, width, _ = overlayed_frame.shape
            bytes_per_line = 3 * width

            image = QImage(overlayed_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            if pixmap.isNull():
                log_warning("Converted QPixmap is null. Frame might be corrupted.")
                return

            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

            self.view.update()
            self.view.repaint()
            QApplication.processEvents()

            end_time = time.time()
            render_time = (end_time - start_time) * 1000  
            log_info(f"[PERF] Frame rendered in {render_time:.2f} ms")
        except Exception as e:
            log_error(f"Error in display_frame: {e}")

    def change_playback_mode(self, mode, speed=1):
        """Sets playback mode and adjusts timer settings."""
        if self.playback_mode == mode:
            self.pause()
            return
        
        self.playback_mode = mode
        self.playback_speed = speed

        if not self.timer.isActive():
            self.timer.start(1000 // self.video_fps)

    def stop(self):
        """Stops the video and resets to the first frame."""
        self.timer.stop()
        self.playback_mode = PlaybackMode.STOPPED
        if self.cap:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.show_frame_at(0)
    
        if hasattr(self, "trajectory_worker"):
            self.trajectory_worker.overlay_cache.clear()
        
        self.trajectory_overlay = np.zeros((self.video_height, self.video_width, 3), dtype = np.uint8)
        self.display_frame(self.frame_cache.get(0))  

    def play(self, speed = 2):
        """Starts video playback."""
        self.change_playback_mode(PlaybackMode.PLAYING, speed=speed)

    def pause(self):
        """Pauses playback."""
        self.timer.stop()
        self.playback_mode = PlaybackMode.STOPPED

    def rewind(self, speed = 2):
        """Starts rewinding at the given speed."""
        self.change_playback_mode(PlaybackMode.REWINDING, speed=speed)

    def forward(self, speed = 4):
        """Starts fast-forwarding at the given speed."""
        self.change_playback_mode(PlaybackMode.FORWARDING, speed=speed)