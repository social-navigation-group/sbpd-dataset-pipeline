import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from utils.logging_utils import log_info, log_error, log_debug

class TrajectoryWorker(QThread):
    update_overlay = pyqtSignal(np.ndarray)

    def __init__(self, trajectory_manager, color_generator, video_width, video_height, total_frames, video_fps = 30, cache_size = 30):
        super().__init__()
        self.running = True
        self.frame_number = -1
        self.overlay_cache = {} 
        self.video_fps = video_fps
        self.cache_size = cache_size  # Buffer only 30 frames (~1 sec)
        self.video_width = video_width
        self.total_frames = total_frames
        self.video_height = video_height
        self.color_generator = color_generator
        self.trajectory_manager = trajectory_manager

    def run(self):
        """Runs the thread and updates overlay when frame changes."""
        last_frame = -1  

        while self.running:
            if self.frame_number < 0 or self.frame_number == last_frame:
                time.sleep(1 / self.video_fps)  
                continue
            last_frame = self.frame_number  

            if self.frame_number in self.overlay_cache:
                overlay = self.overlay_cache[self.frame_number]
            else:
                overlay = self._generate_overlay(self.frame_number)
                self.overlay_cache[self.frame_number] = overlay

            self.update_overlay.emit(overlay)
            self._preload_future_frames(self.frame_number)

    def _generate_overlay(self, frame_number):
        """Generates trajectory overlay and labels for a frame."""
        overlay = np.zeros((self.video_height, self.video_width, 3), dtype = np.uint8)
        active_trajectories = self.trajectory_manager.get_active_trajectories(frame_number)
        highlighted_trajs_info = self.trajectory_manager.get_selected_trajectory()
        highlighted_trajs = [info[0] for info in highlighted_trajs_info]

        log_debug(f"Active Trajectories: {active_trajectories}")
        log_debug(f"Highlighted Trajectory: {highlighted_trajs}")

        for traj_id in active_trajectories:
            if traj_id not in active_trajectories:
                continue

            traj = self.trajectory_manager.trajectories[traj_id]
            traj_start = self.trajectory_manager.traj_starts[traj_id]
            frame_offset = frame_number - traj_start

            if frame_offset < 0 or frame_offset >= len(traj):
                continue

            try:
                active_traj = traj[: max(1, frame_offset + 1)]
                color = self.color_generator.get_color(traj_id)

                if traj_id in highlighted_trajs: 
                    highlighted_color = np.array([0, 255, 0], dtype = np.uint8) 
                    log_debug(f"Drawing trajectory {traj_id} with HIGHLIGHT color {highlighted_color.tolist()}")
                else:
                    highlighted_color = np.array(color, dtype = np.uint8)
                    log_debug(f"Drawing trajectory {traj_id} with NORMAL color {highlighted_color.tolist()}")

                if len(active_traj) > 1:
                    for j in range(len(active_traj) - 1):
                        pt1 = self.scale_point(active_traj[j])
                        pt2 = self.scale_point(active_traj[j + 1])
                        cv2.line(overlay, pt1, pt2, highlighted_color.tolist(), 4)
                        
                        last_point = self.scale_point(active_traj[-1])
                        cv2.putText(overlay, str(traj_id + 1), last_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 146, 35), 1, lineType = cv2.LINE_AA)
            except Exception as e:
                log_error(f"Error processing trajectory {traj_id} at frame {frame_number}: {e}")

        log_debug(f"Returning overlay for frame {frame_number}, np.sum: {np.sum(overlay)}")
        return overlay
    
    def _preload_future_frames(self, current_frame):
        """Preloads a rolling buffer of future frames"""
        selected_trajectory = self.trajectory_manager.get_selected_trajectory()

        # If a trajectory is selected, stop preloading and clear outdated preloaded frames
        if selected_trajectory is not None:
            log_debug("Preloading halted due to active selection.")
            self.overlay_cache.clear() 
            return  

        max_preload = min(self.total_frames, current_frame + self.cache_size)

        for frame_idx in range(current_frame + 1, max_preload):
            if frame_idx not in self.overlay_cache:
                self.overlay_cache[frame_idx] = self._generate_overlay(frame_idx)

        # Keep only the most recent cache_size frames
        keys = list(self.overlay_cache.keys())
        if len(keys) > self.cache_size:
            for key in keys[:-self.cache_size]:  
                del self.overlay_cache[key]

        log_debug(f"Preloaded frames up to {max_preload}, current frame: {current_frame}")

    def scale_point(self, point):
        try:
            x, y = point
            scaled_x = int(x * self.video_width) if x <= 1 else int(round(x))
            scaled_y = int(y * self.video_height) if y <= 1 else int(round(y))
            return max(0, min(self.video_width - 1, scaled_x)), max(0, min(self.video_height - 1, scaled_y))
        except Exception as e:
            log_error(f"Error scaling point {point}: {e}")
            return (0, 0)

    def update_frame(self, frame_number):
        """Ensures the thread starts only if necessary."""
        if self.frame_number == frame_number:
            return
        
        log_debug(f"Updating frame to {frame_number}")
        self.frame_number = frame_number

        if frame_number in self.overlay_cache:
            overlay = self.overlay_cache[frame_number]
        else:
            overlay = self._generate_overlay(frame_number)
            self.overlay_cache[frame_number] = overlay

        self.update_overlay.emit(overlay)

    def stop(self):
        """Gracefully stops the thread."""
        if self.isRunning():  
            self.running = False
            self.wait()  
            self.quit()
        log_info("TrajectoryWorker stopped.")
