import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QBrush
from .playback_mode import PlaybackMode
from utils.logging_utils import log_info, log_error, log_debug
from PyQt6.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsEllipseItem

class TrajectoryClickHandler(QGraphicsView):
    def __init__(self, trajectory_manager, scene, trajectory_overlay, color_generator, parent = None):
        super().__init__(parent)
        self.current_frame = 0
        self.main_window = None
        self.fully_loaded = False
        self.graphics_scene = scene
        self.color_generator = color_generator
        self.trajectory_manager = trajectory_manager 
        self.trajectory_overlay = trajectory_overlay

    def mousePressEvent(self, event):
        """Handles mouse click events and selects the closest trajectory."""
        # getting main_window class
        self.fully_loaded = self.parent().parent().parent().parent().parent().fully_loaded

        if self.fully_loaded: 
            scene_pos = self.mapToScene(event.position().toPoint())
            log_debug(f"User clicked at Scene Coordinates: ({scene_pos.x()}, {scene_pos.y()})")

            if self.trajectory_manager.isDrawing:
                self.draw_red_circles(scene_pos)

                # Drawing finish when clicking at blank.
                for item in self.graphics_scene.items(scene_pos):
                    if isinstance(item, QGraphicsPixmapItem):
                        pixmap_item = item
                        pixmap_rect = pixmap_item.pixmap().rect()
                        item_pos = pixmap_item.mapFromScene(scene_pos)

                        orig_x = int((item_pos.x() / pixmap_item.boundingRect().width()) * pixmap_rect.width())
                        orig_y = int((item_pos.y() / pixmap_item.boundingRect().height()) * pixmap_rect.height())
                        log_info(f"Mapped pixel coordinates: (x = {orig_x}, y = {orig_y})")
                        
                        self.trajectory_manager.store_newTrajectory(orig_x, orig_y)
                        self.trajectory_manager.updateFrame.emit(self.current_frame + self.trajectory_manager.frame_skip)
                        return
                return

            if self.trajectory_overlay is None:
                log_error("ERROR: Click handler has no overlay assigned!")
                return

            found_trajectory = False
            if self.trajectory_manager.isWaitingID:
                for item in self.graphics_scene.items(scene_pos):
                    if isinstance(item, QGraphicsPixmapItem):
                        pixmap_item = item
                        pixmap_rect = pixmap_item.pixmap().rect()
                        item_pos = pixmap_item.mapFromScene(scene_pos)

                        orig_x = int((item_pos.x() / pixmap_item.boundingRect().width()) * pixmap_rect.width())
                        orig_y = int((item_pos.y() / pixmap_item.boundingRect().height()) * pixmap_rect.height())
                        log_info(f"Mapped pixel coordinates: (x = {orig_x}, y = {orig_y})")

                        selected_traj_id = self.get_trajectory_from_overlay(orig_x, orig_y)  

                        if selected_traj_id is not None:
                            log_info(f"Selected trajectory ID: {selected_traj_id}")

                            trajectory = np.array(self.trajectory_manager.trajectories[selected_traj_id])
                            select_idx = np.argmin(np.linalg.norm(trajectory - [orig_x, orig_y], axis = 1))
                            select_frame = int(select_idx + self.trajectory_manager.traj_starts[selected_traj_id])
                            
                            click_limit = 2 if self.get_ui_class().button_controller.mode in [4, 6] else 1
                            if len(self.trajectory_manager.get_selected_trajectory()) >= click_limit:
                                self.clear_highlight()

                            self.trajectory_manager.set_selected_trajectory(selected_traj_id, select_frame)

                            if self.get_ui_class().button_controller.mode != 2:
                                self.write_traj_id_on_input(str(selected_traj_id + 1))
                                self.refresh_frame_if_paused()

                            for item in self.graphics_scene.items():
                                if isinstance(item, QGraphicsEllipseItem):
                                    self.graphics_scene.removeItem(item)
                            
                            found_trajectory = True
                            break
                    
                if not found_trajectory and self.get_ui_class().button_controller.mode in [4, 6] and not self.get_ui_class().button_controller.lock_first_selection:
                    log_info("No trajectory selected, clearing highlight.")
                    self.clear_highlight()

        super().mousePressEvent(event)

    def get_trajectory_from_overlay(self, x, y):
        if self.trajectory_overlay is None:
            log_info("Trajectory overlay is None")
            return None
        
        if not (0 <= x < self.trajectory_overlay.shape[1] and 0 <= y < self.trajectory_overlay.shape[0]):
            log_info(f"Click out of bounds: (x={x}, y={y}), Overlay size: {self.trajectory_overlay.shape}")
            return None

        TOLERANCE = 10
        SEARCH_RADIUS = 10

        for delta_y in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            for delta_x in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                new_y, new_x = (y + delta_y), x + delta_x
                if 0 <= new_x < self.trajectory_overlay.shape[1] and 0 <= new_y < self.trajectory_overlay.shape[0]:
                    color_at_point = self.trajectory_overlay[new_y, new_x]

                    for traj_id, color in self.color_generator.get_active_colors().items():
                        if np.linalg.norm(color_at_point.astype(int) - color.astype(int)) < TOLERANCE:
                            log_info(f"Matched trajectory {traj_id} at offset ({delta_x}, {delta_y})")
                            return traj_id
                        
        log_info("No trajectory found in neighborhood")
        return None

    def clear_highlight(self):
        """Removes the highlight when clicking outside a trajectory."""
        log_info("Clearing highlight")

        if self.trajectory_manager.get_selected_trajectory() is None:
            log_debug("No trajectory was highlighted, skipping clear.")
            return
        
        self.trajectory_manager.clear_selection()
        self.refresh_frame_if_paused()

    def refresh_frame_if_paused(self):
        if self.parent().playback_mode == PlaybackMode.STOPPED:
            log_debug("Video is paused, refreshing frame.")

            if self.parent().current_frame < self.parent().total_frames - 1:
                temp_frame = self.parent().current_frame + 1  
            else:
                temp_frame = self.parent().current_frame - 1  

            self.parent().show_frame_at(temp_frame) 
            self.parent().show_frame_at(self.parent().current_frame) 

    def get_ui_class(self):
        video_controls = self.parent().parent()
        main_window = video_controls.parent().parent().parent()
        tab_dialog = main_window.tab_dialog
        trajectory_controls = tab_dialog.trajectory_controls

        return trajectory_controls

    def write_traj_id_on_input(self, traj_id):
        trajectory_controls = self.get_ui_class()
        
        h_layout = trajectory_controls.labeling_layout.itemAt(1).layout()
        line_edit = h_layout.itemAt(0).widget() 
        line_edit.setText(traj_id)

    def draw_red_circles(self, scene_pos):
        click_marker = QGraphicsEllipseItem(scene_pos.x() - 5, scene_pos.y() - 5, 10, 10)
        click_marker.setPen(QPen(Qt.GlobalColor.red, 2))
        click_marker.setBrush(QBrush(Qt.GlobalColor.red))
        self.graphics_scene.addItem(click_marker)
        return click_marker
