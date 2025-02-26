import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QBrush
from .playback_mode import PlaybackMode
from utils.logging_utils import log_info, log_error
from PyQt6.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QGraphicsEllipseItem

class TrajectoryClickHandler(QGraphicsView):
    def __init__(self, trajectory_manager, scene, trajectory_overlay, color_generator, parent = None):
        super().__init__(parent)
        self.current_frame = 0
        self.graphics_scene = scene
        self.dual_selection_enabled = False
        self.color_generator = color_generator
        self.trajectory_manager = trajectory_manager 
        self.trajectory_overlay = trajectory_overlay

    def mousePressEvent(self, event):
        """Handles mouse click events and selects the closest trajectory."""
        scene_pos = self.mapToScene(event.position().toPoint())
        log_info(f"User clicked at Scene Coordinates: ({scene_pos.x()}, {scene_pos.y()})")

        # Draw a red marker at the clicked position
        click_marker = QGraphicsEllipseItem(scene_pos.x() - 5, scene_pos.y() - 5, 10, 10)
        click_marker.setPen(QPen(Qt.GlobalColor.red, 2))
        click_marker.setBrush(QBrush(Qt.GlobalColor.red))
        self.graphics_scene.addItem(click_marker)
                
        if self.trajectory_manager.isDrawing:
            # Drawing finish when clicking at blank.
            for item in self.graphics_scene.items(scene_pos):
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap_item = item
                    pixmap_rect = pixmap_item.pixmap().rect()
                    item_pos = pixmap_item.mapFromScene(scene_pos)

                    orig_x = int((item_pos.x() / pixmap_item.boundingRect().width()) * pixmap_rect.width())
                    orig_y = int((item_pos.y() / pixmap_item.boundingRect().height()) * pixmap_rect.height())
                    log_info(f"Mapped pixel coordinates: (x={orig_x}, y={orig_y})")
                    
                    self.trajectory_manager.store_newTrajectory(orig_x, orig_y)
                    self.trajectory_manager.updateFrame.emit(self.current_frame + 30)
                    return
            self.trajectory_manager.drawingFinished.emit(1)
            print('Blank clicked.')
            for item in self.graphics_scene.items():
                if isinstance(item, QGraphicsEllipseItem):
                    self.graphics_scene.removeItem(item)
            return
                    
        if self.trajectory_overlay is None:
            log_error("ERROR: Click handler has no overlay assigned!")
            return

        found_trajectory = False

        for item in self.graphics_scene.items(scene_pos):
            if isinstance(item, QGraphicsPixmapItem):
                pixmap_item = item
                pixmap_rect = pixmap_item.pixmap().rect()
                item_pos = pixmap_item.mapFromScene(scene_pos)

                orig_x = int((item_pos.x() / pixmap_item.boundingRect().width()) * pixmap_rect.width())
                orig_y = int((item_pos.y() / pixmap_item.boundingRect().height()) * pixmap_rect.height())
                log_info(f"Mapped pixel coordinates: (x={orig_x}, y={orig_y})")

                selected_traj_id = self.get_trajectory_from_overlay(orig_x, orig_y)  

                if selected_traj_id is not None:
                    log_info(f"Selected trajectory ID: {selected_traj_id}")
                    if self.trajectory_manager.isWaitingID:
                        self.trajectory_manager.set_selected_trajectory(selected_traj_id)
                        self.trajectory_manager.ID_selected.emit(1)
                        for item in self.graphics_scene.items():
                            if isinstance(item, QGraphicsEllipseItem):
                                self.graphics_scene.removeItem(item)
                    self.highlight_selected_trajectory(selected_traj_id)
                    
                    found_trajectory = True
                    break

        log_info(f"[DEBUG] Found trajectory? {found_trajectory}") 
            
        if not found_trajectory:
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

        TOLERANCE = 30
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
    
    def highlight_selected_trajectory(self, traj_id):
        log_info(f"Highlighting trajectory {traj_id}")
        self.trajectory_manager.set_selected_trajectory(traj_id)
        self.refresh_frame_if_paused()

    def clear_highlight(self):
        """Removes the highlight when clicking outside a trajectory."""
        log_info("[DEBUG] Clearing highlight")
        log_info(f"[DEBUG] Current selected trajectory: {self.trajectory_manager.get_selected_trajectory()}")

        if self.trajectory_manager.get_selected_trajectory() is None:
            log_info("[DEBUG] No trajectory was highlighted, skipping clear.")
            return
        
        self.trajectory_manager.clear_selection()
        self.refresh_frame_if_paused()

    def refresh_frame_if_paused(self):
        if self.parent().playback_mode == PlaybackMode.STOPPED:
            log_info("[DEBUG] Video is paused, refreshing frame.")

            if self.parent().current_frame < self.parent().total_frames - 1:
                temp_frame = self.parent().current_frame + 1  
            else:
                temp_frame = self.parent().current_frame - 1  

            self.parent().show_frame_at(temp_frame) 
            self.parent().show_frame_at(self.parent().current_frame) 
