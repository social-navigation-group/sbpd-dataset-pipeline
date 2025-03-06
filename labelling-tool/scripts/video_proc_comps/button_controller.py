from copy import deepcopy
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox, QGraphicsEllipseItem

from utils.persistent_message_box import PersistentMessageBox
from utils.logging_utils import log_info, log_error, log_debug

class ButtonController():
    def __init__(self, trajectory_controls):
        self.startFrame = 0
        self.prev_operation_btn = 0
        self.cancel_operation = False
        self.trajectory_controls = trajectory_controls
        self.video_player = self.trajectory_controls.video_player    
        self.human_config = self.video_player.human_config
        self.trajectory_click_handler = self.video_player.view
        self.trajectory_manager = self.video_player.trajectory_manager
        
        self.trajectory_manager.drawingFinished.connect(self.on_drawFinished)
        self.trajectory_manager.ID_selected.connect(self.on_ID_selected)
        
        self.mode = 0
        self.human_config_backup = []
        """0: Default, 1: Relabel, 2: Missing, 3: Break, 4: Join, 
           5: Delete, 6: Disentagle"""

    def on_relabel_clicked(self):
        """Relabel Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Relabel button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 1
        self.highlight_selected_button(0)

        PersistentMessageBox.show_message(
            self.trajectory_controls, "relabel_trajectory_start",
                "Information",
                "Select the trajectory you want to relabel."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_missing_clicked(self):
        """Missing Clicking Fuction: Starts drawing new trajectory."""
        log_info("Missing button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 2
        self.highlight_selected_button(1)

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_break_clicked(self):
        """Break Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Break button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 3
        self.highlight_selected_button(2)
        
        PersistentMessageBox.show_message(
            self.trajectory_controls, "break_trajectory",
                "Information",
                "Move through the frames to find the point where the trajectory should break. Clicking Apply to finalize the action."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_join_clicked(self):
        """Join Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID(1) input."""
        log_info("Join button was pushed")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 4
        self.highlight_selected_button(3)
        
        PersistentMessageBox.show_message(
            self.trajectory_controls, "join_trajectory",
                "Information",
                "Select the first trajectory.\n"
                "Next, select the second trajectory to merge into the first one. Then just click the Apply button."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
        
    def on_delete_clicked(self):
        """Delete Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Delete button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()
        
        self.mode = 5
        self.highlight_selected_button(4)

        PersistentMessageBox.show_message(
            self.trajectory_controls, "delete_trajectory",
                "Information",
                "Select the trajectory that you want to delete and apply the change."
        )
        self.trajectory_manager.ID_selected.emit(1)
    
    def on_disentangle_clicked(self):
        """Disentangle Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID(1) input."""
        log_info("Disentangle button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 6
        self.highlight_selected_button(5)

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_undo_clicked(self):
        """Undo Clicking Fuction: Reverses human_config to former one."""
        log_info("Undo button was pushed.")
        self.trajectory_manager.clear_selection()
        self.cancel_operation = True

        self.trajectory_manager.ID_selected.emit(1)
        self.undo_func()
    
    def on_ID_selected(self):
        """Handles when ID selected (both QLineEdit and mouse click)."""
        if self.cancel_operation:
            if self.mode == 2:
                for item in self.trajectory_click_handler.graphics_scene.items():
                    if isinstance(item, QGraphicsEllipseItem):
                        if item.pen().color() == Qt.GlobalColor.red:
                            self.trajectory_click_handler.graphics_scene.removeItem(item)

            self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
            self.highlight_selected_button(self.prev_operation_btn)
            self.trajectory_click_handler.clear_highlight()
            self.mode = 0
            self.trajectory_manager.isWaitingID = False
            self.trajectory_manager.isDrawing = False
            self.cancel_operation = False
            return

        if self.trajectory_manager.get_selected_trajectory() == []:
            if self.mode == 2:
                self.trajectory_controls.create_trajID_input(self.trajectory_controls.labeling_layout, 0, self.mode)
                self.trajectory_manager.isDrawing = True
                self.trajectory_controls.labeling_layout.itemAt(0).widget().setEnabled(True)
                PersistentMessageBox.show_message(
                    self.trajectory_controls, "missing_trajectory",
                        "Information",
                        "Draw the desired trajectory by clicking in the video. For better accuracy, focus on the person's feet (when possible)."
                )
            else:
                log_info("Click a trajectory or input trajectory ID.")
                self.trajectory_controls.create_trajID_input(self.trajectory_controls.labeling_layout, 1, self.mode)
            self.trajectory_manager.isWaitingID = True
        else:
            selected_IDs = self.trajectory_manager.get_selected_trajectory()
            print(selected_IDs)
            # selected_ID = [id[0] for id in selected_ID]

            if len(selected_IDs) == 1 and self.mode in [4, 6]:
                log_info("Click the second trajectory or input the ID.")
                self.trajectory_controls.create_trajID_input(self.trajectory_controls.labeling_layout, 2, self.mode)
            else:
                log_info("Selecting trajectories was done.")
                self.trajectory_manager.isWaitingID = False
                # self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
                # edit_frame = self.video_player.current_frame

                # Relabel
                # Start drawing trajectories.
                if self.mode == 1:
                    self.trajectory_manager.isDrawing = True
                    PersistentMessageBox.show_message(
                        self.trajectory_controls, "relabel_trajectory",
                            "Information",
                            "Redraw the desired trajectory by clicking in the video. For better accuracy, focus on the person's feet (when possible)."
                    )
            
                # Break
                elif self.mode == 3:
                    self.break_func(selected_IDs[0][0], selected_IDs[0][1])
                
                # Join
                elif self.mode == 4:
                    self.join_func(selected_IDs[0][0], selected_IDs[1][0])
                    
                # Delete
                elif self.mode == 5:
                    self.delete_func(selected_IDs[0][0])
                
                # Disentangle
                elif self.mode == 6:
                    self.disentangle_func(selected_IDs[0][0], selected_IDs[1][0], selected_IDs[1][1])

    def on_select_pressed(self):
        line_edit = self.trajectory_controls.labeling_layout.itemAt(1).layout().itemAt(0).widget() 
        select_btn = self.trajectory_controls.labeling_layout.itemAt(1).layout().itemAt(1).widget() 
        # active_trajectories = self.trajectory_manager.get_active_trajectories(self.startFrame)

        try:
            input_text = line_edit.text().strip() 
            if not input_text:
                raise ValueError("Input is empty.") 
            
            selected_ID = int(input_text) - 1  
 
            self.trajectory_manager.set_selected_trajectory(selected_ID, self.startFrame)
            self.trajectory_click_handler.refresh_frame_if_paused()
            line_edit.setEnabled(False)
            select_btn.setEnabled(False)

            self.trajectory_controls.labeling_layout.itemAt(2).widget().setEnabled(True)
            self.trajectory_manager.ID_selected.emit(1)
            
        except ValueError as e:
            QMessageBox.warning(self.trajectory_controls, "Input Error", f"Please input a valid trajectory ID.\n{str(e)}")
            log_error("Please input trajectory ID.")

    def on_cancel_pressed(self):
        self.cancel_operation = True
        self.trajectory_manager.ID_selected.emit(1)

    def on_apply_pressed(self):
        if self.mode in [1, 2]:
            graphic_scene = self.trajectory_click_handler.graphics_scene
            red_circle_found = False

            for item in graphic_scene.items():
                if isinstance(item, QGraphicsEllipseItem):
                    if item.pen().color() == Qt.GlobalColor.red:
                        red_circle_found = True

                        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
                        self.trajectory_manager.drawingFinished.emit(1)

                        graphic_scene.removeItem(item)
                        self.highlight_selected_button(self.prev_operation_btn)

            if not red_circle_found:
                QMessageBox.warning(self.trajectory_controls, "Warning", "A new trajectory as not been draw! After finished click on Apply, else cancel the action.")
        else:
            self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
            self.highlight_selected_button(self.prev_operation_btn)

            if self.mode in [3, 5]:
                self.trajectory_manager.updateFrame.emit(self.startFrame)
                
            self.mode = 0
            self.highlight_selected_button(self.prev_operation_btn)

    def on_drawFinished(self):
        """Functions for after drawing trajecotries."""
        self.trajectory_manager.isDrawing = False
        selected_ID = self.trajectory_manager.get_selected_trajectory()
        selected_ID = [id[0] for id in selected_ID]
        
        # Relabel
        if self.mode == 1:
            self.relabel_func(selected_ID[0], self.startFrame, self.trajectory_manager.trajectory_now)
            
        # Missing
        elif self.mode == 2:
            self.missing_func(self.startFrame, self.trajectory_manager.trajectory_now)
        
        self.trajectory_manager.trajectory_now = []
        self.trajectory_manager.updateFrame.emit(self.startFrame)
    
    def relabel_func(self, humanID, startFrame, new_trajectories):
        """Relabel function: fix trajectory with drawed trajectory from startFrame."""
        traj_start = self.trajectory_manager.traj_starts[humanID]
        trajectories_old = self.trajectory_manager.trajectories[humanID]
        
        # new_trajectories = trajectories_old[:(startFrame - traj_start)] + new_trajectories
        if startFrame > traj_start:
            traj_end = traj_start + len(trajectories_old)
            new_traj_end = startFrame + len(new_trajectories)
            tmp_traj = []

            if new_traj_end < traj_end:
                tmp_traj = trajectories_old[(new_traj_end - traj_end):]
            new_trajectories = trajectories_old[:(startFrame - traj_start)] + new_trajectories + tmp_traj
        else:
            tmp_traj = []
            new_traj_end = startFrame + len(new_trajectories)
            start_pos = new_trajectories[-1]
            end_pos = trajectories_old[0]

            if new_traj_end  < traj_start:
                for i in range(traj_start - new_traj_end - 1):
                    tmp_pos0 = start_pos[0] + (end_pos[0] - start_pos[0]) * (i + 1) / (traj_start - new_traj_end)
                    tmp_pos1 = start_pos[1] + (end_pos[1] - start_pos[1]) * (i + 1) / (traj_start - new_traj_end)
                    tmp_traj.append([tmp_pos0, tmp_pos1])

            new_trajectories = new_trajectories + tmp_traj + trajectories_old[(startFrame + len(new_trajectories) - traj_start):]
            traj_start = startFrame
        
        self.backup()
        self.trajectory_manager.set_newValues(humanID, traj_start, new_trajectories)
        
        log_info(f"Relabel Complete: {humanID}")
        
        self.trajectory_manager.clear_selection()
        self.mode = 0
        
    def missing_func(self, startFrame, new_trajectories):
        """Missing function: add trajecotory with drawed trajectory."""
        new_trajID = self.human_config.newID_init()
        self.backup()
        self.trajectory_manager.set_newValues(new_trajID, startFrame, new_trajectories)
        
        log_info(f"Added missed person: {new_trajID}")
        self.trajectory_manager.clear_selection()
        self.mode = 0
    
    def break_func(self, humanID, startFrame):
        """Break function: break into 2 trajectory 
           (former ID is same as old one and latter one is added as new)."""
        if startFrame == None:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Please select the frame where you want to break the trajectory.")
            return
        
        print(f"humanID in break_func: {type(humanID)}")
        print(f"startFrame in break_func: {startFrame}")
        
        traj_start_old = self.trajectory_manager.traj_starts[humanID]
        trajectories_old = self.trajectory_manager.trajectories[humanID]
        
        traj_start_new1 = traj_start_old
        traj_start_new2 = startFrame + 1
        
        trajectories_new1 = trajectories_old[:(startFrame - traj_start_old)]
        trajectories_new2 = trajectories_old[(startFrame - traj_start_old + 1):]
        
        self.backup()
        self.trajectory_manager.set_newValues(humanID, traj_start_new1, trajectories_new1)
        new_trajID = self.trajectory_manager.add_trajectory(trajectories_new2, traj_start_new2)
        
        log_info(f"Trajectory {humanID} was divided.")
        log_debug(f"{traj_start_old} - {startFrame - 1} -> ID: {humanID}")
        log_debug(f"{startFrame} - {startFrame + len(trajectories_new2) - 1} -> ID: {new_trajID}")
        
        self.trajectory_manager.clear_selection()
        # self.mode = 0
        
    def join_func(self, humanID1, humanID2):
        """Join function: join 2 trajectories into one.
           + Delete the trajectory data of latter one."""
        traj_start1 = self.trajectory_manager.traj_starts[humanID1]
        traj_start2 = self.trajectory_manager.traj_starts[humanID2]

        if traj_start2 < traj_start1:
            humanID1, humanID2 = humanID2, humanID1
            traj_start1, traj_start2 = traj_start2, traj_start1

        trajectories1 = self.trajectory_manager.trajectories[humanID1]
        trajectories2 = self.trajectory_manager.trajectories[humanID2]
        
        # trajectories_new = trajectories1[:startFrame - traj_start1] + trajectories2[startFrame - traj_start2:]
        traj_end1 = traj_start1 + len(trajectories1) - 1

        if traj_start2 <= traj_end1:
            tmp_traj_len = traj_end1 - traj_start2 + 1
            tmp_traj = []

            for i in range(tmp_traj_len):
                wt = 1 - (i + 1) / (tmp_traj_len + 1)
                tmp_pos0 = trajectories1[-tmp_traj_len + i][0] * wt + trajectories2[i][0] * (1 - wt)
                tmp_pos1 = trajectories1[-tmp_traj_len + i][1] * wt + trajectories2[i][1] * (1 - wt)
                tmp_traj.append([tmp_pos0, tmp_pos1])
            trajectories_new = trajectories1[:-tmp_traj_len] + tmp_traj + trajectories2[tmp_traj_len:]
        else:
            tmp_traj_len = traj_start2 - traj_end1 - 1
            tmp_traj = []
            start_pos = trajectories1[-1]
            end_pos = trajectories2[0]

            for i in range(tmp_traj_len):
                tmp_pos0 = start_pos[0] + (end_pos[0] - start_pos[0]) * (i + 1) / (tmp_traj_len + 1)
                tmp_pos1 = start_pos[1] + (end_pos[1] - start_pos[1]) * (i + 1) / (tmp_traj_len + 1)
                tmp_traj.append([tmp_pos0, tmp_pos1])
            trajectories_new = trajectories1 + tmp_traj + trajectories2
        
        self.backup()
        self.trajectory_manager.set_newValues(humanID1, traj_start1, trajectories_new)
        self.trajectory_manager.remove_trajectory(humanID2)
        
        log_info(f"No.{humanID1} and No.{humanID2} are joined into No.{humanID1}.")
        
        self.trajectory_manager.clear_selection()
        self.mode = 0
        
    def delete_func(self, humanID):
        """Delete function: delete trajectory."""
        self.backup()
        self.trajectory_manager.remove_trajectory(humanID)
        
        self.trajectory_manager.clear_selection()
        # self.mode = 0
    
    def disentangle_func(self, humanID1, humanID2, startFrame):
        """Disentangle function: swap two trajectories after the startFrame"""
        if startFrame == None:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Please select the frame where you want to break the trajectory.")
            return
        
        traj_start1 = self.trajectory_manager.traj_starts[humanID1]
        trajectories1 = self.trajectory_manager.trajectories[humanID1]
        
        traj_start2 = self.trajectory_manager.traj_starts[humanID2]
        trajectories2 = self.trajectory_manager.trajectories[humanID2]
        
        trajectories1_new = trajectories1[:(startFrame - traj_start1)] + trajectories2[(startFrame - traj_start2):]
        trajectories2_new = trajectories2[:(startFrame - traj_start2)] + trajectories1[(startFrame - traj_start1):]
        
        self.backup()
        self.trajectory_manager.set_newValues(humanID1, traj_start1, trajectories1_new)
        self.trajectory_manager.set_newValues(humanID2, traj_start2, trajectories2_new)
        
        self.trajectory_manager.clear_selection()
        self.mode = 0

    def undo_func(self):
        """Undo function: reverse human_config to former one."""
        if not self.human_config_backup:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Nothing to undo! No changes have been done to undo.")
            self.trajectory_manager.clear_selection()
            self.mode = 0
            return
        
        self.highlight_selected_button(6)
        human_config_old = self.human_config_backup.pop(-1)

        self.human_config.dict = human_config_old.dict
        self.trajectory_manager.undo()
        
        self.trajectory_manager.clear_selection()
        self.highlight_selected_button(self.prev_operation_btn)

        self.mode = 0
        self.trajectory_manager.updateFrame.emit(self.startFrame)
        
    def backup(self):
        """Make backup of human_config for Undo."""
        if len(self.human_config_backup) >= self.trajectory_manager.backup_limit:
            self.human_config_backup.pop(0)
        self.human_config_backup.append(deepcopy(self.human_config))
        self.trajectory_manager.backup()

    def highlight_selected_button(self, button_idx):
        self.put_back_to_blue()

        if not self.cancel_operation and self.mode != 0:
            self.trajectory_controls.buttons[button_idx].setStyleSheet("background-color: orange;")
            self.prev_operation_btn = button_idx
        else:
            self.put_back_to_blue()

    def put_back_to_blue(self):
        for i in range(len(self.trajectory_controls.buttons) - 1):
            self.trajectory_controls.buttons[i].setStyleSheet("background-color: #0078D7;")