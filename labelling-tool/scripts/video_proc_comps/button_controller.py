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
        self.lock_first_selection = False
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
        log_info("Relabel button was pushed.")
        
        self.trajectory_manager.isDrawing = False
        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 1
        self.highlight_selected_button(0)

        PersistentMessageBox.show_message(
            self.trajectory_controls, "relabel_trajectory_start",
                "Information",
                "Select the trajectory you want to relabel - input manually in the input field or click directly on the trajectory you want to relabel - and click the Select button."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_missing_clicked(self):
        log_info("Missing button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 2
        self.highlight_selected_button(1)

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_break_clicked(self):
        log_info("Break button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 3
        self.highlight_selected_button(2)
        
        PersistentMessageBox.show_message(
            self.trajectory_controls, "break_trajectory",
                "Information",
                "In the trajectory you want to break, click directly on point you want it to break.\n"
                "Click Apply to finalize the action."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_join_clicked(self):
        log_info("Join button was pushed")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 4
        self.highlight_selected_button(3)
        
        PersistentMessageBox.show_message(
            self.trajectory_controls, "join_trajectory",
                "Information",
                "Input the trajectory ID manually in the input field and click Select. \n"
                "Next, input the second trajectory (to merge into the first one) and click select. Click Apply to finalize the action."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
        
    def on_delete_clicked(self):
        log_info("Delete button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()
        
        self.mode = 5
        self.highlight_selected_button(4)

        PersistentMessageBox.show_message(
            self.trajectory_controls, "delete_trajectory",
                "Information",
                "Select the trajectory you want to delete - input manually in the input field or click directly on the trajectory you want to delete - and click the Select button\n."
                "Click Apply to finalize the action."
        )
        self.trajectory_manager.ID_selected.emit(1)
    
    def on_disentangle_clicked(self):
        log_info("Disentangle button was pushed.")

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.trajectory_manager.clear_selection()

        self.mode = 6
        self.highlight_selected_button(5)

        PersistentMessageBox.show_message(
            self.trajectory_controls, "disentangle_trajectory",
                "Information",
                "Select the trajectory you want to disentangle:\n"
                "Trajectory 1 - Input manually in the input field or click directly on the trajectory you want to relabel - and click the Select button.\n"
                "Trajectory 2 - ONLY click directly in the 2nd trajectory on the point you want to disentangle - and click the Select button.\n"
                "Click Apply to finalize the action."
        )

        self.trajectory_manager.ID_selected.emit(1)
        self.startFrame = self.video_player.current_frame
    
    def on_undo_clicked(self):
        log_info("Undo button was pushed.")
        self.trajectory_manager.clear_selection()
        self.cancel_operation = True

        self.trajectory_manager.ID_selected.emit(1)
        self.undo_func()

    def cancel_operation_func(self):
        """Cancel the current operation."""
        if self.mode in [1, 2]:
            for item in self.trajectory_click_handler.graphics_scene.items():
                if isinstance(item, QGraphicsEllipseItem):
                    if item.pen().color() == Qt.GlobalColor.red:
                        self.trajectory_click_handler.graphics_scene.removeItem(item)

        self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
        self.highlight_selected_button(self.prev_operation_btn)
        self.trajectory_click_handler.clear_highlight()
        self.mode = 0
        self.enable_all_buttons()
        self.trajectory_manager.isWaitingID = False
        self.trajectory_manager.isDrawing = False
        self.lock_first_selection = False
        self.cancel_operation = False
    
    def on_ID_selected(self):
        """Handles when ID selected (both QLineEdit and mouse click)."""
        if self.cancel_operation:
            self.cancel_operation_func()
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
            self.disable_buttons_by_mode()
            self.trajectory_manager.isWaitingID = True
        else:
            selected_IDs = self.trajectory_manager.get_selected_trajectory()

            if len(selected_IDs) == 1 and self.mode in [4, 6]:
                log_info("Click the second trajectory or input the ID.")
                self.trajectory_controls.create_trajID_input(self.trajectory_controls.labeling_layout, 2, self.mode)
                self.lock_first_selection = True
            else:
                log_info("Selecting trajectories was done.")
                self.trajectory_manager.isWaitingID = False

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
                QMessageBox.warning(self.trajectory_controls, "Warning", "A new trajectory has not been draw! After finished click on Apply, else cancel the action.")
        else:
            self.trajectory_controls.delete_trajID_input(self.trajectory_controls.labeling_layout, self.mode)
            self.highlight_selected_button(self.prev_operation_btn)
            self.trajectory_manager.updateFrame.emit(self.startFrame)
                
            self.mode = 0
            self.enable_all_buttons()
            self.lock_first_selection = False
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
        self.enable_all_buttons()

        self.lock_first_selection = False
        self.trajectory_manager.updateFrame.emit(self.startFrame)

    def check_humanID(self, humanID):
        """Check if the humanID exists in the current frame."""
        if humanID not in self.trajectory_manager.traj_starts:
            QMessageBox.warning(self.trajectory_controls, "Warning", f"Trajectory {humanID} does not exist in the current frame.")
            return False
        return True
    
    def clear(self):
        self.trajectory_manager.clear_selection()
        self.mode = 0
        return
    
    def relabel_func(self, humanID, startFrame, new_trajectories):
        """Relabel function: fix trajectory with drawed trajectory from startFrame."""
        if self.check_humanID(humanID) == False:
            self.clear()
            return

        traj_start = self.trajectory_manager.traj_starts[humanID]
        trajectories_old = self.trajectory_manager.trajectories[humanID]
        
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
        self.clear()
        
    def missing_func(self, startFrame, new_trajectories):
        """Missing function: add trajecotory with drawed trajectory."""
        new_trajID = self.human_config.newID_init()
        self.backup()
        self.trajectory_manager.set_newValues(new_trajID, startFrame, new_trajectories)
        
        log_info(f"Added missed person: {new_trajID}")
        self.clear()
    
    def break_func(self, humanID, startFrame):
        """Break function: break into 2 trajectory 
           (former ID is same as old one and latter one is added as new)."""
        if not self.check_humanID(humanID):
            self.clear()
            return
        
        if startFrame == None:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Please select the frame where you want to break the trajectory.")
            self.clear()
            return
        
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
        
        self.clear()
        
    def join_func(self, humanID1, humanID2):
        """Join function: join 2 trajectories into one.
           + Delete the trajectory data of latter one."""
        if not self.check_humanID(humanID1) or not self.check_humanID(humanID2):
            self.clear()
            return

        traj_start1 = self.trajectory_manager.traj_starts[humanID1]
        traj_start2 = self.trajectory_manager.traj_starts[humanID2]

        if traj_start2 < traj_start1:
            humanID1, humanID2 = humanID2, humanID1
            traj_start1, traj_start2 = traj_start2, traj_start1

        trajectories1 = self.trajectory_manager.trajectories[humanID1]
        trajectories2 = self.trajectory_manager.trajectories[humanID2]
        
        traj_end1 = traj_start1 + len(trajectories1) - 1
        traj_end2 = traj_start2 + len(trajectories2) - 1

        if traj_start2 <= traj_end1:

            if traj_end2 <= traj_end1:
                #QMessageBox.warning(self.trajectory_controls, "Warning", "Trajectory 2 is included in Trajectory 1.")
                log_info(f"Trajectory {humanID2} is included in Trajectory {humanID1}.")
                self.delete_func(humanID2)
                return

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
        self.clear()
        
    def delete_func(self, humanID):
        """Delete function: delete trajectory."""
        if not self.check_humanID(humanID):
            self.clear()
            return
        
        self.backup()
        self.trajectory_manager.remove_trajectory(humanID)

        log_info(f"Trajectory {humanID} was deleted.")
        self.clear()
    
    def disentangle_func(self, humanID1, humanID2, startFrame):
        """Disentangle function: swap two trajectories after the startFrame"""
        if startFrame == None:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Please select the frame where you want to break the trajectory.")
            self.clear()
            return
        
        if not self.check_humanID(humanID1) or not self.check_humanID(humanID2):
            self.clear()
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

        log_info(f"Trajectory {humanID1} and {humanID2} were disentangled.")
        self.clear()

    def undo_func(self):
        """Undo function: reverse human_config to former one."""
        if not self.human_config_backup:
            QMessageBox.warning(self.trajectory_controls, "Warning", "Nothing to undo! No changes have been done to undo.")
            self.clear()
            return
        
        self.highlight_selected_button(6)
        human_config_old = self.human_config_backup.pop(-1)

        self.human_config.dict = human_config_old.dict
        self.trajectory_manager.undo()
        
        self.trajectory_manager.clear_selection()
        self.highlight_selected_button(self.prev_operation_btn)

        log_info("Undo Complete.")
        self.clear()
        
    def backup(self):
        """Make backup of human_config for Undo."""
        if len(self.human_config_backup) >= self.trajectory_manager.backup_limit:
            self.human_config_backup.pop(0)
        self.human_config_backup.append(deepcopy(self.human_config))
        self.trajectory_manager.backup()

    def highlight_selected_button(self, button_idx):
        self.put_back_to_blue()

        if not self.cancel_operation and self.mode != 0:
            self.trajectory_controls.buttons[button_idx].setStyleSheet("""
                QPushButton {
                    background-color: orange;
                }

                QPushButton:disabled {
                    background-color: lightgray;
                }
            """)
            self.prev_operation_btn = button_idx
        else:
            self.put_back_to_blue()

    def put_back_to_blue(self):
        for i in range(len(self.trajectory_controls.buttons) - 1):
            self.trajectory_controls.buttons[i].setStyleSheet("""
                QPushButton {
                    background-color: #0078D7;
                }

                QPushButton:disabled {
                    background-color: lightgray;
                }
            """)

    def disable_buttons_by_mode(self):
        for btn_idx in range(len(self.trajectory_controls.buttons)):
            if self.mode == btn_idx + 1:
                self.trajectory_controls.buttons[btn_idx].setEnabled(True)
                continue
            self.trajectory_controls.buttons[btn_idx].setEnabled(False)

    def enable_all_buttons(self):
        for btn in self.trajectory_controls.buttons:
            btn.setEnabled(True)