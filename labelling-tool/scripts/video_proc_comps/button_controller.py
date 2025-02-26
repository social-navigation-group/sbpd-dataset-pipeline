from copy import deepcopy
from utils.logging_utils import log_info, log_warning, log_error

class ButtonController():
    def __init__(self, trajectoryControls):
        self.trajectoryControls = trajectoryControls
        self.video_player = self.trajectoryControls.video_player    
        self.human_config = self.video_player.human_config
        self.trajectory_manager = self.video_player.trajectory_manager
        
        self.trajectory_manager.drawingFinished.connect(self.on_drawFinished)
        self.trajectory_manager.ID_selected.connect(self.on_ID_selected)
        
        self.human_config_backup = []
        self.mode = 0
        """0: Default, 1: Relabel, 2: Missing, 3: Break, 4: Join, 
           5: Delete, 6: Disentagle"""
    
    def on_Relabel_clicked(self):
        """Relabel Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Relabel button was pushed.")
        
        self.trajectory_manager.clear_selection()
        self.trajectory_manager.ID_selected.emit(1)
        
        self.mode = 1
        self.startFrame = self.video_player.current_frame
        self.trajectory_manager.isWaitingID = True
    
    def on_Missing_clicked(self):
        """Missing Clicking Fuction: Starts drawing new trajectory."""
        log_info("Missing button was pushed.")
        
        self.trajectoryControls.delete_trajID_input(self.trajectoryControls.labeling_layout)
        self.trajectory_manager.clear_selection()
        
        self.mode = 2
        self.startFrame = self.video_player.current_frame
        self.trajectory_manager.isDrawing = True
    
    def on_Break_clicked(self):
        """Break Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Break button was pushed.")
        
        self.trajectory_manager.clear_selection()
        self.trajectory_manager.ID_selected.emit(1)
        
        self.startFrame = self.video_player.current_frame
        self.mode = 3
        self.trajectory_manager.isWaitingID = True
    
    def on_Join_clicked(self):
        """Join Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID(1) input."""
        log_info("Join button was pushed")
        
        self.trajectory_manager.clear_selection()
        self.trajectory_manager.ID_selected.emit(1)
        
        self.startFrame = self.video_player.current_frame
        self.mode = 4
        self.trajectory_manager.isWaitingID = True
        
    def on_Delete_clicked(self):
        """Delete Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID input."""
        log_info("Delete button was pushed.")
        
        self.trajectory_manager.clear_selection()
        self.trajectory_manager.ID_selected.emit(1)
        
        self.mode = 5
        self.trajectory_manager.isWaitingID = True
    
    def on_Disentangle_clicked(self):
        """Disentangle Clicking Fuction: Appears QLineEdit and starts waiting for trajectory ID(1) input."""
        log_info("Disentangle button was pushed.")
        
        self.trajectory_manager.clear_selection()
        self.trajectory_manager.ID_selected.emit(1)
        
        self.startFrame = self.video_player.current_frame
        self.mode = 6
        self.trajectory_manager.isWaitingID = True
    
    def on_Undo_clicked(self):
        """Undo Clicking Fuction: Reverses human_config to former one."""
        log_info("Undo button was pushed.")
        self.undo_func()
    
    def on_ID_selected(self):
        """Handles when ID selected (both QLineEdit and mouse click)."""
        if self.trajectory_manager.get_selected_trajectory() == []:
            log_info("Click a trajectory or input trajectory ID.")
            self.trajectoryControls.create_trajID_input(self.trajectoryControls.labeling_layout, 1)
            self.trajectory_manager.isWaitingID = True
        else:
            selected_IDs = self.trajectory_manager.get_selected_trajectory()
            if len(selected_IDs) == 1 and self.mode in [4, 6]:
                log_info("Click the second trajectory or input the ID.")
                self.trajectoryControls.create_trajID_input(self.trajectoryControls.labeling_layout, 2)
            
            else:
                log_info("Selecting trajectories was done.")
                self.trajectory_manager.isWaitingID = False
                self.trajectoryControls.delete_trajID_input(self.trajectoryControls.labeling_layout)

                # Relabel
                # Start drawing trajectories.
                if self.mode == 1:
                    self.trajectory_manager.isDrawing = True
                
                # Break
                elif self.mode == 3:
                    self.break_func(selected_IDs[0], self.startFrame)
                
                # Join
                elif self.mode == 4:
                    self.join_func(selected_IDs[0], selected_IDs[1], self.startFrame)
                    
                # Delete
                elif self.mode == 5:
                    self.delete_func(selected_IDs[0])
                
                # Disentangle
                elif self.mode == 6:
                    self.disentangle_func(selected_IDs[0], selected_IDs[1], self.startFrame)
                
                self.trajectory_manager.updateFrame.emit(self.startFrame)
            
    def on_select_pressed(self):
        """Function of pressing select: emit ID_selected"""
        line_edit = self.trajectoryControls.labeling_layout.itemAt(1).widget()
        try:
            selected_ID = int(line_edit.text()) - 1
            self.trajectory_manager.set_selected_trajectory(selected_ID)
            self.trajectory_manager.ID_selected.emit(1)
        except:
            log_error("Please input trajectory ID.")
    
    def on_drawFinished(self):
        """Functions for after drawing trajecotries."""
        self.trajectory_manager.isDrawing = False
        selected_ID = self.trajectory_manager.get_selected_trajectory()
        
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
        
        new_trajectories = trajectories_old[:(startFrame - traj_start)] + new_trajectories
        
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
        print(self.trajectory_manager.traj_starts)
        
        log_info(f"Added missed person: {new_trajID}")
        
        self.trajectory_manager.clear_selection()
        self.mode = 0
    
    def break_func(self, humanID, startFrame):
        """Break function: break into 2 trajectory 
           (former ID is same as old one and latter one is added as new)."""
        traj_start_old = self.trajectory_manager.traj_starts[humanID]
        trajectories_old = self.trajectory_manager.trajectories[humanID]
        
        traj_start_new1 = traj_start_old
        traj_start_new2 = startFrame
        
        trajectories_new1 = trajectories_old[:(startFrame - traj_start_old)]
        trajectories_new2 = trajectories_old[(startFrame - traj_start_old):]
        
        self.backup()
        self.trajectory_manager.set_newValues(humanID, traj_start_new1, trajectories_new1)
        new_trajID = self.trajectory_manager.add_trajectory(trajectories_new2, traj_start_new2)
        
        log_info(f"Trajectory {humanID} was divided.")
        log_info(f"{traj_start_old} - {startFrame - 1} -> ID: {humanID}")
        log_info(f"{startFrame} - {startFrame + len(trajectories_new2) - 1} -> ID: {new_trajID}")
        
        self.trajectory_manager.clear_selection()
        self.mode = 0
        
    def join_func(self, humanID1, humanID2, startFrame):
        """Join function: join 2 trajectories into one.
           + Delete the trajectory data of latter one."""
        traj_start1 = self.trajectory_manager.traj_starts[humanID1]
        traj_start2 = self.trajectory_manager.traj_starts[humanID2]
        trajectories1 = self.trajectory_manager.trajectories[humanID1]
        trajectories2 = self.trajectory_manager.trajectories[humanID2]
        
        trajectories_new = trajectories1[:startFrame - traj_start1] + trajectories2[startFrame - traj_start2:]
        
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
        self.mode = 0
    
    def disentangle_func(self, humanID1, humanID2, startFrame):
        """Disentangle function: swap two trajectories after the startFrame"""
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
            log_warning("Nothing to undo.")
            self.trajectory_manager.clear_selection()
            self.mode = 0
            return
        
        # update video with this new human_config
        human_config_old = self.human_config_backup.pop(-1)
        self.human_config.dict = human_config_old.dict
        self.trajectory_manager.undo()
        
        self.trajectory_manager.clear_selection()
        self.mode = 0
        
    def backup(self):
        """Make backup of human_config for Undo."""
        self.human_config_backup.append(deepcopy(self.human_config))
        self.trajectory_manager.backup()