from utils.logging_utils import log_info, log_debug
from PyQt6.QtCore import pyqtSignal, QObject
from copy import deepcopy

class TrajectoryManager(QObject):
    drawingFinished = pyqtSignal(bool)
    ID_selected = pyqtSignal(bool)
    updateFrame = pyqtSignal(int)
    
    def __init__(self, human_config, parent = None):
        super().__init__(parent)
        UNDO_LIMIT = 10
        self.frame_skip = 10
        self.traj_starts = {}
        self.trajectories = {}
        self.selected_trajs = []
        self.human_config = human_config

        self.isWaitingID = False
        self.isDrawing = False
        
        self.trajectory_now = []
        self.backup_data = []
        self.backup_limit = UNDO_LIMIT

    def set_trajectories(self):
        """Assigns trajectories using stored IDs"""
        self.traj_starts = {}
        self.trajectories = {}

        for humanID in self.human_config.used_indices:
            traj = self.human_config.get_element(humanID, "trajectories")
            start = self.human_config.get_element(humanID, "traj_start")

            if traj is not None and start is not None:
                traj_id = humanID
                self.trajectories[traj_id] = traj
                self.traj_starts[traj_id] = start

    def add_trajectory(self, new_trajectory, start_frame):
        traj_id = self.human_config.newID_init()
        self.trajectories[traj_id] = new_trajectory
        self.traj_starts[traj_id] = start_frame

        self.human_config.set_element(traj_id, "trajectories", new_trajectory)
        self.human_config.set_element(traj_id, "traj_start", start_frame)
        self.human_config.set_element(traj_id, "human_context", "Adult")
        return traj_id
    
    def remove_trajectory(self, traj_id):
        if traj_id in self.trajectories:
            del self.trajectories[traj_id]
            del self.traj_starts[traj_id]
            self.human_config.delete_ID(traj_id)

    def get_active_trajectories(self, current_frame):
        active_trajectories = []

        for traj_id, traj_start in self.traj_starts.items():
            if traj_start <= current_frame < traj_start + len(self.trajectories[traj_id]):  
                active_trajectories.append(traj_id)

        return active_trajectories

    def set_selected_trajectory(self, selected_traj_id, selected_frame = None):
        if selected_traj_id not in [info[0] for info in self.selected_trajs]:
            self.selected_trajs.append((selected_traj_id, selected_frame))
            log_debug(f"Selected trajectories: {self.selected_trajs}")

    def get_selected_trajectory(self):
        return list(self.selected_trajs)

    def clear_selection(self):
        self.selected_trajs = []
        # NOTE: Possibly not clean but this function is called everywhere (all the buttons)
        self.human_config.autosave()
        
    def set_newValues(self, traj_id, traj_start, trajectories):
        self.traj_starts[traj_id] = traj_start
        self.trajectories[traj_id] = trajectories
        
        self.human_config.set_element(traj_id, "traj_start", traj_start)
        self.human_config.set_element(traj_id, "trajectories", trajectories)
        self.human_config.set_element(traj_id, "human_context", "Adult")
    
    def store_newTrajectory(self, new_x, new_y):
        if not self.trajectory_now:
            self.trajectory_now.append([new_x, new_y])
        else:
            (last_x, last_y) = self.trajectory_now[-1]
            for i in range(self.frame_skip):
                temp_x = last_x + (new_x - last_x) / self.frame_skip * (i + 1)
                temp_y = last_y + (new_y - last_y) / self.frame_skip * (i + 1)
                self.trajectory_now.append([temp_x, temp_y])
        return

    def backup(self):
        if len(self.backup_data) >= self.backup_limit:
            self.backup_data.pop(0)
        self.backup_data.append([deepcopy(self.traj_starts), deepcopy(self.trajectories)])
    
    def undo(self):
        if len(self.backup_data) == 0:
            log_info("No more backup data to undo.")
        (traj_starts_old, trajectories_old) = self.backup_data.pop(-1)
        self.traj_starts = traj_starts_old
        self.trajectories = trajectories_old