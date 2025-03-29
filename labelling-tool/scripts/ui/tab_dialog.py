from .human_controls import HumanControls
from .trajectory_controls import TrajectoryControls
from PyQt6.QtWidgets import QDialog, QWidget, QTabWidget, QVBoxLayout

class TabDialog(QDialog):
    def __init__(self, video_controls, parent: QWidget = None):
        super().__init__(parent)

        self.video_controls = video_controls
        self.video_player = self.video_controls.get_video_player()

        self.trajectory_controls = TrajectoryControls(self.video_player, self)
        self.trajectory_controls.time_window_changed.connect(self.on_time_window_changed)
        self.trajectory_controls.time_window_enabled_changed.connect(self.on_time_window_enabled_changed)

        # self.human_controls = HumanControls(self.video_player, self)

        tab_widget = QTabWidget()
        tab_widget.addTab(self.trajectory_controls, "Trajectory")
        # tab_widget.addTab(self.human_controls, "Human")

        main_layout = QVBoxLayout()
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

    def on_time_window_changed(self):
        if hasattr(self.video_player, "trajectory_worker"):
            self.video_player.trajectory_worker.overlay_cache.clear()
            self.video_player.trajectory_worker.update_frame(self.video_player.current_frame)
    
    def on_time_window_enabled_changed(self, enabled: bool):
        self.video_player.trajectory_worker.set_time_window_enabled(enabled)