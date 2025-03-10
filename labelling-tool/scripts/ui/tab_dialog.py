from .human_controls import HumanControls
from .trajectory_controls import TrajectoryControls
from PyQt6.QtWidgets import QDialog, QWidget, QTabWidget, QVBoxLayout

class TabDialog(QDialog):
    def __init__(self, video_controls, parent: QWidget = None):
        super().__init__(parent)

        self.video_controls = video_controls
        self.video_player = self.video_controls.get_video_player()

        self.trajectory_controls = TrajectoryControls(self.video_player, self)
        # self.human_controls = HumanControls(self.video_player, self)

        tab_widget = QTabWidget()
        tab_widget.addTab(self.trajectory_controls, "Trajectory")
        # tab_widget.addTab(self.human_controls, "Human")

        main_layout = QVBoxLayout()
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
