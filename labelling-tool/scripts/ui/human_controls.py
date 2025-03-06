from utils.logging_utils import log_info
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox

class HumanControls(QWidget):
    def __init__(self, video_player, parent: QWidget):
        super().__init__(parent)
        self.video_player = video_player

        # SET TESTING IN THE LAYOUT
        self.setLayout(self.create_human_controls())
        
        self.traj_starts, self.trajectories = self.load_trajectories_dummy()
        self.numPerson = len(self.traj_starts)
        self.labelling_now = 0
        
        self.humanLabel = ['Adults' for _ in range(self.numPerson)]
        
    def create_human_controls(self):
        human_context_group = QGroupBox("Human Context: ")
        human_context_layout = QVBoxLayout()
        
        for label in ["Strollers", "Children", "Adults", "Elderly", "Wheelchairs", "Blind"]:
            button = QPushButton(label)
            button.clicked.connect(self.on_label_button_clicked)
            human_context_layout.addWidget(button)

        human_context_group.setLayout(human_context_layout)

        labeling_layout = QVBoxLayout()
        labeling_layout.addWidget(human_context_group)
        
        return labeling_layout
            
    def on_label_button_clicked(self):
        clicked_button = self.sender()
        
        if clicked_button is not None:
            clicked_label = clicked_button.text()
            log_info(f'Human ID {self.labelling_now + 1}: {clicked_label}')
            self.humanLabel[self.labelling_now] = clicked_label
            self.labelling_now += 1
    
    def load_trajectories_dummy(self):
        traj_starts = [i * 10 for i in range(10)]
        trajectories = [[100 + i, 200 + i] for i in range(30)]
        trajectories = [trajectories for _ in range(10)]
        
        return traj_starts, trajectories