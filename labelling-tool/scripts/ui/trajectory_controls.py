from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit
)

from video_proc_comps.button_controller import ButtonController

class TrajectoryControls(QWidget):
    def __init__(self, video_player, parent: QWidget):
        super().__init__(parent)
        self.video_player = video_player
        
        self.buttonController = ButtonController(self)

        # SET THE CONTROLS IN THE LAYOUT
        self.labeling_layout = self.create_labeling_controls()
        self.setLayout(self.labeling_layout)
        
        self.trajID_input_visible = False

    def create_labeling_controls(self):
        """Creates and returns the labeling control UI."""
        labeling_layout = QVBoxLayout()

        labels = ["Relabel", "Missing", "Break", "Join", "Delete", "Disentangle", "Undo"]
        buttons = [QPushButton(label) for label in labels]
        for label, button in zip(labels, buttons):
            button.clicked.connect(getattr(self.buttonController, f"on_{label}_clicked"))
            labeling_layout.addWidget(button)

        self.log_labeling = QTextEdit()
        self.log_labeling.setPlaceholderText("Log the labeling here if needed...")
        self.log_submit_button = QPushButton()
        self.log_submit_button.setText("Submit")

        labeling_layout.addWidget(self.log_labeling)
        labeling_layout.addWidget(self.log_submit_button)

        return labeling_layout
    
    def create_trajID_input(self, labeling_layout, number):
        """Add QLineEdit and Select Button."""
        # delete QLineEdit and Select Button for init.
        self.delete_trajID_input(self.labeling_layout)
        labeling_layout.insertWidget(0, QLabel(f"Trajectory {number}"))
        trajectory_input = QLineEdit()
        labeling_layout.insertWidget(1, trajectory_input)
            
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.buttonController.on_select_pressed)
        labeling_layout.insertWidget(2, self.select_button)
        
        self.trajID_input_visible = True

        return labeling_layout
    
    def delete_trajID_input(self, labeling_layout):
        """Delete QLineEdit and Select Button."""
        if self.trajID_input_visible is False:
            return labeling_layout
        for i in range(3):
            item = labeling_layout.itemAt(0)
            
            if item is not None:
                widget = item.widget()
                labeling_layout.removeWidget(widget)
                widget.deleteLater()
                
        self.trajID_input_visible = False

        return labeling_layout