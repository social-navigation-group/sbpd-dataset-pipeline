from utils.logging_utils import log_info
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox, QLineEdit, QMessageBox, QLabel
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal
from copy import deepcopy

class HumanControls(QWidget):
    prev_info_updated = pyqtSignal(int)
    temp_info_updated = pyqtSignal(int)
    
    def __init__(self, video_player, parent: QWidget):
        super().__init__(parent)
        self.video_player = video_player
        self.trajectory_manager = self.video_player.trajectory_manager

        # SET TESTING IN THE LAYOUT
        self.human_controls = self.create_human_controls()
        self.setLayout(self.human_controls)
        
        self.human_config = self.video_player.human_config
        
        self.labeling_idx = 0
        self.adding_context_mode = False
        
        self.human_config_backup = []
        
        self.prev_info_updated.connect(self.on_prev_updated)
        
    def set_start_frame(self):
        labeling_ID = self.human_config.used_indices[self.labeling_idx]
        startFrame = self.human_config.get_element(labeling_ID, "traj_start")
        
        self.video_player.show_frame_at(startFrame + 1)
        self.trajectory_manager.set_selected_trajectory(labeling_ID)
    
    def set_context_value(self, labeling_ID, human_labels):
        self.human_config_backup.append(deepcopy(self.human_config))
        self.human_config.set_element(labeling_ID, "human_context", human_labels)
        return
    
    def set_multiple_label(self, labeling_ID, clicked_label):
        human_labels = self.human_config.get_element(labeling_ID, "human_context")
        
        if type(human_labels) == list:
            if clicked_label not in human_labels:
                human_labels.append(clicked_label)
            else:
                print(f"\"{clicked_label}\": Skip adding.")
        else: #str
            human_labels = [clicked_label]
        
        self.set_context_value(labeling_ID, human_labels)

        return
        
    def create_human_controls(self):
        human_context_group = QGroupBox("Human Context: ")
        human_context_layout = QVBoxLayout()
        
        human_context_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.startIDinput = QLineEdit()
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self.on_start_button_clicked)
        
        self.startIDinput.setEnabled(False)
        self.startButton.setEnabled(False)
        
        human_context_layout.addWidget(self.startIDinput)
        human_context_layout.addWidget(self.startButton)
        
        self.buttons = []
        for label in ["Strollers", "Children", "Adults", "Elderly", "Wheelchairs", "Blind"]:
            button = RightClickHandledButton(label)
            button.clicked.connect(self.on_label_button_clicked)
            button.right_clicked.connect(self.on_right_button_clicked)
            human_context_layout.addWidget(button)
            button.setEnabled(False)
            self.buttons.append(button)
        
        undo_button = QPushButton('Undo')

        human_context_group.setLayout(human_context_layout)

        labeling_layout = QVBoxLayout()
        labeling_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        labeling_layout.addWidget(human_context_group)
        
        return labeling_layout
    
    def get_clicked_label(self):
        clicked_button = self.sender()
        if clicked_button:
            clicked_label = clicked_button.text()
            log_info(f"Button \"{clicked_label}\" pushed.")
            return clicked_label
        else:
            log_info("None of buttons are pushed.")
            return None
    
    def on_start_button_clicked(self):
        try:
            input_ID = int(self.startIDinput.text())
            if input_ID == 0:
                self.labeling_idx = 0
            elif input_ID - 1 in self.human_config.used_indices:
                self.labeling_idx = self.human_config.used_indices.index(input_ID - 1)
            else:
                raise ValueError(f"invalid trajectory ID ({input_ID})\n")
            
            self.set_start_frame()
            self.video_player.stop()
            self.video_player.play()
            self.prev_info_label = QLabel("[PREV]: ")
            self.temp_info = QLabel(f'[NOW]: No.{input_ID}')
            
            self.human_controls.insertWidget(0, self.prev_info_label)
            self.human_controls.insertWidget(1, self.temp_info)
            
            self.startIDinput.deleteLater()
            self.startButton.deleteLater()
            
            for btn in self.buttons:
                btn.setEnabled(True)
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", f"Please input a valid trajectory ID.\n{ValueError}")
        
        except Exception as e:
            print(e)
            
    def on_label_button_clicked(self):
        self.video_player.stop()
        clicked_label = self.get_clicked_label()

        if clicked_label is not None:
            labeling_ID = self.human_config.used_indices[self.labeling_idx]
            
            if not self.adding_context_mode:
                log_info(f'Human ID {labeling_ID + 1}: {clicked_label}')
                self.set_context_value(labeling_ID, clicked_label)
            else:
                log_info(f'Human ID {labeling_ID + 1}: {clicked_label}')
                self.set_multiple_label(labeling_ID, clicked_label)
                self.adding_context_mode = False
            
            self.trajectory_manager.clear_selection()
            self.labeling_idx += 1
            
            if self.labeling_idx == len(self.human_config):
                log_info('Labeling ended.')
                self.human_config.save_human_config('temp.toml')
                exit()
            
            self.set_start_frame()
            self.video_player.play()
        
    def on_right_button_clicked(self):
        log_info('right_clicked.')
        clicked_label = self.get_clicked_label()
        
        if clicked_label is not None:
            labeling_ID = self.human_config.used_indices[self.labeling_idx]
            log_info(f'Human ID {labeling_ID + 1}: {clicked_label} <- Still labeling.')
            self.set_multiple_label(labeling_ID, clicked_label)
            self.adding_context_mode = True
    
    def on_prev_updated(self, prev_idx):
        labeling_ID = self.human_config.used_indices[prev_idx]
        human_labels = self.human_config.get_element(labeling_ID, "human_context")
        self.prev_info_label = QLabel(f"[PREV]: No.{labeling_ID}-{human_labels}")
        return
    
    
    
    
class RightClickHandledButton(QPushButton):
    right_clicked = pyqtSignal(bool)
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(1)

        else:
            super().mousePressEvent(event)