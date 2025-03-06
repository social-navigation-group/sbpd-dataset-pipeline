from PyQt6.QtCore import QSettings
from .logging_utils import log_debug
from PyQt6.QtWidgets import QMessageBox, QCheckBox, QVBoxLayout, QWidget

class PersistentMessageBox(QMessageBox):
    SETTINGS_GROUP = "PersistentMessages"
    
    def __init__(self, parent = None, message_id = "default", title = "Information", text = "", icon = QMessageBox.Icon.Information):
        super().__init__(parent)
        
        self.settings = QSettings("Miraikan", "Labeling_Tool_App")
        self.message_id = message_id

        log_debug(f"Checking setting for {self.message_id}: {self.settings.value(f'{self.SETTINGS_GROUP}/{self.message_id}', 'false')}")
        
        if self.settings.value(f"{self.SETTINGS_GROUP}/{self.message_id}", "false") == "true":
            log_debug(f"Skipping message: {self.message_id}")
            return

        self.setWindowTitle(title)
        self.setText(text)
        self.setIcon(icon)
        
        self.checkbox = QCheckBox("Do not show this message again")
        
        custom_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        custom_widget.setLayout(layout)
        self.setCheckBox(self.checkbox)
        
        self.addButton(QMessageBox.StandardButton.Ok)
        
        result = self.exec()
        
        if result == QMessageBox.StandardButton.Ok and self.checkbox.isChecked():
            self.settings.setValue(f"{self.SETTINGS_GROUP}/{self.message_id}", "true")
            log_debug(f"Saved setting: {self.SETTINGS_GROUP}/{self.message_id} -> true")

    @staticmethod
    def show_message(parent, message_id, title, text, icon = QMessageBox.Icon.Information):
        PersistentMessageBox(parent, message_id, title, text, icon)
