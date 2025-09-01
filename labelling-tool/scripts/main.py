import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_ROOT = ''
sys.path.append(PROJECT_ROOT)

from ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from utils.resource_manager import ResourceManager

if __name__ == "__main__":
    app = QApplication(sys.argv)
    resource_manager = ResourceManager(PROJECT_ROOT,VIDEO_ROOT)
    window = MainWindow(resource_manager)
    window.show()
    sys.exit(app.exec())