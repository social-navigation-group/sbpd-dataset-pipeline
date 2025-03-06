import os
from .logging_utils import log_info, log_warning

class ResourceManager:
    """Handles resource file paths dynamically"""

    def __init__(self, base_path: str):
        self.video_name = ""
        self.base_path = base_path

        self.resources_path = os.path.join(self.base_path, "resources")
        self.icon_path = os.path.join(self.resources_path, "icons")
        self.video_path = os.path.join(self.resources_path, "videos")
        self.config_path = os.path.join(self.resources_path, "config")

    def get_icon(self, sub_folder: str, name: str):
        """Returns the path of an icon."""
        path = os.path.join(self.icon_path, sub_folder, f"{name}.png")

        if os.path.exists(path):
            log_info(f"Loaded {sub_folder} icon: {path}")
            return path
        else:
            log_warning(f"{sub_folder.capitalize()} icon {path} not found.")
            return None 

    def get_stylesheet(self):
        """Returns the full path of the QSS stylesheet."""
        return os.path.join(self.resources_path, "styles.qss")

    def get_video(self, name):
        """Returns the path of the videos directory."""
        self.video_name = name
        return os.path.join(self.video_path, f"{name}")