import os

def ensure_directory_exists(directory: str):
    """Ensures that the given directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)

def get_file_size(file_path: str) -> str:
    """Returns the file size in a human-readable format."""
    if not os.path.exists(file_path):
        return "File not found"
    
    size_in_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

    return f"{size_in_bytes:.2f} PB"

def is_valid_video_file(file_path: str) -> bool:
    """Checks if a given file path points to a valid video file."""
    valid_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in valid_extensions

def list_video_files(directory: str):
    """Lists all valid video files in a given directory."""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if is_valid_video_file(os.path.join(directory, f))]
