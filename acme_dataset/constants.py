"""Names and layout constants for the ACME dataset."""

HF_REPO_ID = "acmesocnavdataset/ACME-dataset"
HF_REPO_TYPE = "dataset"

# Top-level folders of the dataset (identical on Hugging Face and on disk).
RAW_BAGS_DIR = "ros2 bags"
HUMAN_READABLE_DIR = "human_readable_sampled"
BEV_DIR = "bev"
BEV_ANNOTATIONS_DIR = "bev/bev_annotations"
BEV_VIDEOS_DIR = "bev/bev_videos"
TAGS_FILE = "dataset_tags.xlsx"

COMPONENTS = {
    "raw": RAW_BAGS_DIR,
    "human_readable": HUMAN_READABLE_DIR,
    "bev": BEV_DIR,
    "tags": TAGS_FILE,
}

# Institution folder name -> display name.
INSTITUTIONS = {
    "go2nus": "NUS",
    "miraikan": "Miraikan",
    "cmu": "CMU",
    "keio": "Keio",
    "umich": "UMich",
    "uex": "UEx",
    "gmu": "GMU",
    "ubonn": "UBonn",
}

# Human-readable data is uniformly sampled at this rate.
HUMAN_READABLE_RATE_HZ = 4.0

# BEV pedestrian annotations are recorded at this rate.
BEV_FPS = 10.0

# BEV semantic tag ids (see bev_semantic_annotation_tool/README.md).
BEV_TAGS = {
    1: "Open Area",
    2: "Wide Corridor",
    3: "Narrow Corridor",
    4: "Intersection",
    5: "Entry/Exit",
    6: "Blind Corner",
    7: "Ignore",
}
BEV_IGNORE_TAG = 7
BEV_TAG_COLORS = {
    1: "#FFD700",
    2: "#00B4FF",
    3: "#FF5050",
    4: "#00DC50",
    5: "#DC00DC",
    6: "#FF8C00",
    7: "#A0A0A0",
}
