import pandas as pd

from acme_dataset import filter_by_tags, load_scenario_tags, tag_counts, tags_for
from acme_dataset.download import allow_patterns
from acme_dataset.human_readable import load_utterances
from acme_dataset.tags import trajectory_key


def _write_tags(path):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"trajectory_name": ["a_merged_0", "b_merged_0"], "dataset_name": "go2nus",
                      "tags": ["['open_area', 'with_traffic']", "['narrow_corridors']"]}
                     ).to_excel(writer, sheet_name="go2nus", index=False)
        pd.DataFrame({"trajectory_name": ["c_1"], "dataset_name": "uex", "tags": ["['open_area']"],
                      "problems and observations": ["odometry fault"]}
                     ).to_excel(writer, sheet_name="uex", index=False)


def test_tags(tmp_path):
    path = tmp_path / "acme_tags.xlsx"
    _write_tags(path)
    tags = load_scenario_tags(path)
    assert tags["institution"].tolist() == ["go2nus", "go2nus", "uex"]
    assert tags["tags"].iloc[0] == ["open_area", "with_traffic"]
    assert "problems_and_observations" in tags.columns
    assert tag_counts(tags).loc["open_area", "trajectories"] == 2
    assert filter_by_tags(tags, all_of=["open_area"], none_of=["with_traffic"])["trajectory_name"].tolist() == ["c_1"]
    assert filter_by_tags(tags, any_of=["narrow_corridors", "x"], institution="go2nus").shape[0] == 1
    assert tags_for(tags, "a_merged") == ["open_area", "with_traffic"]   # bag name
    assert tags_for(tags, "c_1_0") == ["open_area"]                      # human-readable folder name
    assert tags_for(tags, "zzz") == []


def test_trajectory_key():
    # tag-sheet name, bag name and sampled folder name of the same trajectory
    assert trajectory_key("engr_elev_1_10") == trajectory_key("engr_elev_1_10_filtered") \
        == trajectory_key("engr_elev_1_10_merged_0")
    assert trajectory_key("cabot_x_ros2bag_11") == trajectory_key("cabot_x_ros2bag_11_merged_0_trimmed_lio_0")
    assert trajectory_key("r_08_21_51_mer_r_08_21_51_mer_3") == trajectory_key("r_08_21_51_merged_3_0")
    assert trajectory_key("03-24-2025-3-Keio_Entrance_West_47") \
        == trajectory_key("03-24-2025-3-Keio_Entrance_West_0_1_47") \
        == trajectory_key("03-24-2025-3-Keio_Entrance_West_0_1_47_0")
    assert trajectory_key("a_1") != trajectory_key("a_10")


def test_utterances(tmp_path):
    path = tmp_path / "keio_utterances.csv"
    path.write_text("utterance,trajectory,timestamps\n"
                    "Excuse Me,traj_28,[1742786484.67]\n"
                    "Excuse Me,traj_47,\"[1742791503.27, 1742791505.0]\"\n")
    df = load_utterances(path)
    assert df.columns.tolist() == ["institution", "trajectory", "utterance", "timestamp"]
    assert len(df) == 3 and set(df["institution"]) == {"keio"}
    assert len(load_utterances(tmp_path)) == 3


def test_allow_patterns():
    assert allow_patterns("tags") == ["acme_tags.xlsx"]
    assert allow_patterns("raw", "miraikan") == ["ROS2_Bags/miraikan/*"]
    assert allow_patterns("human_readable", names=["t"]) == ["sampled_data/*/t/*", "sampled_data/*/t.zip"]
    assert allow_patterns("calibration", "keio") == ["sampled_data/extrinsics_intrinsics/keio/*"]
    assert allow_patterns("utterances") == ["sampled_data/utterances/*_utterances.csv"]
    assert allow_patterns("bev") == ["BEV/*"]
    assert allow_patterns("bev_annotations") == ["BEV/BEV_annotations/*"]
    assert "BEV/BEV_videos/cmu/*" in allow_patterns("bev_videos", "cmu")
