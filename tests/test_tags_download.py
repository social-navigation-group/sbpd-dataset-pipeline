import pandas as pd

from acme_dataset import filter_by_tags, load_scenario_tags, tag_counts, tags_for
from acme_dataset.download import allow_patterns


def _write_tags(path):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"trajectory_name": ["a_merged_0", "b_merged_0"], "dataset_name": "go2nus",
                      "tags": ["['open_area', 'with_traffic']", "['narrow_corridors']"]}
                     ).to_excel(writer, sheet_name="go2nus", index=False)
        pd.DataFrame({"trajectory_name": ["c_1"], "dataset_name": "uex", "tags": ["['open_area']"],
                      "problems and observations": ["odometry fault"]}
                     ).to_excel(writer, sheet_name="uex", index=False)


def test_tags(tmp_path):
    path = tmp_path / "dataset_tags.xlsx"
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


def test_allow_patterns():
    assert allow_patterns("tags") == ["dataset_tags.xlsx"]
    assert allow_patterns("raw", "miraikan") == ["ros2 bags/miraikan/*"]
    assert allow_patterns("human_readable", names=["t"]) == ["human_readable_sampled/*/t/*",
                                                              "human_readable_sampled/*/t.zip"]
    assert allow_patterns("bev") == ["bev/*"]
