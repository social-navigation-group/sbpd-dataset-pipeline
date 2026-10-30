"""Scenario tags of the robot trajectories (``acme_tags.xlsx``).

The spreadsheet has one sheet per institution with the columns
``trajectory_name, dataset_name, tags`` (plus free-text notes for some
institutions). ``tags`` holds location tags (e.g. ``narrow_corridors``,
``intersections``) and pedestrian tags (e.g. ``against_traffic``,
``conversational_groups``).

Trajectory names in the spreadsheet do not always match the bag / sampled
file names exactly (suffixes such as ``_merged``, ``_filtered`` or ``_0``
differ); :func:`trajectory_key` normalizes both sides.
"""

import ast
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _parse_tags(value) -> List[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value.split(",")
    return [str(t).strip() for t in parsed if str(t).strip()]


def load_scenario_tags(path: PathLike) -> pd.DataFrame:
    """Load the tag table from ``acme_tags.xlsx`` (all sheets) or a CSV export.

    Returns a DataFrame with columns ``institution, trajectory_name, tags``
    (``tags`` is a list of strings) followed by any extra note columns.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        frames = []
        for sheet, df in sheets.items():
            df = df.copy()
            df.insert(0, "institution", sheet)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "institution" not in df.columns:
        df.insert(0, "institution", df.get("dataset_name"))
    df = df.dropna(subset=["trajectory_name"])
    df["trajectory_name"] = df["trajectory_name"].astype(str).str.strip()
    df["tags"] = df["tags"].map(_parse_tags)
    df = df.drop(columns=[c for c in ("dataset_name",) if c in df.columns])
    return df.reset_index(drop=True)


def tag_counts(tags: pd.DataFrame, by_institution: bool = False) -> pd.DataFrame:
    """Number of trajectories per tag (optionally per institution)."""
    exploded = tags.explode("tags").dropna(subset=["tags"])
    if by_institution:
        return exploded.groupby(["tags", "institution"]).size().unstack(fill_value=0)
    return exploded["tags"].value_counts().rename_axis("tag").to_frame("trajectories")


def filter_by_tags(
    tags: pd.DataFrame,
    all_of: Optional[Iterable[str]] = None,
    any_of: Optional[Iterable[str]] = None,
    none_of: Optional[Iterable[str]] = None,
    institution: Optional[str] = None,
) -> pd.DataFrame:
    """Rows whose tags contain all of ``all_of``, at least one of ``any_of`` and none of ``none_of``."""
    all_of, any_of, none_of = set(all_of or ()), set(any_of or ()), set(none_of or ())
    keep = tags["tags"].map(lambda t: all_of <= set(t)
                            and (not any_of or bool(any_of & set(t)))
                            and not (none_of & set(t)))
    if institution is not None:
        keep &= tags["institution"] == institution
    return tags[keep]


def trajectory_key(name: str) -> str:
    """Normalized trajectory name shared by tag rows, bag names and sampled folder names.

    Drops the ``_0`` suffix of sampled folders and processing suffixes such as
    ``_merged``, ``_merged_lio``, ``_merged_0_trimmed_lio`` and ``_filtered``,
    rewrites the UMich tag style ``<x>_mer_<x>_mer_<n>`` to ``<x>_merged_<n>``,
    and drops the ``_0_1`` recording infix of Keio bags (``<x>_0_1_<n>`` is tagged as ``<x>_<n>``).
    """
    match = re.match(r"^(.+)_mer_\1_mer_(\d+)$", name)
    if match:
        name = f"{match.group(1)}_merged_{match.group(2)}"
    name = re.sub(r"_0$", "", name)
    name = re.sub(r"_merged(_0)?(_trimmed)?(_lio)?$", "", name)
    name = re.sub(r"_(filtered|corrected_merged|merged_corrected)$", "", name)
    if "-Keio_" in name:
        name = re.sub(r"_0_1_(\d+)$", r"_\1", name)
    return name


def tags_for(tags: pd.DataFrame, trajectory_name: str) -> List[str]:
    """Tags of one trajectory, given a tag-sheet name, bag name or sampled folder name.
    Returns an empty list if no row matches."""
    exact = tags[tags["trajectory_name"] == trajectory_name]
    if len(exact):
        return list(exact.iloc[0]["tags"])
    key = trajectory_key(trajectory_name)
    match = tags[tags["trajectory_name"].map(trajectory_key) == key]
    return list(match.iloc[0]["tags"]) if len(match) else []
