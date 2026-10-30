"""Scenario tags of the robot trajectories (``dataset_tags.xlsx``).

The spreadsheet has one sheet per institution with the columns
``trajectory_name, dataset_name, tags`` (plus free-text notes for some
institutions). ``tags`` holds location tags (e.g. ``narrow_corridors``,
``intersections``) and pedestrian tags (e.g. ``against_traffic``,
``conversational_groups``).
"""

import ast
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
    """Load the tag table from ``dataset_tags.xlsx`` (all sheets) or a CSV export.

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


def tags_for(tags: pd.DataFrame, trajectory_name: str) -> List[str]:
    """Tags of one trajectory. Accepts bag names and human-readable folder names
    (which carry an extra ``_0`` suffix)."""
    names = {trajectory_name, f"{trajectory_name}_0"}
    if trajectory_name.endswith("_0"):
        names.add(trajectory_name[:-2])
    match = tags[tags["trajectory_name"].isin(names)]
    return list(match.iloc[0]["tags"]) if len(match) else []
