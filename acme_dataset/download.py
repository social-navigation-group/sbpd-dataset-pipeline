"""Download parts of the ACME dataset from Hugging Face.

Layout of the Hugging Face repository::

    ROS2_Bags/<institution>/<bag>.zip
    sampled_data/<institution>/<trajectory>.zip     (one human-readable folder per zip)
    sampled_data/extrinsics_intrinsics/<institution>/...
    sampled_data/utterances/<institution>_utterances.csv
    BEV/BEV_annotations/...  BEV/BEV_videos/<institution>/...
    acme_tags.xlsx

A download into ``local_dir`` keeps this layout. Human-readable zips are
extracted next to themselves so the loaders find the trajectory folders.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from .constants import BEV_ANNOTATIONS_DIR, COMPONENTS, HF_REPO_ID, HF_REPO_TYPE, TAGS_FILE, UTTERANCES_DIR

PathLike = Union[str, Path]


def _component_dir(component: str) -> str:
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {sorted(COMPONENTS)}, got {component!r}")
    return COMPONENTS[component]


def allow_patterns(component: str, institution: Optional[str] = None,
                   names: Optional[Iterable[str]] = None) -> List[str]:
    """Hugging Face ``allow_patterns`` selecting a component, institution and/or items.

    ``names`` are bag names (raw), trajectory names (human_readable) or BEV
    dataset names (bev: ``bonn``, ``cmu``, ``gmu``, ``keio``, ``nus``,
    ``short-term``, ``umich`` for annotations; institution names for videos).
    """
    base = _component_dir(component)
    if component == "tags":
        return [TAGS_FILE]
    names = list(names or [])
    inst = institution or "*"
    if component == "utterances":
        return [f"{UTTERANCES_DIR}/{inst}_utterances.csv"]
    if component in ("bev", "bev_annotations", "bev_videos"):
        if not names and institution is None:
            return [f"{base}/*"]
        keys = names or [institution]
        patterns = [p for key in keys
                    for p in (f"{base}/{key}/*", f"{base}/*/{key}/*", f"{base}/{key}_*", f"{base}/*/{key}_*")]
        if component != "bev_videos":
            patterns += [f"{BEV_ANNOTATIONS_DIR}/*.ipynb", f"{BEV_ANNOTATIONS_DIR}/*.py",
                         f"{BEV_ANNOTATIONS_DIR}/requirements.txt"]
        return patterns
    if component == "calibration" or not names:
        return [f"{base}/{inst}/*"]
    return [p for name in names for p in (f"{base}/{inst}/{name}/*", f"{base}/{inst}/{name}.zip")]


def list_remote(component: str, institution: Optional[str] = None,
                repo_id: str = HF_REPO_ID, revision: Optional[str] = None) -> List[str]:
    """Names available on Hugging Face without downloading anything.

    Without ``institution`` this lists the institutions of a component;
    with it, the bags / trajectories / files of that institution.
    """
    from huggingface_hub import HfApi

    path = _component_dir(component)
    if institution is not None:
        path = f"{path}/{institution}"
    entries = HfApi().list_repo_tree(repo_id, path_in_repo=path, repo_type=HF_REPO_TYPE, revision=revision)
    names = [Path(e.path).name for e in entries]
    return sorted(n[:-4] if n.endswith(".zip") else n for n in names)


def download(
    component: str = "human_readable",
    institution: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
    local_dir: PathLike = "ACME-dataset",
    repo_id: str = HF_REPO_ID,
    revision: Optional[str] = None,
    extract: Optional[bool] = None,
    max_workers: int = 8,
) -> Path:
    """Download (part of) a component into ``local_dir`` and return ``local_dir``.

    Args:
        component: ``raw``, ``human_readable``, ``calibration``, ``utterances``,
            ``bev`` (``bev_annotations`` + ``bev_videos``) or ``tags``.
        institution: Restrict to one institution folder (e.g. ``miraikan``).
        names: Restrict to specific bags / trajectories / BEV datasets.
        local_dir: Destination; the repository layout is kept below it.
        extract: Unzip downloaded ``.zip`` archives in place. Defaults to True
            for ``human_readable`` and False otherwise (bags are extracted
            on first use by :class:`acme_dataset.rosbag.Bag`).

    Example:
        >>> download("human_readable", "miraikan", ["03-08-2025-1-3F_Geo_Cosmos_0_0_10_0"])
        >>> download("raw", "miraikan", ["03-08-2025-1-3F_Geo_Cosmos_0_0_10"])
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    snapshot_download(
        repo_id,
        repo_type=HF_REPO_TYPE,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns(component, institution, names),
        max_workers=max_workers,
    )
    if extract is None:
        extract = component == "human_readable"
    if extract:
        extract_archives(local_dir / _component_dir(component))
    return local_dir


def extract_archives(root: PathLike) -> List[Path]:
    """Extract every ``.zip`` below ``root`` next to itself (skips already extracted archives)."""
    import zipfile

    from .rosbag import extract_bag

    out = []
    root = Path(root)
    if root.is_file():
        return out
    for archive in sorted(root.rglob("*.zip")):
        try:
            out.append(extract_bag(archive))
        except ValueError:  # not a bag: plain archive
            if archive.with_suffix("").is_dir():
                out.append(archive.with_suffix(""))
                continue
            with zipfile.ZipFile(archive) as zf:
                target = archive.parent.resolve()
                if any(not (target / n).resolve().is_relative_to(target) for n in zf.namelist()):
                    raise ValueError(f"unsafe path in archive {archive}")
                zf.extractall(archive.parent)
            out.append(archive.with_suffix(""))
    return out

