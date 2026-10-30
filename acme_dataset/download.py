"""Download parts of the ACME dataset from Hugging Face.

The Hugging Face repository mirrors the on-disk layout::

    ros2 bags/<institution>/<bag>.zip
    human_readable_sampled/<institution>/<trajectory>/...
    bev/bev_annotations/...  bev/bev_videos/<institution>/...
    dataset_tags.xlsx

so a download into ``local_dir`` gives the same tree the loaders expect.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from .constants import BEV_DIR, COMPONENTS, HF_REPO_ID, HF_REPO_TYPE, TAGS_FILE

PathLike = Union[str, Path]


def _component_dir(component: str) -> str:
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {sorted(COMPONENTS)}, got {component!r}")
    return COMPONENTS[component]


def allow_patterns(component: str, institution: Optional[str] = None,
                   names: Optional[Iterable[str]] = None) -> List[str]:
    """Hugging Face ``allow_patterns`` selecting a component, institution and/or items.

    ``names`` are bag names (raw), trajectory folder names (human_readable)
    or BEV dataset/session name prefixes (bev).
    """
    base = _component_dir(component)
    if component == "tags":
        return [TAGS_FILE]
    names = list(names or [])
    inst = institution or "*"
    if component == "bev":
        if not names and institution is None:
            return [f"{BEV_DIR}/*"]
        keys = names or [institution]
        return [p for key in keys for p in (f"{BEV_DIR}/*/{key}/*", f"{BEV_DIR}/*/{key}_*", f"{BEV_DIR}/*.ipynb")]
    if not names:
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
    extract: bool = False,
    max_workers: int = 8,
) -> Path:
    """Download (part of) a component into ``local_dir`` and return ``local_dir``.

    Args:
        component: ``raw``, ``human_readable``, ``bev`` or ``tags``.
        institution: Restrict to one institution folder (e.g. ``go2nus``).
        names: Restrict to specific bags / trajectories / BEV datasets.
        local_dir: Destination; the repository layout is kept below it.
        extract: Unzip downloaded ``.zip`` archives (e.g. raw bags) in place.

    Example:
        >>> download("human_readable", "go2nus", ["deck_1_without_overhead_2025-03-26_12-05-52_3_corrected_merged_0"])
        >>> download("raw", "miraikan", extract=True)
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
    if extract:
        extract_archives(local_dir / _component_dir(component))
    return local_dir


def extract_archives(root: PathLike) -> List[Path]:
    """Extract every ``.zip`` below ``root`` next to itself (skips already extracted bags)."""
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
            with zipfile.ZipFile(archive) as zf:
                target = archive.parent.resolve()
                if any(not (target / n).resolve().is_relative_to(target) for n in zf.namelist()):
                    raise ValueError(f"unsafe path in archive {archive}")
                zf.extractall(archive.parent)
            out.append(archive.with_suffix(""))
    return out

